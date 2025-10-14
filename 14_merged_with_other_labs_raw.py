import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

# merge_other_labs_raw.py

# ---------- CONFIG ----------
BASE = Path(".")
MERGED_INITIAL_PATH = BASE / "admissions_expanded.csv"
OTHER_LABS_PATH = BASE / "other_labs_raw.csv"

TMP_AGG_PATH = BASE / "tmp_other_labs_partial.csv"   # append-only intermediate
OUT_PATH = BASE / "merged_with_other_labs_raw.csv"

SRC_CHUNKSIZE = 200_000   # reading other_labs_raw in chunks
WRITE_CHUNK = 20_000      # merging: number of rows from admissions_expanded per write
MATCH_CHUNK = 200_000     # when scanning tmp agg file for matches
# ---------------------------

def nowstr():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def parse_numeric(x):
    if pd.isna(x):
        return np.nan
    try:
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(',','')
        if s in ("", "NaN", "nan", "None", "none", "___"):
            return np.nan
        return float(s)
    except:
        return np.nan

def read_header(path: Path):
    if not path.exists():
        return []
    return pd.read_csv(path, nrows=0).columns.tolist()

def build_admit_map(merged_initial_path):
    print(f"[{nowstr()}] Building admit_map...")
    usecols = ['subject_id','hadm_id','admittime']
    mi = pd.read_csv(merged_initial_path, usecols=usecols, parse_dates=['admittime'], low_memory=False)
    mi['subject_id'] = pd.to_numeric(mi['subject_id'], errors='coerce').astype('Int64')
    mi['hadm_id'] = pd.to_numeric(mi['hadm_id'], errors='coerce').astype('Int64')
    adm = mi.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
    adm['admit_date'] = pd.to_datetime(adm['admit_time']).dt.normalize()
    d = dict(zip(zip(adm['subject_id'].astype(int), adm['hadm_id'].astype(int)), adm['admit_date']))
    print(f"[{nowstr()}] admit_map size: {len(d)}")
    del mi, adm; gc.collect()
    return d

def key_str(s,h,d):
    return f"{int(s)}|{int(h)}|{int(d)}"

def key_series_from_df(df):
    s = pd.to_numeric(df['subject_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    h = pd.to_numeric(df['hadm_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    d = pd.to_numeric(df['day_index'], errors='coerce').fillna(-1).astype(int).astype(str)
    return s + '|' + h + '|' + d

def collect_matching_rows_and_collapse(tmp_agg_path, keys_set, usecols=None, parse_dates=None, chunksize=MATCH_CHUNK):
    """
    Read tmp_agg_path in chunks, select rows whose (subject_id,hadm_id,day_index) in keys_set,
    then collapse duplicate partial-aggregates per key into one final row using rules:
      - labs_count: sum
      - labs_numeric_sum: sum
      - labs_numeric_count: sum
      - labs_numeric_max: max
      - labs_abnormal_count: sum
      - last_*: choose row with max last_charttime
    Returns dataframe (possibly empty) with unique keys.
    """
    if not tmp_agg_path.exists() or not keys_set:
        return pd.DataFrame(columns=(usecols or []))

    keys_s = set(key_str(s,h,d) for (s,h,d) in keys_set)
    matches = []
    cols_seen = None

    for chunk in pd.read_csv(tmp_agg_path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize, low_memory=True):
        if not {'subject_id','hadm_id','day_index'}.issubset(set(chunk.columns)):
            continue
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
        chunk['day_index'] = pd.to_numeric(chunk['day_index'], errors='coerce').astype('Int64')

        ks = key_series_from_df(chunk)
        mask = ks.isin(keys_s)
        sel = chunk.loc[mask]
        if not sel.empty:
            matches.append(sel)
            if cols_seen is None:
                cols_seen = sel.columns.tolist()
        del chunk, ks, mask; gc.collect()

    if not matches:
        return pd.DataFrame(columns=(usecols or []))

    df = pd.concat(matches, ignore_index=True)

    # normalize key dtypes
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce').astype('Int64')
    df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce').astype('Int64')
    df['day_index'] = pd.to_numeric(df['day_index'], errors='coerce').astype('Int64')

    # Now collapse duplicates per key
    # Define aggregator
    agg_dict = {}
    for c in df.columns:
        lc = c.lower()
        if c in ('subject_id','hadm_id','day_index'):
            agg_dict[c] = 'first'
        elif lc in ('labs_count','labs_numeric_count','labs_abnormal_count'):
            agg_dict[c] = 'sum'
        elif lc == 'labs_numeric_sum':
            agg_dict[c] = 'sum'
        elif lc == 'labs_numeric_max':
            agg_dict[c] = 'max'
        elif lc.startswith('last_'):
            # we'll handle last_* by selecting the row with max last_charttime after grouping
            agg_dict[c] = lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] > 0 else pd.NA
        elif lc.endswith('_charttime'):
            agg_dict[c] = 'max'
        else:
            # fallback: take last non-null
            agg_dict[c] = lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] > 0 else pd.NA

    grouped = df.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(agg_dict)

    # ensure dtypes
    grouped['subject_id'] = pd.to_numeric(grouped['subject_id'], errors='coerce').astype('Int64')
    grouped['hadm_id'] = pd.to_numeric(grouped['hadm_id'], errors='coerce').astype('Int64')
    grouped['day_index'] = pd.to_numeric(grouped['day_index'], errors='coerce').astype('Int64')

    return grouped

# ---------------- Workflow ----------------

print(f"[{nowstr()}] START other_labs_raw -> aggregated merging")

admit_map = build_admit_map(MERGED_INITIAL_PATH)

# Phase A: create tmp per-chunk partial aggregates
if TMP_AGG_PATH.exists():
    TMP_AGG_PATH.unlink()
print(f"[{nowstr()}] Phase A: scanning OTHER_LABS_PATH and writing partial aggregates to {TMP_AGG_PATH}")

hdr = read_header(OTHER_LABS_PATH)
usecols_raw = [c for c in ['subject_id','hadm_id','itemid','charttime','value','valuenum','valueuom','flag'] if c in hdr]
parse_dates = ['charttime'] if 'charttime' in hdr else []

first_out = True
total_partial_rows = 0
chunk_no = 0
for chunk in pd.read_csv(OTHER_LABS_PATH, usecols=usecols_raw, parse_dates=parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True):
    chunk_no += 1
    t0 = time.time()
    print(f"[{nowstr()}] other_labs chunk {chunk_no}: rows={len(chunk):,}")
    # normalize ids
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')

    # map admit_date
    keys = list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date'] = [admit_map.get((int(s), int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]

    before = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    dropped = before - len(chunk)
    if dropped:
        print(f"  -> dropped {dropped:,} other_labs rows without admission mapping")
    if chunk.empty:
        del chunk; gc.collect(); continue

    # compute day_index
    chunk['charttime'] = pd.to_datetime(chunk['charttime'], errors='coerce')
    chunk['chart_date'] = chunk['charttime'].dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    # numeric parsing: use valuenum if exists else parse value string
    if 'valuenum' in chunk.columns:
        chunk['numeric_val'] = pd.to_numeric(chunk['valuenum'], errors='coerce')
    else:
        chunk['numeric_val'] = pd.NA
    # parse missing numerics from 'value'
    mask_num_missing = chunk['numeric_val'].isna()
    if mask_num_missing.any():
        parsed = chunk.loc[mask_num_missing,'value'].astype(str).str.replace(',','')
        parsed_num = pd.to_numeric(parsed, errors='coerce')
        chunk.loc[mask_num_missing,'numeric_val'] = parsed_num

    # abnormal flag detection
    if 'flag' in chunk.columns:
        chunk['is_abnormal'] = chunk['flag'].astype(str).str.lower().str.contains('abnormal', na=False).astype(int)
    else:
        chunk['is_abnormal'] = 0

    # For last_* fields we compute per-chunk the last by charttime (sort and take last)
    chunk = chunk.sort_values('charttime')

    # partial aggregation per chunk: sum counts, sum numeric, max numeric, sum abnormal, last_* by charttime
    partial = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        labs_count = ('itemid','count'),
        labs_numeric_sum = ('numeric_val', 'sum'),
        labs_numeric_count = ('numeric_val', lambda s: int(s.notna().sum())),
        labs_numeric_max = ('numeric_val', 'max'),
        labs_abnormal_count = ('is_abnormal', 'sum'),
        last_charttime = ('charttime', 'max'),
        last_itemid = ('itemid', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        last_value = ('value', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        last_flag = ('flag', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA)
    )

    # ensure dtypes
    partial['subject_id'] = pd.to_numeric(partial['subject_id'], errors='coerce').astype('Int64')
    partial['hadm_id'] = pd.to_numeric(partial['hadm_id'], errors='coerce').astype('Int64')
    partial['day_index'] = pd.to_numeric(partial['day_index'], errors='coerce').astype('Int64')

    # append to TMP_AGG_PATH
    partial.to_csv(TMP_AGG_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False
    total_partial_rows += len(partial)
    print(f"[{nowstr()}] other_labs chunk {chunk_no} -> wrote {len(partial):,} partial agg rows (total partial {total_partial_rows:,}) took {time.time()-t0:.1f}s")

    del chunk, partial; gc.collect()

print(f"[{nowstr()}] Phase A done. partial agg rows total (appended): {total_partial_rows}")
print(f"[{nowstr()}] Temporary aggregated file: {TMP_AGG_PATH}")

# ---------------- Phase B: merge partial aggregates into base in chunks ----------------
print(f"[{nowstr()}] Phase B: merge tmp partial aggregates into {OUT_PATH} in chunks (WRITE_CHUNK={WRITE_CHUNK})")

# helper reuse: collect_matching_rows_and_collapse defined above
# columns in TMP_AGG_PATH: subject_id,hadm_id,day_index,labs_count,labs_numeric_sum,labs_numeric_count,labs_numeric_max,labs_abnormal_count,last_charttime,last_itemid,last_value,last_flag

if OUT_PATH.exists():
    OUT_PATH.unlink()

first_write = True
mchunk_no = 0
for mchunk in pd.read_csv(MERGED_INITIAL_PATH, chunksize=WRITE_CHUNK, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'], low_memory=False):
    mchunk_no += 1
    t0 = time.time()
    print(f"[{nowstr()}] Processing merged_initial chunk {mchunk_no}, rows={len(mchunk):,}")
    mchunk['subject_id'] = pd.to_numeric(mchunk['subject_id'], errors='coerce').astype('Int64')
    mchunk['hadm_id'] = pd.to_numeric(mchunk['hadm_id'], errors='coerce').astype('Int64')
    mchunk['day_index'] = pd.to_numeric(mchunk['day_index'], errors='coerce').astype('Int64')

    # build key set for this chunk
    keys = set((int(r['subject_id']), int(r['hadm_id']), int(r['day_index'])) for _, r in mchunk[['subject_id','hadm_id','day_index']].iterrows())

    # collect partial matches from TMP_AGG_PATH and collapse duplicates
    matches = collect_matching_rows_and_collapse(TMP_AGG_PATH, keys, usecols=None, parse_dates=['last_charttime'], chunksize=MATCH_CHUNK)

    # post-process matches: compute mean from sum/count
    if not matches.empty:
        matches = matches.rename(columns={
            'labs_count':'other_labs_count',
            'labs_numeric_sum':'other_labs_numeric_sum',
            'labs_numeric_count':'other_labs_numeric_count',
            'labs_numeric_max':'other_labs_numeric_max',
            'labs_abnormal_count':'other_labs_abnormal_count',
            'last_charttime':'other_labs_last_charttime',
            'last_itemid':'other_labs_last_itemid',
            'last_value':'other_labs_last_value',
            'last_flag':'other_labs_last_flag'
        })
        # compute mean
        matches['other_labs_numeric_mean'] = matches.apply(
            lambda r: (float(r['other_labs_numeric_sum']) / int(r['other_labs_numeric_count'])) if (pd.notna(r['other_labs_numeric_sum']) and pd.notna(r['other_labs_numeric_count']) and int(r['other_labs_numeric_count'])>0) else pd.NA,
            axis=1
        )

        # keep only relevant columns for merging (keys + features)
        keep_cols = ['subject_id','hadm_id','day_index',
                     'other_labs_count','other_labs_numeric_count','other_labs_numeric_mean','other_labs_numeric_max','other_labs_abnormal_count',
                     'other_labs_last_charttime','other_labs_last_itemid','other_labs_last_value','other_labs_last_flag']
        # make sure all keep_cols exist
        for c in keep_cols:
            if c not in matches.columns:
                matches[c] = pd.NA
        matches = matches[keep_cols]

    # merge into mchunk (left join)
    if not matches.empty:
        mchunk = mchunk.merge(matches, on=['subject_id','hadm_id','day_index'], how='left')

    # write out chunk
    mchunk.to_csv(OUT_PATH, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False

    print(f"[{nowstr()}] Written merged chunk {mchunk_no} (took {time.time()-t0:.1f}s)")
    del mchunk, matches; gc.collect()

print(f"[{nowstr()}] DONE. Output: {OUT_PATH}")
print(f"[{nowstr()}] Temporary file kept at: {TMP_AGG_PATH} (you can remove it when satisfied)")
