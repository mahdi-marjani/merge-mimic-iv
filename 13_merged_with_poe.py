import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

# ---------------- CONFIG ----------------
BASE = Path(".")
MERGED_INITIAL_PATH = BASE / "admissions_expanded.csv"
POE_PATH = BASE / "poe.csv"

AGG_POE_PATH = BASE / "agg_poe_daily.csv"
OUT_MERGED_POE = BASE / "merged_with_poe.csv"

SRC_CHUNKSIZE = 200_000   # read size for poe.csv
WRITE_CHUNK = 20_000      # rows of merged_initial processed per write
MATCH_CHUNK = 200_000     # chunk size when scanning agg csvs for matches
# ----------------------------------------

# ---------- helpers ----------
def nowstr(): return time.strftime("%Y-%m-%d %H:%M:%S")
def parse_numeric(x):
    if pd.isna(x): return np.nan
    try:
        if isinstance(x,(int,float,np.number)): return float(x)
        s = str(x).strip().replace(',','')
        if s in ("","NaN","nan","None","none","___"): return np.nan
        return float(s)
    except:
        return np.nan

def read_header(path: Path):
    if not path.exists(): return []
    return pd.read_csv(path, nrows=0).columns.tolist()

def build_admit_map(merged_initial_path):
    print(f"[{nowstr()}] Build admit_map (light read)...")
    mi = pd.read_csv(merged_initial_path, usecols=['subject_id','hadm_id','admittime'], parse_dates=['admittime'], low_memory=False)
    mi['subject_id'] = pd.to_numeric(mi['subject_id'], errors='coerce').astype('Int64')
    mi['hadm_id'] = pd.to_numeric(mi['hadm_id'], errors='coerce').astype('Int64')
    adm = mi.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
    adm['admit_date'] = pd.to_datetime(adm['admit_time']).dt.normalize()
    d = dict(zip(zip(adm['subject_id'].astype(int), adm['hadm_id'].astype(int)), adm['admit_date']))
    print(f"[{nowstr()}] admit_map keys: {len(d)}")
    del mi, adm; gc.collect()
    return d

def key_str(s,h,d): return f"{int(s)}|{int(h)}|{int(d)}"
def key_series_from_df(df):
    s = pd.to_numeric(df['subject_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    h = pd.to_numeric(df['hadm_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    d = pd.to_numeric(df['day_index'], errors='coerce').fillna(-1).astype(int).astype(str)
    return s + '|' + h + '|' + d

def _choose_agg_for_col(col_name):
    name = col_name.lower()
    if name in ('subject_id','hadm_id','day_index'):
        return 'first'
    if 'count' in name or 'total' in name or name.endswith('_sum'):
        return 'sum'
    if 'max' in name or name.endswith('_max'):
        return 'max'
    if 'amount' in name:
        return 'sum'
    if 'rate' in name or name.endswith('_num') or 'val' in name or 'value' in name or 'ordertime' in name:
        # for ordertime we want min/max, but groupby agg requires consistent functions:
        # ordetime we'll handle separately; here default to 'max' for numeric-like
        return 'max'
    return lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] > 0 else pd.NA

def collect_matching_rows(agg_path, keys_set, usecols=None, parse_dates=None, chunksize=MATCH_CHUNK):
    """
    Read agg_path in chunks and collect rows matching keys_set.
    Then collapse duplicate keys via heuristics and return unique-key dataframe.
    """
    if not agg_path.exists() or not keys_set:
        return pd.DataFrame(columns=(usecols or []))

    keys_s = set(key_str(s,h,d) for (s,h,d) in keys_set)
    matches = []
    cols_seen = None
    for chunk in pd.read_csv(agg_path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize, low_memory=True):
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
    # ensure types
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce').astype('Int64')
    df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce').astype('Int64')
    df['day_index'] = pd.to_numeric(df['day_index'], errors='coerce').astype('Int64')

    # choose agg per column
    agg_dict = {}
    for col in df.columns:
        agg_dict[col] = _choose_agg_for_col(col)

    # group and aggregate
    grouped = df.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(agg_dict)

    # post-process ordertime columns if present: compute min/max from original matches if needed
    # (In our AGG_POE we included first/last ordertime already; if not, this block can be extended.)

    grouped['subject_id'] = pd.to_numeric(grouped['subject_id'], errors='coerce').astype('Int64')
    grouped['hadm_id'] = pd.to_numeric(grouped['hadm_id'], errors='coerce').astype('Int64')
    grouped['day_index'] = pd.to_numeric(grouped['day_index'], errors='coerce').astype('Int64')

    return grouped

# ---------- prepare admit_map ----------
admit_dict = build_admit_map(MERGED_INITIAL_PATH)

# ---------- Phase A: aggregate poe.csv -> AGG_POE_PATH ----------
print(f"[{nowstr()}] Phase A: aggregate poe -> {AGG_POE_PATH}")
if AGG_POE_PATH.exists(): AGG_POE_PATH.unlink()
hdr = read_header(POE_PATH)
# choose columns we expect; safe to include extras if present
use_want = ['poe_id','poe_seq','subject_id','hadm_id','ordertime','order_type','order_subtype','transaction_type','discontinue_of_poe_id','discontinued_by_poe_id','order_provider_id','order_status']
usecols = [c for c in use_want if c in hdr]
parse_dates = ['ordertime'] if 'ordertime' in usecols else []

first_out=True; total=0
for i, chunk in enumerate(pd.read_csv(POE_PATH, usecols=usecols, parse_dates=parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0=time.time(); print(f"[{nowstr()}] poe chunk {i} rows={len(chunk):,}")
    # normalize ids
    if 'subject_id' not in chunk.columns or 'hadm_id' not in chunk.columns:
        print("ERROR: poe missing subject_id or hadm_id columns. Aborting.")
        raise SystemExit(1)
    chunk['subject_id']=pd.to_numeric(chunk['subject_id'],errors='coerce').astype('Int64')
    chunk['hadm_id']=pd.to_numeric(chunk['hadm_id'],errors='coerce').astype('Int64')

    # map admit_date
    keys = list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date'] = [admit_dict.get((int(s), int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]

    before = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    dropped = before - len(chunk)
    if dropped: print(f"  -> dropped {dropped:,} poe rows w/o admit")

    if chunk.empty:
        del chunk; gc.collect(); continue

    # event_time and day_index: use ordertime
    chunk['event_time'] = pd.to_datetime(chunk.get('ordertime', pd.NaT), errors='coerce')
    chunk['chart_date'] = chunk['event_time'].dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    # ensure provider and status columns exist
    if 'order_provider_id' not in chunk.columns:
        chunk['order_provider_id'] = pd.NA
    if 'order_status' not in chunk.columns:
        chunk['order_status'] = pd.NA
    if 'order_type' not in chunk.columns:
        chunk['order_type'] = pd.NA
    if 'order_subtype' not in chunk.columns:
        chunk['order_subtype'] = pd.NA
    if 'transaction_type' not in chunk.columns:
        chunk['transaction_type'] = pd.NA

    # string/text normalization
    chunk['order_provider_id'] = chunk['order_provider_id'].astype(str).replace('nan','').replace('None','').replace('NoneType','').fillna(pd.NA)
    chunk['order_status'] = chunk['order_status'].astype(str).replace('nan','').replace('None','').fillna(pd.NA)
    chunk['order_type'] = chunk['order_type'].astype(str).replace('nan','').replace('None','').fillna(pd.NA)
    chunk['order_subtype'] = chunk['order_subtype'].astype(str).replace('nan','').replace('None','').fillna(pd.NA)
    chunk['transaction_type'] = chunk['transaction_type'].astype(str).replace('nan','').replace('None','').fillna(pd.NA)

    # Aggregations per chunk (per day)
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        poe_events_count = ('poe_id','count'),
        poe_first_ordertime = ('event_time','min'),
        poe_last_ordertime = ('event_time','max'),
        poe_first_order_provider = ('order_provider_id', lambda s: s.dropna().iloc[0] if s.dropna().shape[0]>0 else pd.NA),
        poe_last_order_provider = ('order_provider_id', lambda s: s.dropna().iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        poe_last_order_type = ('order_type', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        poe_last_order_subtype = ('order_subtype', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        poe_last_transaction_type = ('transaction_type', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA),
        poe_last_order_status = ('order_status', lambda s: s.dropna().astype(str).iloc[-1] if s.dropna().shape[0]>0 else pd.NA)
    )

    # ensure key dtypes
    agg['subject_id']=agg['subject_id'].astype('Int64')
    agg['hadm_id']=agg['hadm_id'].astype('Int64')
    agg['day_index']=agg['day_index'].astype('Int64')

    # write chunked aggregated rows (may produce multiple rows per key across chunks; will collapse on merge)
    agg.to_csv(AGG_POE_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out=False
    total += len(agg)
    print(f"[{nowstr()}] poe chunk {i} -> wrote {len(agg):,} agg rows (total appended {total:,}) took {time.time()-t0:.1f}s")
    del chunk, agg; gc.collect()

print(f"[{nowstr()}] POE aggregation done. raw agg rows appended: {total}")

# ---------- Phase B: merge aggregated poe into merged_initial in chunks ----------
print(f"[{nowstr()}] Phase B: merge aggregated poe into {OUT_MERGED_POE} in chunks (write_chunk={WRITE_CHUNK})")
if OUT_MERGED_POE.exists(): OUT_MERGED_POE.unlink()
first_write = True
mchunk_no = 0

for mchunk in pd.read_csv(MERGED_INITIAL_PATH, chunksize=WRITE_CHUNK, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'], low_memory=False):
    mchunk_no += 1
    t0 = time.time()
    print(f"[{nowstr()}] merged chunk {mchunk_no} rows={len(mchunk):,}")
    mchunk['subject_id'] = pd.to_numeric(mchunk['subject_id'], errors='coerce').astype('Int64')
    mchunk['hadm_id'] = pd.to_numeric(mchunk['hadm_id'], errors='coerce').astype('Int64')
    mchunk['day_index'] = pd.to_numeric(mchunk['day_index'], errors='coerce').astype('Int64')

    keys = set((int(r['subject_id']), int(r['hadm_id']), int(r['day_index'])) for _,r in mchunk[['subject_id','hadm_id','day_index']].iterrows())

    # collect matching rows from AGG_POE_PATH and collapse duplicates per key
    poe_match = collect_matching_rows(AGG_POE_PATH, keys, usecols=None, parse_dates=['poe_first_ordertime','poe_last_ordertime']) if AGG_POE_PATH.exists() else pd.DataFrame()

    if not poe_match.empty:
        # note: parse_dates argument above may or may not have effect since agg csv stores ISO timestamps;
        # ensure datetime dtype for ordertime cols
        if 'poe_first_ordertime' in poe_match.columns:
            poe_match['poe_first_ordertime'] = pd.to_datetime(poe_match['poe_first_ordertime'], errors='coerce')
        if 'poe_last_ordertime' in poe_match.columns:
            poe_match['poe_last_ordertime'] = pd.to_datetime(poe_match['poe_last_ordertime'], errors='coerce')

        mchunk = mchunk.merge(poe_match, on=['subject_id','hadm_id','day_index'], how='left')

    # write chunk
    mchunk.to_csv(OUT_MERGED_POE, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False
    print(f"[{nowstr()}] Written merged chunk {mchunk_no} (took {time.time()-t0:.1f}s)")
    del mchunk, poe_match; gc.collect()

print(f"[{nowstr()}] Done. Output: {OUT_MERGED_POE}")
