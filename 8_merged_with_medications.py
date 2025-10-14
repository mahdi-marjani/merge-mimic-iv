import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

# ---------- config ----------
BASE_DIR = Path(".")
MERGED_INITIAL_PATH = BASE_DIR / "admissions_expanded.csv"
PRESC_PATH = BASE_DIR / "prescriptions.csv"
PHARM_PATH = BASE_DIR / "pharmacy.csv"
EMAR_PATH = BASE_DIR / "emar.csv"

AGG_PRESC_PATH = BASE_DIR / "agg_prescriptions_daily.csv"
AGG_PHARM_PATH = BASE_DIR / "agg_pharmacy_daily.csv"
AGG_EMAR_PATH  = BASE_DIR / "agg_emar_daily.csv"

# chunk sizes (tune these)
SRC_CHUNKSIZE = 200_000
WRITE_CHUNK = 20_000
MATCH_CHUNK = 200_000

# ---------- helpers ----------
def nowstr():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def parse_numeric(x):
    if pd.isna(x): return np.nan
    try:
        if isinstance(x, (int,float,np.number)): return float(x)
        s = str(x).strip().replace(',', '')
        if s in ("", "NaN", "nan", "None", "none", "___"): return np.nan
        return float(s)
    except:
        return np.nan

def read_header(path: Path):
    if not path.exists():
        return []
    return pd.read_csv(path, nrows=0).columns.tolist()

def build_admit_map(merged_initial_path):
    print(f"[{nowstr()}] Building admit_map from {merged_initial_path} (light read)...")
    mi = pd.read_csv(merged_initial_path, usecols=['subject_id','hadm_id','admittime'],
                     parse_dates=['admittime'], low_memory=False)
    mi['subject_id'] = pd.to_numeric(mi['subject_id'], errors='coerce').astype('Int64')
    mi['hadm_id'] = pd.to_numeric(mi['hadm_id'], errors='coerce').astype('Int64')
    admit = mi.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
    admit['admit_date'] = pd.to_datetime(admit['admit_time']).dt.normalize()
    admit_dict = dict(zip(zip(admit['subject_id'].astype(int), admit['hadm_id'].astype(int)), admit['admit_date']))
    print(f"[{nowstr()}] Admit map keys: {len(admit_dict)}")
    del mi, admit
    gc.collect()
    return admit_dict

def consolidate_agg(path: Path, out_path: Path, key_cols=('subject_id','hadm_id','day_index'),
                    sum_cols=None, min_cols=None, max_cols=None, pick_by_max=None, pick_cols=None):
    """
    Read a per-chunk AGG file (which may have duplicates for the same (s,h,d)),
    and consolidate so result has one row per (s,h,d).
    - sum_cols: list of columns to sum across duplicates (e.g., counts)
    - min_cols: list of cols to take min
    - max_cols: list of cols to take max
    - pick_by_max: a timestamp column name — for pick_cols choose the row with max(pick_by_max) and take those values
    """
    sum_cols = sum_cols or []
    min_cols = min_cols or []
    max_cols = max_cols or []
    pick_cols = pick_cols or []

    if not path.exists():
        # nothing to do
        print(f"[{nowstr()}] consolidate_agg: {path} not found, skipping.")
        return

    print(f"[{nowstr()}] Consolidating {path} -> {out_path} ...")
    # read whole AGG file (these AGG files are usually much smaller than raw sources)
    df = pd.read_csv(path, low_memory=False)
    # ensure key types
    for c in key_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # convert candidate datetime columns if present
    candidate_dt = []
    if pick_by_max:
        candidate_dt.append(pick_by_max)
    for col in (min_cols + max_cols + candidate_dt):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Build aggregation dict for groupby. Use named aggregations for readability.
    agg_dict = {}
    for c in sum_cols:
        if c in df.columns:
            agg_dict[c] = 'sum'
    for c in min_cols:
        if c in df.columns:
            agg_dict[c] = 'min'
    for c in max_cols:
        if c in df.columns:
            agg_dict[c] = 'max'

    # if no aggregation columns present (edge), just drop duplicates using first
    if not agg_dict and not pick_cols:
        out = df.drop_duplicates(subset=list(key_cols)).reset_index(drop=True)
        out.to_csv(out_path, index=False)
        print(f"[{nowstr()}] Consolidation done (simple dedupe). rows: {len(out)}")
        return

    grouped = df.groupby(list(key_cols), dropna=False, as_index=False).agg(agg_dict) if agg_dict else df.groupby(list(key_cols), dropna=False, as_index=False).first()

    # handle pick_by_max for pick_cols
    if pick_by_max and any(pc in df.columns for pc in pick_cols):
        # For each group, pick the row index in df with max(pick_by_max)
        # handle groups where pick_by_max is NaT by falling back to first occurrence
        idx = df.groupby(list(key_cols))[pick_by_max].idxmax().dropna()
        # idx is a Series mapping group -> row index (may miss groups where pick_by_max all NaT)
        picked = df.loc[idx].reset_index(drop=True)
        # reduce picked to key_cols + pick_cols (and pick_by_max if exists)
        take_cols = list(key_cols) + [c for c in pick_cols if c in picked.columns]
        picked = picked[take_cols]
        # merge picked into grouped (picked values overwrite grouped's columns if same names)
        grouped = grouped.merge(picked, on=list(key_cols), how='left')

        # for groups not present in picked (all NaT), try to fill pick_cols from first row per group
        missing_mask = grouped[pick_cols[0]].isna() if pick_cols and pick_cols[0] in grouped.columns else None
        if missing_mask is not None and missing_mask.any():
            # get first row per group
            first_rows = df.groupby(list(key_cols), as_index=False).first()[list(key_cols)+[c for c in pick_cols if c in df.columns]]
            grouped = grouped.merge(first_rows, on=list(key_cols), how='left', suffixes=('','__first'))
            # fill NaNs in pick_cols with __first
            for c in pick_cols:
                if c in grouped.columns and (c + "__first") in grouped.columns:
                    grouped[c] = grouped[c].combine_first(grouped[c + "__first"])
                    grouped = grouped.drop(columns=[c + "__first"])
    # Ensure types for keys are Int64 where possible
    for c in key_cols:
        if c in grouped.columns:
            grouped[c] = pd.to_numeric(grouped[c], errors='coerce').astype('Int64')

    grouped.to_csv(out_path, index=False)
    print(f"[{nowstr()}] Consolidation done. rows: {len(grouped)} (written to {out_path})")
    del df, grouped
    gc.collect()


def collect_matching_rows(agg_path: Path, keys_set, usecols=None, parse_dates=None, chunksize=MATCH_CHUNK):
    """Scan agg_path in chunks and return rows matching keys_set (keys_set of (s,h,d) ints)."""
    if not agg_path.exists():
        return pd.DataFrame(columns=(usecols or []))
    if not keys_set:
        return pd.DataFrame(columns=(usecols or []))
    keys_str = set(f"{int(s)}|{int(h)}|{int(d)}" for (s,h,d) in keys_set)
    matches = []
    usecols = usecols or None
    for c_i, chunk in enumerate(pd.read_csv(agg_path, usecols=usecols, parse_dates=parse_dates or [], chunksize=chunksize, low_memory=True)):
        # ensure key columns exist
        if not {'subject_id','hadm_id','day_index'}.issubset(set(chunk.columns)):
            continue
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
        chunk['day_index'] = pd.to_numeric(chunk['day_index'], errors='coerce').astype('Int64')
        keys = chunk['subject_id'].astype(str) + '|' + chunk['hadm_id'].astype(str) + '|' + chunk['day_index'].astype(str)
        mask = keys.isin(keys_str)
        sel = chunk[mask]
        if not sel.empty:
            matches.append(sel)
        del chunk, keys, mask, sel
        gc.collect()
    if matches:
        return pd.concat(matches, ignore_index=True)
    else:
        return pd.DataFrame(columns=(usecols or []))


# -------------------- run --------------------
import gc
admit_dict = build_admit_map(MERGED_INITIAL_PATH)

# ---------- Phase 1: aggregate prescriptions ----------
print(f"[{nowstr()}] Phase 1 — aggregate prescriptions -> {AGG_PRESC_PATH}")
if AGG_PRESC_PATH.exists(): AGG_PRESC_PATH.unlink()

presc_header = read_header(PRESC_PATH)
presc_usecols_want = ['subject_id','hadm_id','starttime','stoptime','drug','formulary_drug_cd','ndc','prod_strength','dose_val_rx','dose_unit_rx','route','doses_per_24_hrs','drug_type','poe_id','poe_seq','order_provider_id']
presc_usecols = [c for c in presc_usecols_want if c in presc_header]
presc_parse_dates = [c for c in ['starttime','stoptime'] if c in presc_header]

first_out = True
total_out = 0
for i, chunk in enumerate(pd.read_csv(PRESC_PATH, usecols=presc_usecols, parse_dates=presc_parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0 = time.time()
    print(f"[{nowstr()}] Presc chunk {i} rows={len(chunk):,}")
    # cast ids
    if 'subject_id' in chunk.columns:
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
    keys = list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date'] = [admit_dict.get((int(s), int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    dropped = before - len(chunk)
    if dropped:
        print(f"  -> dropped {dropped:,} presc rows with no admit mapping")
    if chunk.empty:
        del chunk; gc.collect(); continue
    chunk['event_time'] = pd.to_datetime(chunk['starttime'], errors='coerce').fillna(pd.to_datetime(chunk.get('stoptime', pd.NaT), errors='coerce'))
    chunk['chart_date'] = chunk['event_time'].dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0
    chunk['dose_val_num'] = chunk['dose_val_rx'].apply(parse_numeric) if 'dose_val_rx' in chunk.columns else np.nan
    chunk['prod_strength_num'] = chunk['prod_strength'].apply(parse_numeric) if 'prod_strength' in chunk.columns else np.nan
    chunk['doses_per_24h_num'] = chunk['doses_per_24_hrs'].apply(parse_numeric) if 'doses_per_24_hrs' in chunk.columns else np.nan
    chunk['drug_text'] = chunk['drug'].astype(str).str.strip().replace({'nan':None})
    chunk['route_text'] = chunk['route'].astype(str).str.strip().replace({'nan':None})
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        presc_orders_count = ('drug_text','count'),
        presc_first_start = ('starttime','min') if 'starttime' in chunk.columns else pd.NamedAgg(column='event_time', aggfunc='min'),
        presc_last_stop  = ('stoptime','max') if 'stoptime' in chunk.columns else pd.NamedAgg(column='event_time', aggfunc='max'),
        presc_last_drug  = ('drug_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        presc_last_route = ('route_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        presc_max_dose   = ('dose_val_num','max'),
        presc_max_strength = ('prod_strength_num','max'),
        presc_doses_per_24h = ('doses_per_24h_num','max')
    )
    agg['subject_id'] = agg['subject_id'].astype('Int64')
    agg['hadm_id'] = agg['hadm_id'].astype('Int64')
    agg['day_index'] = agg['day_index'].astype('Int64')
    agg.to_csv(AGG_PRESC_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False
    total_out += len(agg)
    print(f"[{nowstr()}] Presc chunk {i} -> wrote {len(agg):,} agg rows (total {total_out:,})  took {time.time()-t0:.1f}s")
    del chunk, agg
    gc.collect()

print(f"[{nowstr()}] Prescriptions aggregation done. total agg rows (unconsolidated): {total_out}")

# consolidate the AGG presc into unique keys
consolidate_agg(AGG_PRESC_PATH, AGG_PRESC_PATH,
                key_cols=('subject_id','hadm_id','day_index'),
                sum_cols=['presc_orders_count'],
                min_cols=['presc_first_start'],
                max_cols=['presc_last_stop','presc_max_dose','presc_max_strength','presc_doses_per_24h'],
                pick_by_max='presc_last_stop',
                pick_cols=['presc_last_drug','presc_last_route'])

# ---------- Phase 2: aggregate pharmacy ----------
print(f"[{nowstr()}] Phase 2 — aggregate pharmacy -> {AGG_PHARM_PATH}")
if AGG_PHARM_PATH.exists(): AGG_PHARM_PATH.unlink()

pharm_header = read_header(PHARM_PATH)
pharm_usecols_want = ['subject_id','hadm_id','starttime','stoptime','medication','route','frequency','dispensation','fill_quantity','entertime','verifiedtime','expirationdate']
pharm_usecols = [c for c in pharm_usecols_want if c in pharm_header]
pharm_parse_dates = [c for c in ['starttime','stoptime','entertime','verifiedtime','expirationdate'] if c in pharm_header]

first_out = True
total_out = 0
for i, chunk in enumerate(pd.read_csv(PHARM_PATH, usecols=pharm_usecols, parse_dates=pharm_parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0 = time.time()
    print(f"[{nowstr()}] Pharm chunk {i} rows={len(chunk):,}")
    if 'subject_id' in chunk.columns:
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
    keys = list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date'] = [admit_dict.get((int(s), int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    dropped = before - len(chunk)
    if dropped:
        print(f"  -> dropped {dropped:,} pharm rows with no admit mapping")
    if chunk.empty:
        del chunk; gc.collect(); continue
    chunk['event_time'] = pd.to_datetime(chunk['starttime'], errors='coerce').fillna(pd.to_datetime(chunk.get('verifiedtime', pd.NaT), errors='coerce')).fillna(pd.to_datetime(chunk.get('entertime', pd.NaT), errors='coerce'))
    chunk['chart_date'] = chunk['event_time'].dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0
    chunk['fill_qty_num'] = chunk['fill_quantity'].apply(parse_numeric) if 'fill_quantity' in chunk.columns else np.nan
    chunk['med_text'] = chunk['medication'].astype(str).str.strip().replace({'nan':None})
    chunk['route_text'] = chunk['route'].astype(str).str.strip().replace({'nan':None})
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        pharm_dispense_count = ('med_text','count'),
        pharm_first_start = ('starttime','min') if 'starttime' in chunk.columns else pd.NamedAgg(column='event_time', aggfunc='min'),
        pharm_last_stop  = ('stoptime','max') if 'stoptime' in chunk.columns else pd.NamedAgg(column='event_time', aggfunc='max'),
        pharm_last_med   = ('med_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        pharm_last_route = ('route_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        pharm_max_fill_qty = ('fill_qty_num','max')
    )
    agg['subject_id'] = agg['subject_id'].astype('Int64')
    agg['hadm_id'] = agg['hadm_id'].astype('Int64')
    agg['day_index'] = agg['day_index'].astype('Int64')
    agg.to_csv(AGG_PHARM_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False
    total_out += len(agg)
    print(f"[{nowstr()}] Pharm chunk {i} -> wrote {len(agg):,} agg rows (total {total_out:,})  took {time.time()-t0:.1f}s")
    del chunk, agg
    gc.collect()

print(f"[{nowstr()}] Pharmacy aggregation done. total agg rows (unconsolidated): {total_out}")

consolidate_agg(AGG_PHARM_PATH, AGG_PHARM_PATH,
                key_cols=('subject_id','hadm_id','day_index'),
                sum_cols=['pharm_dispense_count'],
                min_cols=['pharm_first_start'],
                max_cols=['pharm_last_stop','pharm_max_fill_qty'],
                pick_by_max='pharm_last_stop',
                pick_cols=['pharm_last_med','pharm_last_route'])


# ---------- Phase 3: aggregate emar ----------
print(f"[{nowstr()}] Phase 3 — aggregate emar -> {AGG_EMAR_PATH}")
if AGG_EMAR_PATH.exists(): AGG_EMAR_PATH.unlink()

emar_header = read_header(EMAR_PATH)
emar_usecols_want = ['subject_id','hadm_id','charttime','medication','event_txt','pharmacy_id']
emar_usecols = [c for c in emar_usecols_want if c in emar_header]
emar_parse_dates = [c for c in ['charttime'] if c in emar_header]

first_out = True
total_out = 0
for i, chunk in enumerate(pd.read_csv(EMAR_PATH, usecols=emar_usecols, parse_dates=emar_parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0 = time.time()
    print(f"[{nowstr()}] Emar chunk {i} rows={len(chunk):,}")
    if 'subject_id' in chunk.columns:
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
    keys = list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date'] = [admit_dict.get((int(s), int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    dropped = before - len(chunk)
    if dropped:
        print(f"  -> dropped {dropped:,} emar rows with no admit mapping")
    if chunk.empty:
        del chunk; gc.collect(); continue
    chunk['chart_date'] = pd.to_datetime(chunk['charttime'], errors='coerce').dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0
    chunk['med_text'] = chunk['medication'].astype(str).str.strip().replace({'nan':None})
    chunk['event_txt'] = chunk['event_txt'].astype(str).str.strip().replace({'nan':None})
    chunk['admin_flag'] = chunk['event_txt'].str.contains('Administered|Given|Given by|Admin|Dose|Flushed', case=False, na=False).astype(int)
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        emar_events_count = ('event_txt','count'),
        emar_admin_count  = ('admin_flag','sum'),
        emar_last_event   = ('event_txt', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        emar_last_med     = ('med_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        emar_last_charttime = ('charttime','max')
    )
    agg['subject_id'] = agg['subject_id'].astype('Int64')
    agg['hadm_id'] = agg['hadm_id'].astype('Int64')
    agg['day_index'] = agg['day_index'].astype('Int64')
    agg.to_csv(AGG_EMAR_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False
    total_out += len(agg)
    print(f"[{nowstr()}] Emar chunk {i} -> wrote {len(agg):,} agg rows (total {total_out:,})  took {time.time()-t0:.1f}s")
    del chunk, agg
    gc.collect()

print(f"[{nowstr()}] Emar aggregation done. total agg rows (unconsolidated): {total_out}")

consolidate_agg(AGG_EMAR_PATH, AGG_EMAR_PATH,
                key_cols=('subject_id','hadm_id','day_index'),
                sum_cols=['emar_events_count','emar_admin_count'],
                min_cols=[],
                max_cols=['emar_last_charttime'],
                pick_by_max='emar_last_charttime',
                pick_cols=['emar_last_event','emar_last_med'])


# -------------------- Phase 4: merge aggregated into merged_initial in chunks --------------------
OUT_PATH = BASE_DIR / "merged_with_medications.csv"
if OUT_PATH.exists(): OUT_PATH.unlink()

print(f"[{nowstr()}] Phase 4 — merge aggregated files into {OUT_PATH} in chunks (write_chunk={WRITE_CHUNK})")
first_write = True
mchunk_no = 0
for mchunk in pd.read_csv(MERGED_INITIAL_PATH, chunksize=WRITE_CHUNK, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'], low_memory=False):
    mchunk_no += 1
    t0 = time.time()
    print(f"[{nowstr()}] Merged chunk {mchunk_no}: rows={len(mchunk):,}")
    mchunk['subject_id'] = pd.to_numeric(mchunk['subject_id'], errors='coerce').astype('Int64')
    mchunk['hadm_id'] = pd.to_numeric(mchunk['hadm_id'], errors='coerce').astype('Int64')
    mchunk['day_index'] = pd.to_numeric(mchunk['day_index'], errors='coerce').astype('Int64')
    # build key set
    keys = set((int(r['subject_id']), int(r['hadm_id']), int(r['day_index'])) for _, r in mchunk[['subject_id','hadm_id','day_index']].iterrows())
    # collect matching rows (each consolidated AGG is now unique per key)
    presc_match = collect_matching_rows(AGG_PRESC_PATH, keys) if AGG_PRESC_PATH.exists() else pd.DataFrame()
    pharm_match = collect_matching_rows(AGG_PHARM_PATH, keys) if AGG_PHARM_PATH.exists() else pd.DataFrame()
    emar_match  = collect_matching_rows(AGG_EMAR_PATH, keys)  if AGG_EMAR_PATH.exists() else pd.DataFrame()
    # merge (left) - since AGG files were consolidated, no key will duplicate
    if not presc_match.empty:
        mchunk = mchunk.merge(presc_match, on=['subject_id','hadm_id','day_index'], how='left')
    if not pharm_match.empty:
        mchunk = mchunk.merge(pharm_match, on=['subject_id','hadm_id','day_index'], how='left')
    if not emar_match.empty:
        mchunk = mchunk.merge(emar_match, on=['subject_id','hadm_id','day_index'], how='left')
    # write
    mchunk.to_csv(OUT_PATH, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False
    print(f"[{nowstr()}] Written merged chunk {mchunk_no} (took {time.time()-t0:.1f}s).")
    del mchunk, presc_match, pharm_match, emar_match
    gc.collect()

print(f"[{nowstr()}] All done. Output file: {OUT_PATH}")
