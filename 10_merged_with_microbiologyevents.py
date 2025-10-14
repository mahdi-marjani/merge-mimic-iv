import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 160)

# Paths - adjust if needed
merged_initial_path = Path("admissions_expanded.csv")   # created earlier by your notebook
micro_path = Path("microbiologyevents.csv")
out_merged_path = Path("merged_with_microbiologyevents.csv")

if not merged_initial_path.exists():
    raise FileNotFoundError(f"{merged_initial_path} not found. Run admissions expansion first.")

if not micro_path.exists():
    raise FileNotFoundError(f"{micro_path} not found. Put microbiologyevents.csv next to admissions.csv")

# 1) Load merged_initial (admission-day base)
merged_initial = pd.read_csv(merged_initial_path, low_memory=False,
                             parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'])
# ensure day_index is integer
merged_initial['day_index'] = merged_initial['day_index'].astype('Int64')

# 2) Build admit_map (admit_date per (subject_id,hadm_id)) and index map
admit_map = merged_initial.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time'], errors='coerce').dt.normalize()
admit_map['key'] = list(zip(admit_map['subject_id'].astype('Int64'), admit_map['hadm_id'].astype('Int64')))
admit_dict = dict(zip(admit_map['key'], admit_map['admit_date']))

# merged_initial index map to assign quickly
merged_initial_index_map = {}
for idx, row in merged_initial[['subject_id','hadm_id','day_index']].iterrows():
    # ensure ints
    try:
        key = (int(row['subject_id']), int(row['hadm_id']), int(row['day_index']))
        merged_initial_index_map[key] = idx
    except Exception:
        continue

print("Loaded merged_initial rows:", len(merged_initial))
print("Admit map keys:", len(admit_dict), "merged rows map size:", len(merged_initial_index_map))

# 3) Read microbiologyevents header to get columns
micro_head = pd.read_csv(micro_path, nrows=0)
micro_cols = micro_head.columns.tolist()
print("microbiologyevents columns:", micro_cols)

# We'll create merged_initial columns for each original micro column (prefixed with micro_)
prefix = "micro_"
new_cols = []
for c in micro_cols:
    if c in ('subject_id','hadm_id'):
        continue
    new_c = prefix + c
    new_cols.append(new_c)
    if new_c not in merged_initial.columns:
        merged_initial[new_c] = pd.Series([pd.NA] * len(merged_initial), dtype="object")

print("Added new micro columns to merged_initial (if missing). Count:", len(new_cols))

# 4) Decide which columns to treat as numeric (attempt a conservative list; we'll coerce in-chunk)
# Common numeric-like fields in microbiologyevents: isolate_num, quantity, dilution_value, ab_itemid, test_seq, microevent_id, spec_itemid
candidate_numeric = ['isolate_num','quantity','dilution_value','ab_itemid','test_seq','microevent_id','spec_itemid']

# 5) Chunked processing
chunksize = 200_000   # adjust if you want larger/smaller chunks
reader = pd.read_csv(micro_path, parse_dates=['charttime','chartdate','storedate'], chunksize=chunksize, low_memory=True)

total_assigned = 0
chunk_no = 0

for chunk in reader:
    chunk_no += 1
    print(f"\n--- Processing chunk {chunk_no} (rows: {len(chunk):,}) ---")
    # Ensure subject_id/hadm_id are numeric ints
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce')
    chunk = chunk.dropna(subset=['subject_id','hadm_id'])
    chunk['subject_id'] = chunk['subject_id'].astype(int)
    chunk['hadm_id'] = chunk['hadm_id'].astype(int)

    # Resolve chart_date: prefer charttime (if available) else chartdate
    if 'charttime' in chunk.columns and chunk['charttime'].notna().any():
        chunk['chartref'] = pd.to_datetime(chunk['charttime'], errors='coerce')
    else:
        chunk['chartref'] = pd.to_datetime(chunk.get('chartdate', pd.NaT), errors='coerce')

    chunk['chart_date'] = chunk['chartref'].dt.normalize()

    # Attach admit_date using admit_dict
    keys = list(zip(chunk['subject_id'].astype(int), chunk['hadm_id'].astype(int)))
    chunk['admit_date'] = [admit_dict.get(k, pd.NaT) for k in keys]

    # Drop rows with no matching admission
    before_drop = len(chunk)
    chunk = chunk[chunk['admit_date'].notna()].copy()
    after_drop = len(chunk)
    if after_drop == 0:
        print("no matching admissions in this chunk -> skipping")
        continue
    if after_drop < before_drop:
        print(f"Dropped {before_drop-after_drop} rows with no admit_date")

    # Compute day_index relative to admit_date (normalized)
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    # Coerce numeric-like candidate columns to numeric where possible
    numeric_cols_present = [c for c in candidate_numeric if c in chunk.columns]
    coerced_numeric = []
    for nc in numeric_cols_present:
        coerced = pd.to_numeric(chunk[nc], errors='coerce')
        # treat as numeric if a non-trivial fraction parse as numeric (>=1%)
        frac_num = coerced.notna().sum() / max(1, len(chunk))
        if frac_num > 0.01:
            chunk[nc + "_num"] = coerced
            coerced_numeric.append(nc)
        # else leave as text (we'll treat as text)

    # Group keys: day-level per admission
    grp_keys = ['subject_id','hadm_id','day_index']

    # For textual fields (and all fields as fallback) pick the last row by chartref (charttime)
    chunk_sorted = chunk.sort_values('chartref')
    grp_last = chunk_sorted.groupby(grp_keys, as_index=False).last()  # last row per field

    # For numeric columns we created "<col>_num" - take max per day
    if coerced_numeric:
        agg_spec = {nc + "_num": "max" for nc in coerced_numeric}
        grp_num = chunk.groupby(grp_keys, as_index=False).agg(agg_spec)
    else:
        grp_num = pd.DataFrame(columns=grp_keys)  # empty

    # Merge last/text and numeric max results
    if not grp_num.empty:
        merged_grps = pd.merge(grp_last, grp_num, on=grp_keys, how='left', suffixes=('','_nummax'))
    else:
        merged_grps = grp_last

    # Now determine final value per original column:
    # - if original column in coerced_numeric -> use the max from "<col>_num" (after merging)
    # - else use the last observed value (grp_last[col])
    assigned = 0
    for _, r in merged_grps.iterrows():
        key = (int(r['subject_id']), int(r['hadm_id']), int(r['day_index']))
        row_idx = merged_initial_index_map.get(key)
        if row_idx is None:
            continue
        # assign each original column value into merged_initial under prefix 'micro_<col>'
        for col in micro_cols:
            if col in ('subject_id','hadm_id'):
                continue
            out_col = prefix + col
            if col in coerced_numeric:
                # numeric stored as "<col>_num" in merged_grps (or "<col>_num" in chunk)
                numcol = col + "_num"
                val = r.get(numcol, pd.NA)
                # fallback: if no numeric, try textual column value
                if pd.isna(val):
                    val = r.get(col, pd.NA)
            else:
                val = r.get(col, pd.NA)
            # assign as-is (preserve strings/dates). Convert numpy types to python native where needed
            merged_initial.at[row_idx, out_col] = val
        assigned += 1

    total_assigned += assigned
    print(f"Chunk {chunk_no}: groups aggregated = {len(merged_grps)}, assigned rows = {assigned}, total_assigned so far = {total_assigned}")

    # cleanup
    del chunk, chunk_sorted, grp_last, grp_num, merged_grps
    gc.collect()

print("\n--- ALL CHUNKS PROCESSED ---")
print("Total assigned day-rows updated:", total_assigned)

# 6) Save out (chunked write to avoid memory surge)
n_rows = len(merged_initial)
write_chunk = 20000
first_out = True
for start in range(0, n_rows, write_chunk):
    end = min(start + write_chunk, n_rows)
    merged_initial.iloc[start:end].to_csv(out_merged_path, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False
    print(f"Saved rows {start}-{end-1}")
print("Saved final merged file with microbiologyevents to:", out_merged_path)
