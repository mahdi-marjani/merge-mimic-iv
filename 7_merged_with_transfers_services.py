import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

merged_initial = pd.read_csv("admissions_expanded.csv")

# Config - tune these to your environment
MERGED_IN_PATH = Path("admissions_expanded.csv")   # or use Memory DataFrame merged_initial
TRANSFERS_PATH   = Path("transfers.csv")
SERVICES_PATH    = Path("services.csv")
OUT_PATH         = Path("merged_with_transfers_services.csv")

TRANS_CHUNKSIZE = 200_000   # transfers often smaller; safe default
SERV_CHUNKSIZE  = 200_000

# Load or reference merged_initial
if 'merged_initial' in globals():
    merged = merged_initial.copy()
else:
    if not MERGED_IN_PATH.exists():
        raise FileNotFoundError(f"{MERGED_IN_PATH} not found. Provide merged_initial DataFrame or file.")
    merged = pd.read_csv(MERGED_IN_PATH, parse_dates=['admittime'], low_memory=False)
print("Loaded merged rows:", len(merged))

# Prepare new columns (if not present)
new_cols = [
    'transfers_eventtype', 'transfers_careunit', 'transfers_intime', 'transfers_outtime',
    'services_prev_service', 'services_curr_service', 'services_transfertime'
]
for c in new_cols:
    if c not in merged.columns:
        merged[c] = pd.NA

# Build admit_map: (subject_id, hadm_id) -> admit_date (normalized)
admit_map = merged.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time'], errors='coerce').dt.normalize()
admit_map['key'] = list(zip(admit_map['subject_id'].astype('Int64'), admit_map['hadm_id'].astype('Int64')))
admit_dict = dict(zip(admit_map['key'], admit_map['admit_date']))
print("Admit map size:", len(admit_dict))

# Build index map for merged rows: (subject_id, hadm_id, day_index) -> row index
merged_index_map = {}
for idx, row in merged[['subject_id','hadm_id','day_index']].iterrows():
    try:
        k = (int(row['subject_id']), int(row['hadm_id']), int(row['day_index']))
    except Exception:
        # skip malformed keys
        continue
    merged_index_map[k] = idx
print("Merged rows map size:", len(merged_index_map))

########################
# Process transfers.csv
########################
if not TRANSFERS_PATH.exists():
    raise FileNotFoundError(f"{TRANSFERS_PATH} not found.")

# We'll collect first-match per (row_idx) via assignment - to mimic your ICU first-match semantics,
# we process transfers sorted by intime so earliest covering transfer is assigned first.
trans_reader = pd.read_csv(TRANSFERS_PATH, parse_dates=['intime','outtime'], chunksize=TRANS_CHUNKSIZE, low_memory=True)

total_assigned_transfers = 0
chunk_no = 0
for chunk in trans_reader:
    chunk_no += 1
    # keep relevant columns
    chunk = chunk[['subject_id','hadm_id','transfer_id','eventtype','careunit','intime','outtime']].copy()
    # coerce ids
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce')
    # normalize times
    chunk['intime'] = pd.to_datetime(chunk['intime'], errors='coerce')
    chunk['outtime'] = pd.to_datetime(chunk['outtime'], errors='coerce')
    chunk = chunk.dropna(subset=['subject_id','hadm_id','intime'])  # need atleast intime
    if chunk.empty:
        continue

    # sort by intime ascending so earliest transfers processed first
    chunk = chunk.sort_values('intime')

    # iterate rows (transfers per admission are usually few; OK to loop)
    assigned = 0
    for _, tr in chunk.iterrows():
        s = int(tr['subject_id'])
        h = int(tr['hadm_id'])
        admit_date = admit_dict.get((s,h), pd.NaT)
        if pd.isna(admit_date):
            continue
        intime = tr['intime']
        outtime = tr['outtime'] if pd.notna(tr['outtime']) else tr['intime']
        intime_norm = intime.normalize()
        outtime_norm = pd.to_datetime(outtime).normalize()
        di_start = int(max(0, (intime_norm - admit_date).days))
        di_end   = int(max(0, (outtime_norm - admit_date).days))
        for di in range(di_start, di_end + 1):
            key = (s, h, di)
            row_idx = merged_index_map.get(key)
            if row_idx is None:
                continue
            existing_val = merged.at[row_idx, 'transfers_careunit']
            if pd.notna(existing_val):
                continue
            merged.at[row_idx, 'transfers_eventtype'] = tr.get('eventtype', pd.NA)
            merged.at[row_idx, 'transfers_careunit'] = tr.get('careunit', pd.NA)
            merged.at[row_idx, 'transfers_intime'] = intime
            merged.at[row_idx, 'transfers_outtime'] = outtime if pd.notna(tr.get('outtime')) else pd.NA
            assigned += 1
    total_assigned_transfers += assigned
    print(f"Transfers chunk {chunk_no}: assigned {assigned} rows (total assigned so far: {total_assigned_transfers})")
    del chunk
    gc.collect()

print("Total transfer assignments:", total_assigned_transfers)

########################
# Process services.csv (last-per-day semantics)
########################
if not SERVICES_PATH.exists():
    raise FileNotFoundError(f"{SERVICES_PATH} not found.")

# We'll keep a dict for best (latest) service per (subject,hadm,day_index)
service_best = {}  # key -> (transfertime, prev_service, curr_service)

serv_reader = pd.read_csv(SERVICES_PATH, parse_dates=['transfertime'], chunksize=SERV_CHUNKSIZE, low_memory=True)
chunk_no = 0
for chunk in serv_reader:
    chunk_no += 1
    # ensure columns exist
    chunk = chunk[['subject_id','hadm_id','transfertime','prev_service','curr_service']].copy()
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce')
    chunk['transfertime'] = pd.to_datetime(chunk['transfertime'], errors='coerce')
    chunk = chunk.dropna(subset=['subject_id','hadm_id','transfertime'])
    if chunk.empty:
        continue

    # iterate rows - services per admission relatively small
    for _, srow in chunk.iterrows():
        s = int(srow['subject_id'])
        h = int(srow['hadm_id'])
        key_admit = (s,h)
        admit_date = admit_dict.get(key_admit, pd.NaT)
        if pd.isna(admit_date):
            continue
        t = srow['transfertime']
        day_idx = int(max(0, (t.normalize() - admit_date).days))
        map_key = (s, h, day_idx)
        existing = service_best.get(map_key)
        # pick the later transfertime on that day
        if existing is None or (pd.notna(existing[0]) and t > existing[0]):
            service_best[map_key] = (t, srow.get('prev_service', pd.NA), srow.get('curr_service', pd.NA))

    del chunk
    gc.collect()
    print(f"Services chunk {chunk_no} processed; current cached service keys: {len(service_best)}")

# Apply service_best into merged DataFrame
assigned_services = 0
for map_key, (t, prev_s, curr_s) in service_best.items():
    row_idx = merged_index_map.get(map_key)
    if row_idx is None:
        continue
    merged.at[row_idx, 'services_prev_service'] = prev_s if prev_s is not None else pd.NA
    merged.at[row_idx, 'services_curr_service'] = curr_s if curr_s is not None else pd.NA
    merged.at[row_idx, 'services_transfertime'] = t
    assigned_services += 1

print("Total service rows applied:", assigned_services)

# Finalize types (optional)
for c in ['transfers_intime','transfers_outtime','services_transfertime']:
    merged[c] = pd.to_datetime(merged[c], errors='coerce')

# Write out in chunks to avoid memory spikes
write_chunk = 20_000
first = True
n_rows = len(merged)
for start in range(0, n_rows, write_chunk):
    end = min(start + write_chunk, n_rows)
    merged.iloc[start:end].to_csv(OUT_PATH, mode='w' if first else 'a', index=False, header=first)
    first = False
    print(f"Saved rows {start}-{end-1}")
print("Saved merged file to:", OUT_PATH)
