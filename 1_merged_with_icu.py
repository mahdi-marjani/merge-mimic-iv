import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

merged_initial = pd.read_csv("admissions_expanded.csv")

icu_path = Path("icustays.csv")
if not icu_path.exists():
    raise FileNotFoundError(f"icustays.csv not found at {icu_path.resolve()}  -- put the file next to admissions.csv")

icustays = pd.read_csv(icu_path, low_memory=False, parse_dates=['intime','outtime'])

for col in ['subject_id','hadm_id','intime','outtime']:
    if col not in icustays.columns:
        raise KeyError(f"Expected column '{col}' in icustays.csv but it is missing.")

optional_cols = ['stay_id','first_careunit','last_careunit','los']
for c in optional_cols:
    if c not in icustays.columns:
        icustays[c] = pd.NA

merged_with_icu = merged_initial.copy().reset_index(drop=False).rename(columns={'index':'row_id'})
for col in ['stay_id_icu','icustay_intime','icustay_outtime','first_careunit_icu','last_careunit_icu','los_icu']:
    if col not in merged_with_icu.columns:
        merged_with_icu[col] = pd.NA

if 'admittime' not in merged_with_icu.columns:
    raise KeyError("merged_initial must contain 'admittime' column")

merged_with_icu['admittime'] = pd.to_datetime(merged_with_icu['admittime'], errors='coerce')
merged_with_icu['day_index_int'] = merged_with_icu['day_index'].fillna(0).astype(int)
merged_with_icu['row_date'] = merged_with_icu['admittime'].dt.normalize() + pd.to_timedelta(merged_with_icu['day_index_int'], unit='D')

icustays['intime'] = pd.to_datetime(icustays['intime'], errors='coerce')
icustays['outtime'] = pd.to_datetime(icustays['outtime'], errors='coerce').fillna(icustays['intime'])
icustays['intime_norm'] = icustays['intime'].dt.normalize()
icustays['outtime_norm'] = icustays['outtime'].dt.normalize()

icu_keep = ['subject_id','hadm_id','stay_id','intime','outtime','intime_norm','outtime_norm','first_careunit','last_careunit','los']
candidate = merged_with_icu.merge(icustays[icu_keep], on=['subject_id','hadm_id'], how='left', suffixes=('','_icu'))

mask_in_icu = (candidate['row_date'] >= candidate['intime_norm']) & (candidate['row_date'] <= candidate['outtime_norm'])
candidate['in_icu'] = mask_in_icu.fillna(False)

matched = candidate[candidate['in_icu']].copy()
if not matched.empty:
    matched = matched.sort_values(by=['row_id','intime'])
    first_matches = matched.groupby('row_id', as_index=False).first()
    map_cols = {
        'stay_id':'stay_id_icu',
        'intime':'icustay_intime',
        'outtime':'icustay_outtime',
        'first_careunit':'first_careunit_icu',
        'last_careunit':'last_careunit_icu',
        'los':'los_icu'
    }
    for src, dst in map_cols.items():
        mapping = first_matches.set_index('row_id')[src]
        merged_with_icu.loc[merged_with_icu['row_id'].isin(mapping.index), dst] = merged_with_icu.loc[merged_with_icu['row_id'].isin(mapping.index), 'row_id'].map(mapping)
    assigned_count = len(first_matches)
else:
    assigned_count = 0

merged_with_icu = merged_with_icu.drop(columns=['day_index_int','row_date'])

print(f"ICU assignment complete. Rows where ICU info filled: {int(assigned_count)}")
print(merged_with_icu[merged_with_icu['stay_id_icu'].notna()].head(20))

rename_map = {
    'stay_id_icu': 'stay_id',
    'first_careunit_icu': 'first_careunit',
    'last_careunit_icu': 'last_careunit',
    'icustay_intime': 'icustays_intime',
    'icustay_outtime': 'icustays_outtime',
    'los_icu': 'los'
}
merged_with_icu = merged_with_icu.rename(columns=rename_map)

icu_cols_ordered = ['stay_id', 'first_careunit', 'last_careunit', 'icustays_intime', 'icustays_outtime', 'los']

other_cols = [c for c in merged_with_icu.columns if c not in icu_cols_ordered]

merged_with_icu = merged_with_icu[other_cols + icu_cols_ordered]

print("Renaming & reordering complete.")
print(merged_with_icu[icu_cols_ordered].head(20))

merged_with_icu.to_csv('merged_with_icu.csv')