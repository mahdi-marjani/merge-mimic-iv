import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

# --- Config / paths ---
drg_path = Path("drgcodes.csv")
merged_initial_path = Path("admissions_expanded.csv")   # or use DataFrame merged_initial if already in memory
out_path = Path("merged_with_drg.csv")

# --- Load merged_initial (admission x day rows) ---
if 'merged_initial' in globals():
    merged_initial = globals()['merged_initial']
else:
    if not merged_initial_path.exists():
        raise FileNotFoundError(f"{merged_initial_path} not found. Provide merged_initial.csv or have merged_initial DataFrame in memory.")
    merged_initial = pd.read_csv(merged_initial_path, parse_dates=['admittime','dischtime','deathtime'], low_memory=False)
# ensure keys have consistent types
merged_initial['subject_id'] = pd.to_numeric(merged_initial['subject_id'], errors='coerce').astype('Int64')
merged_initial['hadm_id'] = pd.to_numeric(merged_initial['hadm_id'], errors='coerce').astype('Int64')

# --- Load drgcodes (small) ---
if not drg_path.exists():
    raise FileNotFoundError(f"{drg_path} not found.")
drg = pd.read_csv(drg_path, low_memory=False)

# normalize / coerce types
drg['subject_id'] = pd.to_numeric(drg['subject_id'], errors='coerce').astype('Int64')
drg['hadm_id'] = pd.to_numeric(drg['hadm_id'], errors='coerce').astype('Int64')
# keep textual drg_code as string but strip
drg['drg_code'] = drg['drg_code'].astype(str).str.strip()
drg['description'] = drg['description'].astype(str).str.strip()
drg['drg_type'] = drg['drg_type'].astype(str).str.strip()

# numeric columns: coerce to numeric (float), keep NaN when missing
drg['drg_severity_num'] = pd.to_numeric(drg['drg_severity'], errors='coerce')
drg['drg_mortality_num'] = pd.to_numeric(drg['drg_mortality'], errors='coerce')

# --- Aggregation strategy ---
# 1) Compute numeric aggregates per admission (max)
numeric_aggs = drg.groupby(['subject_id','hadm_id'], dropna=False).agg({
    'drg_severity_num': 'max',
    'drg_mortality_num': 'max'
}).reset_index().rename(columns={
    'drg_severity_num': 'drg_severity_max',
    'drg_mortality_num': 'drg_mortality_max'
})

# 2) Choose single representative row per admission for textual fields.
# Build a sort key: (drg_severity_num desc, drg_mortality_num desc, drg_type priority desc)
type_priority = {'APR': 2, 'HCFA': 1}  # APR preferred over HCFA; others map to 0
drg['type_prio'] = drg['drg_type'].map(type_priority).fillna(0).astype(int)

# Replace NaN with very small number so numerics that exist always win
drg['_sev_for_sort'] = drg['drg_severity_num'].fillna(-9999)
drg['_mort_for_sort'] = drg['drg_mortality_num'].fillna(-9999)

drg_sorted = drg.sort_values(
    by=['subject_id','hadm_id','_sev_for_sort','_mort_for_sort','type_prio'],
    ascending=[True, True, False, False, False]
)

# pick first per group (best by our rule)
drg_rep = drg_sorted.groupby(['subject_id','hadm_id'], as_index=False).first()[[
    'subject_id','hadm_id','drg_type','drg_code','description','drg_severity','drg_mortality'
]].rename(columns={
    'drg_type': 'drg_type_chosen',
    'drg_code': 'drg_code_chosen',
    'description': 'drg_description_chosen',
    'drg_severity': 'drg_severity_chosen',
    'drg_mortality': 'drg_mortality_chosen'
})

# 3) merge numeric_aggs and textual representative row into a single drg_summary
drg_summary = numeric_aggs.merge(drg_rep, on=['subject_id','hadm_id'], how='left')

# Optional: if you want both numeric_max and chosen textual numeric (string),
# ensure columns consistent and cast types
# (drg_severity_max is float; drg_severity_chosen may be string original - try to coerce)
drg_summary['drg_severity_chosen'] = pd.to_numeric(drg_summary['drg_severity_chosen'], errors='coerce')

# --- Merge into merged_initial (broadcast to day rows) ---
merged_final = merged_initial.merge(
    drg_summary,
    on=['subject_id','hadm_id'],
    how='left',
    validate='m:1'  # many merged_initial rows (days) to one drg_summary row
)

# --- Save output ---
merged_final.to_csv(out_path, index=False)
print("Merged drgcodes -> written to:", out_path)
print("Rows with DRG info:", int(merged_final['drg_code_chosen'].notna().sum()))

# End of script
