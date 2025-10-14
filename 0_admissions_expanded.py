import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 120)

csv_path = Path("admissions.csv")

admissions = pd.read_csv(csv_path, low_memory=False,
                            parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'])
print("Loaded admissions.csv from disk. Rows:", len(admissions))

admissions['admittime'] = pd.to_datetime(admissions['admittime'], errors='coerce')
admissions['dischtime'] = pd.to_datetime(admissions['dischtime'], errors='coerce')

admissions['dischtime'] = admissions['dischtime'].fillna(admissions['admittime'])

def expand_admission_row(row):
    adm_date = row['admittime'].normalize().date()
    dis_date = row['dischtime'].normalize().date()
    n_days = (dis_date - adm_date).days + 1
    if n_days <= 0:
        n_days = 1
    rows = []
    for d in range(n_days):
        new = {
            'subject_id': row['subject_id'],
            'hadm_id': row['hadm_id'],
            'day_index': int(d),
            'admittime': row['admittime'],
            'dischtime': row['dischtime'],
            'deathtime': row['deathtime'],
            'admission_type': row.get('admission_type', np.nan),
            'admit_provider_id': row.get('admit_provider_id', np.nan),
            'admission_location': row.get('admission_location', np.nan),
            'discharge_location': row.get('discharge_location', np.nan),
            'insurance': row.get('insurance', np.nan),
            'language': row.get('language', np.nan),
            'marital_status': row.get('marital_status', np.nan),
            'race': row.get('race', np.nan),
            'edregtime': row.get('edregtime', pd.NaT),
            'edouttime': row.get('edouttime', pd.NaT),
            'hospital_expire_flag': row.get('hospital_expire_flag', np.nan)
        }
        rows.append(new)
    return rows

expanded = []
for _, r in admissions.iterrows():
    expanded.extend(expand_admission_row(r))

merged_initial = pd.DataFrame(expanded)
merged_initial[['subject_id','hadm_id','day_index']] = merged_initial[['subject_id','hadm_id','day_index']].astype('Int64')

print("Expanded admissions -> rows:", merged_initial.shape[0])

merged_initial.to_csv("admissions_expanded.csv", index=False)