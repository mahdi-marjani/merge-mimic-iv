import pandas as pd
import numpy as np
from pathlib import Path

merged_with_icu = pd.read_csv("merged_with_icu.csv")

vanco_path = Path("all_vanco.csv")
if not vanco_path.exists():
    raise FileNotFoundError(f"all_vanco.csv not found at {vanco_path.resolve()}")

all_vanco = pd.read_csv(vanco_path, low_memory=False, parse_dates=['charttime'])

all_vanco['subject_id'] = pd.to_numeric(all_vanco['subject_id'], errors='coerce').astype('Int64')
all_vanco['hadm_id'] = pd.to_numeric(all_vanco['hadm_id'], errors='coerce').astype('Int64')

def resolve_numeric(row):
    v = row.get('value')
    vn = row.get('valuenum')
    if pd.isna(v) or str(v).strip() in ['', '___', 'NaN', 'nan']:
        try:
            return float(vn) if not pd.isna(vn) else np.nan
        except:
            return np.nan
    s = str(v).strip().replace(',', '')
    try:
        return float(s)
    except:
        try:
            return float(vn) if not pd.isna(vn) else np.nan
        except:
            return np.nan

all_vanco['resolved_val'] = all_vanco.apply(resolve_numeric, axis=1)

merged_with_vanco = merged_with_icu.copy()
if 'admittime' not in merged_with_vanco.columns:
    raise KeyError(f"merged_with_vanco must contain 'admittime' column before merging labs.")
merged_with_vanco['admittime'] = pd.to_datetime(merged_with_vanco['admittime'], errors='coerce')

admit_map = merged_with_vanco.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time']).dt.normalize()

all_vanco = all_vanco.merge(admit_map[['subject_id','hadm_id','admit_date']], on=['subject_id','hadm_id'], how='left')

missing_admit = all_vanco['admit_date'].isna().sum()
if missing_admit:
    print(f"Warning: {missing_admit} all_vanco rows have no matching admission (admit_date missing) and will be skipped.")
all_vanco = all_vanco[all_vanco['admit_date'].notna()].copy()

all_vanco['chart_date'] = pd.to_datetime(all_vanco['charttime'], errors='coerce').dt.normalize()
all_vanco['day_index_lab'] = (all_vanco['chart_date'] - all_vanco['admit_date']).dt.days.fillna(0).astype(int)
all_vanco.loc[all_vanco['day_index_lab'] < 0, 'day_index_lab'] = 0

group_cols = ['subject_id','hadm_id','day_index_lab']
usable = all_vanco[~all_vanco['resolved_val'].isna()].copy()
if usable.empty:
    print("No usable numeric vanco values found to aggregate.")
    daily_vanco = pd.DataFrame(columns=['subject_id','hadm_id','day_index_lab',
                                       'charttime','value','valuenum','valueuom','flag','resolved_val'])
else:
    idx = usable.groupby(group_cols)['resolved_val'].idxmax()
    daily_vanco = usable.loc[idx].copy()

daily_vanco = daily_vanco.rename(columns={
    'charttime':'all_vanco_charttime',
    'value':'all_vanco_value',
    'valuenum':'all_vanco_valuenum',
    'valueuom':'all_vanco_valueuom',
    'flag':'all_vanco_flag',
    'day_index_lab':'day_index'
})

merge_cols = ['subject_id','hadm_id','day_index',
              'all_vanco_charttime','all_vanco_value','all_vanco_valuenum','all_vanco_valueuom','all_vanco_flag']
daily_vanco = daily_vanco[merge_cols]

daily_vanco['subject_id'] = daily_vanco['subject_id'].astype('Int64')
daily_vanco['hadm_id'] = daily_vanco['hadm_id'].astype('Int64')
daily_vanco['day_index'] = daily_vanco['day_index'].astype('Int64')

merged_with_vanco = merged_with_vanco.merge(daily_vanco, on=['subject_id','hadm_id','day_index'], how='left')

print(f"all_vanco merged -> rows with vanco info: {int(merged_with_vanco['all_vanco_charttime'].notna().sum())}")

preview = merged_with_vanco[merged_with_vanco['all_vanco_charttime'].notna()].head(20)
print(preview[['subject_id','hadm_id','day_index',
               'all_vanco_charttime','all_vanco_value','all_vanco_valuenum','all_vanco_valueuom','all_vanco_flag']])


merged_with_vanco.to_csv('merged_with_vanco.csv')