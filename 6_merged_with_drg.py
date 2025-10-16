import pandas as pd
from pathlib import Path


drg_path = Path("drgcodes.csv")
merged_initial_path = Path("admissions_expanded.csv")   
out_path = Path("merged_with_drg.csv")


if 'merged_initial' in globals():
    merged_initial = globals()['merged_initial']
else:
    if not merged_initial_path.exists():
        raise FileNotFoundError(f"{merged_initial_path} not found. Provide merged_initial.csv or have merged_initial DataFrame in memory.")
    merged_initial = pd.read_csv(merged_initial_path, parse_dates=['admittime','dischtime','deathtime'], low_memory=False)

merged_initial['subject_id'] = pd.to_numeric(merged_initial['subject_id'], errors='coerce').astype('Int64')
merged_initial['hadm_id'] = pd.to_numeric(merged_initial['hadm_id'], errors='coerce').astype('Int64')


if not drg_path.exists():
    raise FileNotFoundError(f"{drg_path} not found.")
drg = pd.read_csv(drg_path, low_memory=False)


drg['subject_id'] = pd.to_numeric(drg['subject_id'], errors='coerce').astype('Int64')
drg['hadm_id'] = pd.to_numeric(drg['hadm_id'], errors='coerce').astype('Int64')

drg['drg_code'] = drg['drg_code'].astype(str).str.strip()
drg['description'] = drg['description'].astype(str).str.strip()
drg['drg_type'] = drg['drg_type'].astype(str).str.strip()


drg['drg_severity_num'] = pd.to_numeric(drg['drg_severity'], errors='coerce')
drg['drg_mortality_num'] = pd.to_numeric(drg['drg_mortality'], errors='coerce')



numeric_aggs = drg.groupby(['subject_id','hadm_id'], dropna=False).agg({
    'drg_severity_num': 'max',
    'drg_mortality_num': 'max'
}).reset_index().rename(columns={
    'drg_severity_num': 'drg_severity_max',
    'drg_mortality_num': 'drg_mortality_max'
})



type_priority = {'APR': 2, 'HCFA': 1}  
drg['type_prio'] = drg['drg_type'].map(type_priority).fillna(0).astype(int)


drg['_sev_for_sort'] = drg['drg_severity_num'].fillna(-9999)
drg['_mort_for_sort'] = drg['drg_mortality_num'].fillna(-9999)

drg_sorted = drg.sort_values(
    by=['subject_id','hadm_id','_sev_for_sort','_mort_for_sort','type_prio'],
    ascending=[True, True, False, False, False]
)


drg_rep = drg_sorted.groupby(['subject_id','hadm_id'], as_index=False).first()[[
    'subject_id','hadm_id','drg_type','drg_code','description','drg_severity','drg_mortality'
]].rename(columns={
    'drg_type': 'drg_type_chosen',
    'drg_code': 'drg_code_chosen',
    'description': 'drg_description_chosen',
    'drg_severity': 'drg_severity_chosen',
    'drg_mortality': 'drg_mortality_chosen'
})


drg_summary = numeric_aggs.merge(drg_rep, on=['subject_id','hadm_id'], how='left')




drg_summary['drg_severity_chosen'] = pd.to_numeric(drg_summary['drg_severity_chosen'], errors='coerce')


merged_final = merged_initial.merge(
    drg_summary,
    on=['subject_id','hadm_id'],
    how='left',
    validate='m:1'  
)


merged_final.to_csv(out_path, index=False)
print("Merged drgcodes -> written to:", out_path)
print("Rows with DRG info:", int(merged_final['drg_code_chosen'].notna().sum()))


