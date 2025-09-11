import pandas as pd
adm = pd.read_csv('admissions.csv', parse_dates=['admittime','dischtime'])
# آیا تعداد ردیف == تعداد hadm_id یکتا؟
print(len(adm), adm['hadm_id'].nunique())
# نشان دادن ردیف‌هایی که hadm_id تکراری دارند
duplicates = adm[adm.duplicated('hadm_id', keep=False)].sort_values('hadm_id')
print(duplicates.head(20))
