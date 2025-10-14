import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 140)

# فایل‌ها
merged_initial_file = Path("admissions_expanded.csv")   # یا admissions_expanded.csv طبق جریان قبلی
ditems_file = Path("d_items.csv")
datetimeevents_file = Path("datetimeevents.csv")

# پارامترها
chunksize = 500_000   # مثل کد قبلی، می‌توانی تغییر دهی

# بررسی وجود فایل‌ها
if not merged_initial_file.exists():
    raise FileNotFoundError(f"{merged_initial_file} not found. load or generate merged_initial first.")
if not ditems_file.exists():
    raise FileNotFoundError(f"{ditems_file} not found.")
if not datetimeevents_file.exists():
    raise FileNotFoundError(f"{datetimeevents_file} not found.")

# 1) بارگذاری merged_initial (ممکن است بزرگ باشد — اما ما برای افزودن ستون‌ها آن را در حافظه فرض می‌کنیم همانند نوت‌بوک تو)
print("Loading merged_initial (this may use substantial memory)...")
merged_initial = pd.read_csv(merged_initial_file, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'], low_memory=False)
n_rows = len(merged_initial)
print("merged_initial rows:", n_rows)

# 2) دریافت itemidهای مربوط به datetimeevents از d_items
print("Loading d_items and selecting itemids where linksto == 'datetimeevents' ...")
d = pd.read_csv(ditems_file, dtype=str, usecols=['itemid','linksto','label','abbreviation'])
d['linksto'] = d['linksto'].fillna('').astype(str)
dt_items = d.loc[d['linksto'].str.lower() == 'datetimeevents', 'itemid'].dropna().unique().tolist()
# convert to ints (where possible)
dt_itemids = []
for iid in dt_items:
    try:
        dt_itemids.append(int(iid))
    except:
        pass
dt_itemids = sorted(set(dt_itemids))
print(f"Found {len(dt_itemids)} datetimeevents itemids (sample):", dt_itemids[:20])

# 3) اضافه کردن ستون‌های itemid به merged_initial (نام ستون‌ها همان id به صورت رشته)
added = 0
for iid in dt_itemids:
    col = str(iid)
    if col not in merged_initial.columns:
        merged_initial[col] = pd.Series([pd.NA] * n_rows, dtype="object")
        added += 1
print(f"Added {added} new columns to merged_initial for datetimeitems. Total columns now: {len(merged_initial.columns)}")

# 4) ساختن admit_map: (subject_id,hadm_id) -> admit_date (normalized)
print("Building admit_date map from merged_initial ...")
admit_map = merged_initial.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time'], errors='coerce').dt.normalize()
admit_map['key'] = list(zip(admit_map['subject_id'].astype('Int64'), admit_map['hadm_id'].astype('Int64')))
admit_dict = dict(zip(admit_map['key'], admit_map['admit_date']))
print("Admit map entries:", len(admit_dict))

# 5) ساختن index map برای merged_initial: (subject_id,hadm_id,day_index) -> row index
print("Building merged_initial index map (for direct assignment) ...")
merged_initial_index_map = {}
for idx, row in merged_initial[['subject_id','hadm_id','day_index']].iterrows():
    try:
        key = (int(row['subject_id']), int(row['hadm_id']), int(row['day_index']))
        merged_initial_index_map[key] = idx
    except Exception:
        # skip rows with missing keys
        continue
print("Merged rows map size:", len(merged_initial_index_map))

# 6) خواندن datetimeevents چانک‌به‌چانک و نگاشت به day_index و aggregate
usecols = ['subject_id','hadm_id','stay_id','caregiver_id','charttime','storetime','itemid','value','valueuom','warning']
reader = pd.read_csv(datetimeevents_file, usecols=usecols, parse_dates=['charttime','storetime'], chunksize=chunksize, low_memory=True)

total_assigned = 0
chunk_no = 0

for chunk in reader:
    chunk_no += 1
    print(f"\n--- Processing datetimeevents chunk {chunk_no} (rows: {len(chunk)}) ---")
    # itemid numeric
    chunk['itemid'] = pd.to_numeric(chunk['itemid'], errors='coerce').astype('Int64')
    chunk = chunk[chunk['itemid'].notna()]
    if chunk.empty:
        print("no itemids in this chunk")
        continue

    # فیلتر فقط itemidهای مربوط به datetimeevents (از d_items)
    chunk = chunk[chunk['itemid'].isin(dt_itemids)]
    if chunk.empty:
        print("no relevant datetime itemids in this chunk")
        continue

    # تبدیل شناسه‌ها به int برای lookup
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
    # lookup admit_date
    keys = list(zip(chunk['subject_id'].astype('Int64'), chunk['hadm_id'].astype('Int64')))
    chunk['admit_date'] = [admit_dict.get(k, pd.NaT) for k in keys]

    # حذف ردیف‌هایی که admission match ندارند
    chunk = chunk[chunk['admit_date'].notna()].copy()
    if chunk.empty:
        print("no rows with admit_date in this chunk")
        continue

    # محاسبه day_index (بر اساس charttime normalize)
    chunk['chart_date'] = pd.to_datetime(chunk['charttime'], errors='coerce').dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    # تلاش برای پارس کردن مقدار value به datetime (اگر ممکن باشد)
    chunk['value_dt'] = pd.to_datetime(chunk['value'], errors='coerce')

    # اگر value_dt خالی است، می‌توانیم به عنوان fallback از charttime استفاده کنیم (اختیاری)
    # اینجا ما ترجیح می‌دهیم مقدار value_dt را اگر موجود باشد استفاده کنیم؛ در غیر اینصورت مقدار متنی را نگه می‌داریم.
    chunk['value_raw'] = chunk['value'].astype(str)

    # گروه‌بندی: برای هر (subject_id,hadm_id,day_index,itemid) آخرین مقدار بر اساس charttime را می‌گیریم
    grp_keys = ['subject_id','hadm_id','day_index','itemid']
    chunk_sorted = chunk.sort_values('charttime')
    grp_last = chunk_sorted.groupby(grp_keys, as_index=False).last()[grp_keys + ['value_dt','value_raw','charttime']]
    # فرمت نهایی: اگر value_dt موجود است آن را به رشتهٔ ISO ذخیره کن، وگرنه value_raw
    def make_final_val(r):
        if pd.notna(r.get('value_dt')):
            # تبدیل به ISO (بدون timezone)
            return pd.to_datetime(r['value_dt']).isoformat()
        else:
            v = r.get('value_raw')
            if pd.isna(v) or v in ("nan","None","NoneType","NA","<NA>"):
                return pd.NA
            return v

    grp_last['final_value'] = grp_last.apply(make_final_val, axis=1)

    # اکنون مقدارها را به merged_initial اختصاص می‌دهیم (همان روش قبلی: find row_idx و .at assignment)
    assigned = 0
    for _, r in grp_last.iterrows():
        try:
            key = (int(r['subject_id']), int(r['hadm_id']), int(r['day_index']))
        except Exception:
            continue
        row_idx = merged_initial_index_map.get(key)
        if row_idx is None:
            continue
        itemid_col = str(int(r['itemid']))
        val = r['final_value']
        merged_initial.at[row_idx, itemid_col] = val
        assigned += 1

    total_assigned += assigned
    print(f"Chunk {chunk_no}: groups aggregated = {len(grp_last)}, assigned = {assigned}, total_assigned so far = {total_assigned}")

    # پاکسازی
    del chunk, chunk_sorted, grp_last
    gc.collect()

print("\n--- ALL datetimeevents CHUNKS PROCESSED ---")
print("Total assigned datetime cells:", total_assigned)

# 7) ذخیرهٔ خروجی نهایی (chunked write برای حافظه دوستانه)
out_path = Path("merged_with_datetimeevents_filled.csv")
write_chunk = 20000
first = True
n_rows = len(merged_initial)
for start in range(0, n_rows, write_chunk):
    end = min(start + write_chunk, n_rows)
    merged_initial.iloc[start:end].to_csv(out_path, mode='w' if first else 'a', index=False, header=first)
    first = False
    print(f"Saved rows {start}-{end-1}")
print("Saved final to:", out_path)

# rename_datetimeevents_columns.py
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 140)

# فایل‌ها
ditems_path = Path("d_items.csv")
in_path = Path("merged_with_datetimeevents_filled.csv")
out_path = Path("merged_with_datetimeevents_filled_renamed.csv")

# پارامترها
chunksize = 20000
max_name_len = 80

# بررسی وجود فایل‌ها
if not ditems_path.exists():
    raise FileNotFoundError(f"{ditems_path} not found.")
if not in_path.exists():
    raise FileNotFoundError(f"{in_path} not found.")

# بارگذاری دیکشنری آیتم‌ها (فقط datetimeevents)
d = pd.read_csv(ditems_path, usecols=['itemid','label','abbreviation','linksto'], dtype=str)
d['linksto'] = d['linksto'].fillna('').astype(str)
d_dt = d.loc[d['linksto'].str.lower() == 'datetimeevents'].copy()
d_dt['itemid'] = d_dt['itemid'].astype(str).str.strip()
d_dt['label'] = d_dt['label'].fillna('').astype(str).str.strip()
d_dt['abbreviation'] = d_dt['abbreviation'].fillna('').astype(str).str.strip()

# انتخاب نام نهایی (abbreviation اگر هست، وگرنه label، وگرنه fallback)
d_dt['chosen'] = d_dt.apply(lambda r: r['abbreviation'] if r['abbreviation']!='' else (r['label'] if r['label']!='' else ''), axis=1)

def sanitize_name(s):
    if pd.isna(s) or s is None:
        return ''
    s = str(s).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-]', '', s)   # اجازه حروف/اعداد/underscore/dash
    s = re.sub(r'_+', '_', s)
    s = s[:max_name_len]
    return s

# ساخت map itemid -> chosen name (با تضمین یکتایی)
name_map = {}
used = set()
for _, row in d_dt.iterrows():
    iid = row['itemid']
    chosen = row['chosen']
    if chosen == '':
        base = f"item_{iid}"
    else:
        base = sanitize_name(chosen)
        if base == '':
            base = f"item_{iid}"
    name = base
    if name in used:
        name = f"{base}__{iid}"
    counter = 1
    while name in used:
        name = f"{base}__{iid}_{counter}"
        counter += 1
    used.add(name)
    name_map[iid] = name

# آماده‌سازی header جدید
orig_header = pd.read_csv(in_path, nrows=0).columns.tolist()
new_header = []
conflicts = 0

for col in orig_header:
    new_col = col
    col_str = str(col).strip()
    # اگر خودِ ستون دقیقا یک itemid از datetimeevents باشد، نامگذاری کنیم
    if col_str in name_map:
        new_col = "datetimeevents_" + name_map[col_str]
    else:
        # بعضی هدرها ممکنه به صورت عدد/float ذخیره شده باشند؛ تلاش کن به int تبدیل کنی و بررسی کنی
        try:
            icol = str(int(float(col_str)))
            if icol in name_map:
                new_col = "datetimeevents_" + name_map[icol]
        except Exception:
            pass
    # جلوگیری از تضاد نام‌ها در هدر جدید
    if new_col in new_header:
        conflicts += 1
        new_col = f"{new_col}__orig_{sanitize_name(col_str)}"
        k = 1
        while new_col in new_header:
            new_col = f"{new_col}_{k}"; k += 1
    new_header.append(new_col)

print(f"Prepared header mapping. Total cols: {len(orig_header)}, conflicts resolved: {conflicts}")
sample_map = {k: name_map[k] for k in list(name_map)[:10]}
print("sample itemid->name (first 10):", sample_map)

# نوشتن فایل با هدر جدید به صورت chunked
first = True
rows_written = 0
for i, chunk in enumerate(pd.read_csv(in_path, chunksize=chunksize, low_memory=False)):
    chunk.columns = new_header
    chunk.to_csv(out_path, mode='w' if first else 'a', index=False, header=first)
    first = False
    rows_written += len(chunk)
    print(f"Chunk {i+1}: wrote {len(chunk):,} rows (total {rows_written:,})")
    del chunk
    gc.collect()

print("✅ Done. Output saved to:", out_path)
print("Rows written:", rows_written)
