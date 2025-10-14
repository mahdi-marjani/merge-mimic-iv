import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

merged_initial = pd.read_csv("admissions_expanded.csv")

chartevents_path = Path("chartevents.csv")
chartevents = pd.read_csv(chartevents_path, nrows=1000)

chartevents_path = Path("chartevents.csv")
if not chartevents_path.exists():
    raise FileNotFoundError(f"chartevents.csv not found at {chartevents_path.resolve()}")

usecols = ['subject_id','hadm_id','itemid','charttime','value','valuenum']
chunksize = 200_000

total_counts = defaultdict(int)
present_counts = defaultdict(int)

reader = pd.read_csv(chartevents_path, usecols=usecols, chunksize=chunksize, low_memory=True)

chunk_i = 0
for chunk in reader:
    chunk_i += 1
    chunk['itemid'] = pd.to_numeric(chunk['itemid'], errors='coerce').astype('Int64')
    chunk = chunk[chunk['itemid'].notna()]
    if chunk.empty:
        continue

    present_mask = ~chunk['valuenum'].isna()

    need_check = chunk['valuenum'].isna()
    if need_check.any():
        vals = chunk.loc[need_check, 'value'].astype(str).str.strip()
        good = ~vals.isin(["", "___", "NaN", "nan", "None", "none"])
        present_mask.loc[need_check] = good.values

    grp_total = chunk.groupby('itemid').size()
    grp_present = present_mask.groupby(chunk['itemid']).sum()

    for item, cnt in grp_total.items():
        total_counts[int(item)] += int(cnt)
    for item, cnt in grp_present.items():
        if pd.isna(item):
            continue
        present_counts[int(item)] += int(cnt)

    if chunk_i % 10 == 0:
        print(f"Processed {chunk_i*chunksize:,} rows...")

itemids = sorted(set(list(total_counts.keys()) + list(present_counts.keys())))
rows = []
for iid in itemids:
    tot = total_counts.get(iid, 0)
    pres = present_counts.get(iid, 0)
    miss = tot - pres
    frac = pres / tot if tot > 0 else 0.0
    rows.append((iid, tot, pres, miss, frac))

missingness_df = pd.DataFrame(rows, columns=['itemid','total_count','present_count','missing_count','present_fraction'])
missingness_df = missingness_df.sort_values(by='present_fraction', ascending=False).reset_index(drop=True)
missingness_df.to_csv("chartevents_itemid_missingness.csv", index=False)
print("Saved chartevents_itemid_missingness.csv")

df = pd.read_csv("chartevents_itemid_missingness.csv")

drop_ids = df.loc[df['present_fraction'] < 0.5, 'itemid'].tolist()

print(f"تعداد itemid هایی که باید drop بشن: {len(drop_ids)}")
print(drop_ids[:50])

reader = pd.read_csv("chartevents.csv", chunksize=2_000_000)
out_path = "chartevents_missing50_dropped.csv"

first = True
total_rows = 0
total_dropped = 0
total_written = 0

for i, chunk in enumerate(reader, start=1):
    before = len(chunk)
    filtered = chunk.loc[~chunk['itemid'].isin(drop_ids)]
    after = len(filtered)
    
    filtered.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
    first = False
    
    total_rows += before
    total_dropped += before - after
    total_written += after
    print(f"Chunk {i}: rows={before:,}, dropped={before - after:,}, kept={after:,}")

print("---- DONE ----")
print(f"Total rows processed: {total_rows:,}")
print(f"Total rows dropped:   {total_dropped:,}")
print(f"Total rows written:   {total_written:,}")
print("✅ فایل نهایی ذخیره شد:", out_path)

df = pd.read_csv("d_items.csv")

print(df.head(0))

in_path = "d_items.csv"
out_path = "d_items_chartevents_missing50_dropped.csv"

df = pd.read_csv(in_path)

filtered = df.loc[
    (df["linksto"] == "chartevents") & 
    (~df["itemid"].isin(drop_ids))
]

filtered.to_csv(out_path, index=False)

print("✅ d_items filtered and saved:", out_path)
print("before:", len(df), "after:", len(filtered), "drop:", len(df) - len(filtered))

ditems_path = Path("d_items_chartevents_missing50_dropped.csv")
merged_initial_file = Path("merged_initial.csv")
out_path = Path("merged_initial_with_items_cols.csv")

ditems = pd.read_csv(ditems_path, usecols=['itemid'])
itemids = pd.to_numeric(ditems['itemid'], errors='coerce').dropna().astype(int).unique().tolist()
cols_to_add = [str(i) for i in itemids]
print("Will add columns (count):", len(cols_to_add))

try:
    merged_initial
    print("Using merged_initial from memory (existing DataFrame). rows:", len(merged_initial))
except NameError:
    if not merged_initial_file.exists():
        raise FileNotFoundError(f"merged_initial not in memory and file {merged_initial_file} not found.")
    print("Loading merged_initial from disk:", merged_initial_file)
    merged_initial = pd.read_csv(merged_initial_file, low_memory=False, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'])
    print("Loaded merged_initial rows:", len(merged_initial))

n_rows = len(merged_initial)
added = 0
for c in cols_to_add:
    if c not in merged_initial.columns:
        merged_initial[c] = pd.Series([pd.NA] * n_rows, dtype="object")
        added += 1
print(f"Added {added} new columns. Total columns now: {len(merged_initial.columns)}")

chunksize = 10000
first = True
written = 0
for start in range(0, n_rows, chunksize):
    end = min(start + chunksize, n_rows)
    chunk = merged_initial.iloc[start:end]
    chunk.to_csv(out_path, mode="w" if first else "a", index=False, header=first)
    first = False
    written += len(chunk)
    print(f"Wrote rows {start:,}..{end-1:,} -> {len(chunk):,} rows")
    del chunk
    gc.collect()

print("✅ Done. Output saved to:", out_path)
print("Rows written:", written, "Columns in output:", len(merged_initial.columns))

print(merged_initial.head(3))

chartevents_path = Path("chartevents_missing50_dropped.csv")
ditems_path = Path("d_items_chartevents_missing50_dropped.csv")
merged_initial_file = None
chunksize = 500_000
save_after = False

if not chartevents_path.exists():
    raise FileNotFoundError(chartevents_path)
if not ditems_path.exists():
    raise FileNotFoundError(ditems_path)

ditems = pd.read_csv(ditems_path, usecols=['itemid'])
keep_itemids = pd.to_numeric(ditems['itemid'], errors='coerce').dropna().astype(int).unique().tolist()
keep_itemids_set = set(keep_itemids)
print("Keep itemids count:", len(keep_itemids))

try:
    merged_initial
except NameError:
    if merged_initial_file is None:
        raise NameError("merged_initial not in memory. Set merged_initial_file path or load it.")
    print("Loading merged_initial from disk...")
    merged_initial = pd.read_csv(merged_initial_file, low_memory=False, parse_dates=['admittime'])
    print("loaded merged_initial rows:", len(merged_initial))

for iid in keep_itemids:
    col = str(iid)
    if col not in merged_initial.columns:
        merged_initial[col] = pd.Series([pd.NA] * len(merged_initial), dtype="object")

admit_map = merged_initial.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time'], errors='coerce').dt.normalize()
admit_map['key'] = list(zip(admit_map['subject_id'].astype('Int64'), admit_map['hadm_id'].astype('Int64')))
admit_dict = dict(zip(admit_map['key'], admit_map['admit_date']))

merged_initial_index_map = {}
for idx, row in merged_initial[['subject_id','hadm_id','day_index']].iterrows():
    key = (int(row['subject_id']), int(row['hadm_id']), int(row['day_index']))
    merged_initial_index_map[key] = idx

print("Admit map keys:", len(admit_dict), "merged rows map size:", len(merged_initial_index_map))

reader = pd.read_csv(chartevents_path, usecols=['subject_id','hadm_id','itemid','charttime','value','valuenum'],
                     parse_dates=['charttime'], chunksize=chunksize, low_memory=True)

total_assigned = 0
chunk_no = 0

for chunk in reader:
    chunk_no += 1
    print(f"\n--- Processing chunk {chunk_no} (rows: {len(chunk)}) ---")
    chunk['itemid'] = pd.to_numeric(chunk['itemid'], errors='coerce').astype('Int64')
    chunk = chunk[chunk['itemid'].notna()]
    chunk = chunk[chunk['itemid'].isin(keep_itemids)]
    if chunk.empty:
        print("no relevant itemids in this chunk")
        continue

    chunk['subject_id'] = chunk['subject_id'].astype(int)
    chunk['hadm_id'] = chunk['hadm_id'].astype(int)

    def lookup_admit_date(s):
        return admit_dict.get((int(s.subject_id), int(s.hadm_id)), pd.NaT)
    keys = list(zip(chunk['subject_id'].astype(int), chunk['hadm_id'].astype(int)))
    chunk['admit_date'] = [admit_dict.get(k, pd.NaT) for k in keys]

    chunk = chunk[chunk['admit_date'].notna()]
    if chunk.empty:
        print("no rows with admit_date in this chunk")
        continue

    chunk['chart_date'] = chunk['charttime'].dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    chunk['numeric_val'] = pd.to_numeric(chunk['valuenum'], errors='coerce')
    mask_num_missing = chunk['numeric_val'].isna()
    if mask_num_missing.any():
        parsed = pd.to_numeric(chunk.loc[mask_num_missing, 'value'].astype(str).str.replace(',',''), errors='coerce')
        chunk.loc[mask_num_missing, 'numeric_val'] = parsed

    chunk['value_raw'] = chunk['value'].astype(str)

    grp_keys = ['subject_id','hadm_id','day_index','itemid']

    numeric_rows = chunk[chunk['numeric_val'].notna()].copy()
    if not numeric_rows.empty:
        grp_num = numeric_rows.groupby(grp_keys, as_index=False)['numeric_val'].max()
        grp_num = grp_num.rename(columns={'numeric_val':'agg_value_num'})
    else:
        grp_num = pd.DataFrame(columns=grp_keys + ['agg_value_num'])

    chunk_sorted = chunk.sort_values('charttime')
    grp_last = chunk_sorted.groupby(grp_keys, as_index=False).last()[grp_keys + ['value_raw','charttime']]
    grp_last = grp_last.rename(columns={'value_raw':'agg_value_text', 'charttime':'agg_time_text'})

    merged_grps = pd.merge(grp_last, grp_num, on=grp_keys, how='left')

    def pick_final_val(row):
        if pd.notna(row.get('agg_value_num')):
            return row['agg_value_num']
        else:
            v = row.get('agg_value_text')
            if pd.isna(v) or v in ("nan","None","NoneType","NA","<NA>"):
                return pd.NA
            return v

    merged_grps['final_value'] = merged_grps.apply(pick_final_val, axis=1)

    assigned = 0
    for _, r in merged_grps.iterrows():
        key = (int(r['subject_id']), int(r['hadm_id']), int(r['day_index']))
        row_idx = merged_initial_index_map.get(key)
        if row_idx is None:
            continue
        itemid_col = str(int(r['itemid']))
        val = r['final_value']
        merged_initial.at[row_idx, itemid_col] = val
        assigned += 1

    total_assigned += assigned
    print(f"Chunk {chunk_no}: groups aggregated = {len(merged_grps)}, assigned = {assigned}, total_assigned so far = {total_assigned}")

    del chunk, chunk_sorted, numeric_rows, grp_num, grp_last, merged_grps
    gc.collect()

print("\n--- ALL CHUNKS PROCESSED ---")
print("Total assigned cells:", total_assigned)

out_path = Path("merged_with_chartevents_filled.csv")
n_rows = len(merged_initial)
write_chunk = 20000
first = True
for start in range(0, n_rows, write_chunk):
    end = min(start + write_chunk, n_rows)
    merged_initial.iloc[start:end].to_csv(out_path, mode='w' if first else 'a', index=False, header=first)
    first = False
    print(f"Saved rows {start}-{end-1}")
print("Saved final to:", out_path)

merged_with_chartevents_filled_path = Path("merged_with_chartevents_filled.csv")
merged_with_chartevents_filled = pd.read_csv(merged_with_chartevents_filled_path, nrows=500)

input_path = Path("chartevents_missing50_dropped.csv")
output_path = Path("chartevents_missing50_dropped_filtered_hadm_id_23282506.csv")

chunksize = 2_000_000

first = True
total_rows = 0

for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunksize, low_memory=False)):
    filtered = chunk[chunk['hadm_id'] == 23282506]
    if not filtered.empty:
        filtered.to_csv(output_path, mode='w' if first else 'a',
                        index=False, header=first)
        first = False
        total_rows += len(filtered)
        print(f"Chunk {i}: wrote {len(filtered)} rows (total so far: {total_rows})")

print("Done! Final rows written:", total_rows)
print("Output file:", output_path)

ditems_path = Path("d_items_chartevents_missing50_dropped.csv")
in_path = Path("merged_with_chartevents_filled.csv")
out_path = Path("merged_with_chartevents_filled_renamed.csv")
chunksize = 20_000
max_name_len = 80

if not ditems_path.exists():
    raise FileNotFoundError(ditems_path)
if not in_path.exists():
    raise FileNotFoundError(in_path)

d = pd.read_csv(ditems_path, usecols=['itemid','label','abbreviation'], dtype=str)
d['itemid'] = d['itemid'].str.strip()
d['label'] = d['label'].fillna('').astype(str).str.strip()
d['abbreviation'] = d['abbreviation'].fillna('').astype(str).str.strip()

d['chosen'] = d.apply(lambda r: r['abbreviation'] if r['abbreviation']!='' else (r['label'] if r['label']!='' else ''), axis=1)

def sanitize_name(s):
    if pd.isna(s) or s is None:
        return ''
    s = str(s).strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-]', '', s)
    s = re.sub(r'_+', '_', s)
    s = s[:max_name_len]
    return s

name_map = {}
used = set()

for _, row in d.iterrows():
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
    name_map[str(iid)] = name

orig_header = pd.read_csv(in_path, nrows=0).columns.tolist()
new_header = []
conflicts = 0
for col in orig_header:
    new_col = col
    col_str = str(col).strip()
    if col_str in name_map:
        new_col = "chartevents_" + name_map[col_str]
    else:
        try:
            icol = str(int(float(col_str)))
            if icol in name_map:
                new_col = name_map[icol]
        except Exception:
            pass
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

merged_with_chartevents_filled_renamed_path = Path("merged_with_chartevents_filled_renamed.csv")
merged_with_chartevents_filled_renamed = pd.read_csv(merged_with_chartevents_filled_renamed_path, nrows=500)
