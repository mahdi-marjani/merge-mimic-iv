import pandas as pd
from pathlib import Path
import gc


procedures_path = Path("procedures_icd.csv")
dprocedures_path = Path("d_icd_procedures.csv")
merged_initial_path = Path("admissions_expanded.csv")
intermediate_chunks_path = Path("procedures_daily_chunks.csv")
final_daily_path = Path("procedures_daily_final.csv")
out_merged_path = Path("merged_with_procedures.csv")

chunksize = 500_000   
write_chunk = 20000


for p in (procedures_path, dprocedures_path, merged_initial_path):
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {p.resolve()}")


print("Loading merged_initial admissions (admit_time -> admit_date map)...")
mi_cols = ['subject_id', 'hadm_id', 'admittime']
mi = pd.read_csv(merged_initial_path, usecols=mi_cols, parse_dates=['admittime'], low_memory=False)
mi['subject_id'] = pd.to_numeric(mi['subject_id'], errors='coerce').astype('Int64')
mi['hadm_id'] = pd.to_numeric(mi['hadm_id'], errors='coerce').astype('Int64')


admit_map = mi.groupby(['subject_id', 'hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
admit_map['admit_date'] = pd.to_datetime(admit_map['admit_time'], errors='coerce').dt.normalize()


admit_dict = {}
for r in admit_map.itertuples(index=False):
    try:
        key = (int(r.subject_id), int(r.hadm_id))
    except Exception:
        continue
    admit_dict[key] = r.admit_date
print("Admit map size:", len(admit_dict))

del mi, admit_map; gc.collect()


print("Streaming procedures_icd in chunks and writing per-chunk daily aggregates...")
first_out = True
total_skipped_no_admit = 0
total_rows_processed = 0
reader = pd.read_csv(procedures_path,
                     usecols=['subject_id','hadm_id','seq_num','chartdate','icd_code','icd_version'],
                     parse_dates=['chartdate'],
                     chunksize=chunksize,
                     low_memory=True)

for chunk_i, chunk in enumerate(reader, start=1):
    total_rows_processed += len(chunk)
    
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')

    
    chunk = chunk[chunk['subject_id'].notna() & chunk['hadm_id'].notna()]
    if chunk.empty:
        print(f"Chunk {chunk_i}: no valid subject/hadm ids, skipping")
        continue

    
    chunk['icd_code'] = chunk['icd_code'].astype(str).str.strip()
    
    keys = list(zip(chunk['subject_id'].astype(int), chunk['hadm_id'].astype(int)))
    chunk['admit_date'] = [admit_dict.get(k, pd.NaT) for k in keys]

    
    missing_admit_mask = chunk['admit_date'].isna()
    n_missing = int(missing_admit_mask.sum())
    total_skipped_no_admit += n_missing
    if n_missing:
        
        chunk = chunk.loc[~missing_admit_mask]
    if chunk.empty:
        print(f"Chunk {chunk_i}: {n_missing} rows had no admit_date; chunk empty after drop -> continue")
        continue

    
    chunk['chart_date'] = pd.to_datetime(chunk['chartdate'], errors='coerce').dt.normalize()
    chunk['day_index'] = (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int)
    chunk.loc[chunk['day_index'] < 0, 'day_index'] = 0

    
    
    
    
    def join_unique_codes(series):
        s = set([str(x).strip() for x in series.dropna() if str(x).strip() not in ("", "nan", "None")])
        if not s:
            return ""
        return ";".join(sorted(s))

    grp = chunk.groupby(['subject_id','hadm_id','day_index'], dropna=False)
    df_agg = grp.agg(
        proc_count = ('icd_code', 'size'),
        last_proc_charttime = ('chartdate', 'max'),
        proc_codes = ('icd_code', join_unique_codes)
    ).reset_index()

    
    df_agg.to_csv(intermediate_chunks_path, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out = False

    print(f"Chunk {chunk_i}: rows_in={len(chunk):,}, groups_out={len(df_agg):,}, skipped_no_admit={n_missing}")
    del chunk, df_agg, grp
    gc.collect()

print("Streaming done. Total rows processed:", total_rows_processed)
print("Total rows skipped because no matching admission:", total_skipped_no_admit)


print("Reading intermediate chunks and final-aggregating...")
if not intermediate_chunks_path.exists():
    raise FileNotFoundError(f"Expected intermediate file {intermediate_chunks_path} not found.")

daily = pd.read_csv(intermediate_chunks_path, parse_dates=['last_proc_charttime'], low_memory=False)


def union_semicolon_lists(series):
    sset = set()
    for val in series.dropna():
        if val == "":
            continue
        parts = [p.strip() for p in str(val).split(";") if p.strip() != ""]
        sset.update(parts)
    if not sset:
        return ""
    return ";".join(sorted(sset))

final = daily.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
    proc_count = ('proc_count', 'sum'),
    last_proc_charttime = ('last_proc_charttime', 'max'),
    proc_codes = ('proc_codes', union_semicolon_lists)
)

final.to_csv(final_daily_path, index=False)
print("Final per-day procedures saved to:", final_daily_path)
del daily; gc.collect()


print("Loading d_icd_procedures to map codes -> titles (if available)...")
dproc = pd.read_csv(dprocedures_path, dtype=str, low_memory=False)
dproc['icd_code'] = dproc['icd_code'].astype(str).str.strip()
code2title = dict(zip(dproc['icd_code'], dproc['long_title'].fillna("").astype(str)))

def map_codes_to_titles(codes_str):
    if pd.isna(codes_str) or codes_str == "":
        return ""
    codes = [c for c in codes_str.split(";") if c.strip() != ""]
    titles = [code2title.get(c, "") for c in codes]
    titles = [t for t in titles if t != ""]
    return ";".join(titles)

final['proc_titles'] = final['proc_codes'].apply(map_codes_to_titles)

final.to_csv(final_daily_path, index=False)
print("Final per-day procedures (with titles) saved to:", final_daily_path)


print("Merging final daily procedure aggregates into merged_initial master table...")
merged = pd.read_csv(merged_initial_path, low_memory=False, parse_dates=['admittime','dischtime','deathtime'])

merged['subject_id'] = pd.to_numeric(merged['subject_id'], errors='coerce').astype('Int64')
merged['hadm_id'] = pd.to_numeric(merged['hadm_id'], errors='coerce').astype('Int64')
merged['day_index'] = pd.to_numeric(merged['day_index'], errors='coerce').astype('Int64')


proc_daily = pd.read_csv(final_daily_path, parse_dates=['last_proc_charttime'], low_memory=False)
proc_daily['subject_id'] = pd.to_numeric(proc_daily['subject_id'], errors='coerce').astype('Int64')
proc_daily['hadm_id'] = pd.to_numeric(proc_daily['hadm_id'], errors='coerce').astype('Int64')
proc_daily['day_index'] = pd.to_numeric(proc_daily['day_index'], errors='coerce').astype('Int64')


merged_with_proc = merged.merge(proc_daily, on=['subject_id','hadm_id','day_index'], how='left')


merged_with_proc['proc_count'] = merged_with_proc['proc_count'].fillna(0).astype('Int64')

merged_with_proc['proc_codes'] = merged_with_proc['proc_codes'].fillna("").astype(str)
merged_with_proc['proc_titles'] = merged_with_proc['proc_titles'].fillna("").astype(str)


print("Writing final merged file (chunked writes)...")
n_rows = len(merged_with_proc)
first = True
for start in range(0, n_rows, write_chunk):
    end = min(start + write_chunk, n_rows)
    merged_with_proc.iloc[start:end].to_csv(out_merged_path, mode='w' if first else 'a', index=False, header=first)
    first = False
    print(f"Wrote rows {start}..{end-1}")
print("Merged output saved to:", out_merged_path)

