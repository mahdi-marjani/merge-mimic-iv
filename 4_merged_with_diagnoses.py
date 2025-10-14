import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

# --- Config ---
DIAGNOSIS_CSV = Path("diagnoses_icd.csv")
DICT_CSV = Path("d_icd_diagnoses.csv")
MERGED_INITIAL_CSV = Path("admissions_expanded.csv")   # input (admission x day)
OUT_CSV = Path("merged_with_diagnoses.csv")      # output
CHUNK_DIAG = 200_000        # chunk size for diagnoses reading (diagnoses table is usually small)
CHUNK_WRITE = 20_000        # chunk size for writing merged_initial with diagnoses
TOP_K = 5                   # produce diag_1..diag_K columns (set to 0 to disable)

# --- Helpers ---
def normalize_code(code):
    """Normalize ICD code strings for matching: str, strip, uppercase, remove dots."""
    if pd.isna(code):
        return ""
    s = str(code).strip().upper()
    s = s.replace(".", "")
    return s

def try_lookup_description(code, version, dict_map):
    """Attempt to find long_title for (code, version) with fallback strategies."""
    key = (code, str(int(version)) if pd.notna(version) else str(version))
    if key in dict_map:
        return dict_map[key]
    # fallback: remove leading zeros from both sides (e.g., '0010' -> '10')
    code_nolead = code.lstrip("0")
    key2 = (code_nolead, key[1])
    if key2 in dict_map:
        return dict_map[key2]
    # fallback: if dict keys have leading zeros and code doesn't, try to left-pad to 4 (common for old formats)
    if code.isdigit():
        for pad in (3,4,5):
            kp = (code.zfill(pad), key[1])
            if kp in dict_map:
                return dict_map[kp]
    return pd.NA

# --- Step 1: load ICD dictionary into memory (small) ---
if not DICT_CSV.exists():
    raise FileNotFoundError(f"{DICT_CSV} not found. Place d_icd_diagnoses.csv next to this script.")

dict_df = pd.read_csv(DICT_CSV, dtype=str)  # icd_code, icd_version, long_title
# normalize dict codes:
dict_df['icd_code_norm'] = dict_df['icd_code'].astype(str).apply(normalize_code)
dict_df['icd_version_norm'] = dict_df['icd_version'].astype(str).str.strip()
# build mapping (code_norm, version) -> long_title
dict_map = dict(((row.icd_code_norm, row.icd_version_norm), row.long_title) for row in dict_df.itertuples(index=False))

print(f"Loaded ICD dictionary rows: {len(dict_df)}")

# --- Step 2: read diagnoses_icd in chunks and accumulate per-admission lists ---
if not DIAGNOSIS_CSV.exists():
    raise FileNotFoundError(f"{DIAGNOSIS_CSV} not found. Place diagnoses_icd.csv next to this script.")

acc = defaultdict(list)   # key -> list of (seq_num (int), icd_code_norm (str), icd_version)
rows_seen = 0
for chunk in pd.read_csv(DIAGNOSIS_CSV, chunksize=CHUNK_DIAG, dtype=str, low_memory=False):
    # ensure required columns exist
    for col in ("subject_id","hadm_id","seq_num","icd_code","icd_version"):
        if col not in chunk.columns:
            raise KeyError(f"Expected column '{col}' in diagnoses_icd.csv but missing.")
    # normalize and iterate
    chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
    chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
    chunk['seq_num'] = pd.to_numeric(chunk['seq_num'], errors='coerce').fillna(99999).astype(int)
    chunk['icd_code_norm'] = chunk['icd_code'].astype(str).apply(normalize_code)
    chunk['icd_version_norm'] = chunk['icd_version'].astype(str).str.strip()

    for r in chunk.itertuples(index=False):
        # skip if missing hadm or subject
        if pd.isna(r.subject_id) or pd.isna(r.hadm_id):
            continue
        key = (int(r.subject_id), int(r.hadm_id))
        acc[key].append((int(r.seq_num), r.icd_code_norm, r.icd_version_norm))
        rows_seen += 1
    print(f"Processed diagnoses rows so far: {rows_seen}", end='\r')

print(f"\nTotal diagnosis rows processed: {rows_seen}; unique admissions with diagnoses: {len(acc)}")

# --- Step 3: build per-admission aggregate DataFrame ---
agg_rows = []
for (subj, hadm), entries in acc.items():
    # sort by seq_num ascending
    entries_sorted = sorted(entries, key=lambda x: (x[0] if x[0] is not None else 99999))
    codes = [e[1] for e in entries_sorted if e[1] != ""]
    versions = [e[2] for e in entries_sorted]
    # lookup descriptions (preserve order)
    descs = [ try_lookup_description(c, v, dict_map) if c != "" else pd.NA for c,v in zip(codes, versions) ]
    n = len(codes)
    primary_code = codes[0] if n >= 1 else pd.NA
    primary_desc = descs[0] if n >= 1 else pd.NA
    # top-K split
    top_codes = {}
    top_descs = {}
    for k in range(1, TOP_K+1):
        if n >= k:
            top_codes[f"diag_{k}_code"] = codes[k-1]
            top_descs[f"diag_{k}_desc"] = descs[k-1]
        else:
            top_codes[f"diag_{k}_code"] = pd.NA
            top_descs[f"diag_{k}_desc"] = pd.NA

    agg_rows.append({
        "subject_id": int(subj),
        "hadm_id": int(hadm),
        "diag_n": int(n),
        "diag_codes": ";".join(codes) if codes else pd.NA,
        "diag_descs": ";".join([str(d) for d in descs]) if descs else pd.NA,
        "primary_diag_code": primary_code,
        "primary_diag_desc": primary_desc,
        **top_codes,
        **top_descs
    })

diag_df = pd.DataFrame(agg_rows)
# ensure dtypes
if not diag_df.empty:
    diag_df['subject_id'] = diag_df['subject_id'].astype('Int64')
    diag_df['hadm_id'] = diag_df['hadm_id'].astype('Int64')
    diag_df['diag_n'] = diag_df['diag_n'].astype('Int64')

print("Built diag_df with rows:", len(diag_df))

# --- Step 4: merge diag_df into merged_initial.csv in chunks (so we don't load merged_initial fully) ---
if not MERGED_INITIAL_CSV.exists():
    raise FileNotFoundError(f"{MERGED_INITIAL_CSV} not found. Place merged_initial.csv next to this script.")

first_write = True
written = 0
for chunk in pd.read_csv(MERGED_INITIAL_CSV, chunksize=CHUNK_WRITE, parse_dates=['admittime'], low_memory=False):
    # ensure keys have correct dtype
    if 'subject_id' in chunk.columns:
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
    if 'hadm_id' in chunk.columns:
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')

    merged_chunk = chunk.merge(diag_df, on=['subject_id','hadm_id'], how='left')
    # if diag_df is empty, the merge will just add nothing; that's okay.

    merged_chunk.to_csv(OUT_CSV, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write = False
    written += len(merged_chunk)
    print(f"Wrote merged rows: {written}", end='\r')
    del chunk, merged_chunk
    gc.collect()

print(f"\nDone. Output saved to: {OUT_CSV} (rows written: {written})")

