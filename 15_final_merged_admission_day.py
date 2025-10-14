import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import gc
import time
import json

def safe_name(s):
    # make small safe filename-like token
    s = re.sub(r'[^0-9A-Za-z]+', '_', s)
    return s.strip('_')[:40] or 'file'

def side_by_side_merge(base_path, other_paths, out_path, chunksize=5000,
                       base_cols_keep=None, prefix_conflicts=True, verbose=True):
    base_path = Path(base_path)
    other_paths = [Path(p) for p in other_paths]
    out_path = Path(out_path)

    # read base header
    base_header = pd.read_csv(base_path, nrows=0).columns.tolist()
    if base_cols_keep is None:
        base_cols_keep = base_header  # keep them as canonical

    base_cols_keep = list(base_cols_keep)

    # Pre-open readers for all files
    base_reader = pd.read_csv(base_path, chunksize=chunksize, dtype=str, low_memory=True)
    other_readers = {}
    other_headers = {}
    for p in other_paths:
        other_readers[p] = pd.read_csv(p, chunksize=chunksize, dtype=str, low_memory=True)
        other_headers[p] = pd.read_csv(p, nrows=0).columns.tolist()

    first_out = True
    chunk_i = 0

    try:
        while True:
            chunk_i += 1
            base_chunk = next(base_reader)  # may raise StopIteration -> finished
            # keep only canonical base columns (in case file has extra)
            base_chunk = base_chunk.loc[:, [c for c in base_cols_keep if c in base_chunk.columns]]

            out_chunk = base_chunk.copy()

            # drop typical unnamed index columns if present
            drop_unnamed = [c for c in out_chunk.columns if str(c).startswith('Unnamed:') or str(c).strip()=='' ]
            if drop_unnamed:
                out_chunk = out_chunk.drop(columns=drop_unnamed, errors='ignore')

            for p, reader in other_readers.items():
                try:
                    other_chunk = next(reader)
                except StopIteration:
                    raise RuntimeError(f"File {p} ended earlier than base file at chunk {chunk_i}. Row counts differ!")

                # drop unnamed index columns
                other_chunk = other_chunk.loc[:, [c for c in other_chunk.columns if not (str(c).startswith('Unnamed:') or str(c).strip()=='')]]

                # remove base columns duplicated in this other file
                overlap = [c for c in base_cols_keep if c in other_chunk.columns]
                if overlap:
                    other_chunk = other_chunk.drop(columns=overlap, errors='ignore')

                # If any remaining columns collide with out_chunk columns, rename them
                collisions = [c for c in other_chunk.columns if c in out_chunk.columns]
                if collisions:
                    safe = safe_name(p.name)
                    new_names = {}
                    for c in collisions:
                        if prefix_conflicts:
                            new_names[c] = f"{c}__from_{safe}"
                        else:
                            # fallback: append file suffix to make unique
                            new_names[c] = f"{c}__{safe}"
                    other_chunk = other_chunk.rename(columns=new_names)

                # As a final safety, if other_chunk has more/less rows than base_chunk, check lengths
                if len(other_chunk) != len(out_chunk):
                    # If lengths mismatch within chunk, that's fatal for exact alignment
                    raise RuntimeError(
                        f"Row-count mismatch in chunk {chunk_i} for file {p.name}: "
                        f"base_chunk rows={len(out_chunk)}, other_chunk rows={len(other_chunk)}. "
                        "Ensure files are aligned and have same row order/count."
                    )

                # concatenate horizontally (preserve column order: base then others incrementally)
                out_chunk = pd.concat([out_chunk.reset_index(drop=True), other_chunk.reset_index(drop=True)], axis=1)

            # write out_chunk to CSV (append after first)
            if first_out:
                out_chunk.to_csv(out_path, index=False, mode='w', header=True)
                first_out = False
            else:
                out_chunk.to_csv(out_path, index=False, mode='a', header=False)

            if verbose:
                print(f"Chunk {chunk_i}: wrote {len(out_chunk)} rows, total cols now: {len(out_chunk.columns)}")

    except StopIteration:
        # base finished normally; ensure all other_readers are also finished
        pass

    # verify other readers exhausted
    for p, reader in other_readers.items():
        try:
            next(reader)
            raise RuntimeError(f"File {p} has MORE rows than base file — row counts differ.")
        except StopIteration:
            # good, finished
            if verbose:
                print(f"Verified {p.name} finished (same length as base).")

    if verbose:
        print("Merge complete. Output saved to:", out_path)


if __name__ == "__main__":
    # ----- CONFIGURE THESE -----
    base = "admissions_expanded.csv"   # canonical base
    others = [
        "merged_with_icu.csv",
        "merged_with_vanco.csv",
        "merged_with_chartevents_filled_renamed.csv", # تا این چک شد 
        "merged_with_diagnoses.csv",
        "merged_with_procedures.csv",
        "merged_with_drg.csv",
        "merged_with_transfers_services.csv",
        "merged_with_medications.csv",
        "merged_with_inputs_procs.csv",
        "merged_with_microbiologyevents.csv",
        "merged_with_datetimeevents_filled_renamed.csv",
        "merged_with_outputevents_filled_renamed.csv",
        "merged_with_poe.csv",
        "merged_with_other_labs_raw.csv",
        
        # add other merged_*.csv files here in the order you want columns appended
    ]
    out = "final_merged_admission_day.csv"
    CHUNK = 5000   # tune down/up depending on memory
    PREF_CONFLICT = True
    # --------------------------

    # you can also call side_by_side_merge() from another script and pass different args
    side_by_side_merge(base, others, out, chunksize=CHUNK, prefix_conflicts=PREF_CONFLICT, verbose=True)
