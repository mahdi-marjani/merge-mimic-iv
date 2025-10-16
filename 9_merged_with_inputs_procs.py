import pandas as pd
import numpy as np
from pathlib import Path
import gc
import time


BASE = Path(".")
MERGED_INITIAL_PATH = BASE / "admissions_expanded.csv"   
ING_PATH  = BASE / "ingredientevents.csv"
INP_PATH  = BASE / "inputevents.csv"
PROC_PATH = BASE / "procedureevents.csv"

AGG_ING_PATH  = BASE / "agg_ingredient_daily.csv"
AGG_INP_PATH  = BASE / "agg_input_daily.csv"
AGG_PROC_PATH = BASE / "agg_procedure_daily.csv"

SRC_CHUNKSIZE = 200_000   
WRITE_CHUNK    = 20_000   
MATCH_CHUNK    = 200_000   



def nowstr(): return time.strftime("%Y-%m-%d %H:%M:%S")
def parse_numeric(x):
    if pd.isna(x): return np.nan
    try:
        if isinstance(x,(int,float,np.number)): return float(x)
        s = str(x).strip().replace(',','')
        if s in ("","NaN","nan","None","none","___"): return np.nan
        return float(s)
    except:
        return np.nan

def read_header(path: Path):
    if not path.exists(): return []
    return pd.read_csv(path, nrows=0).columns.tolist()

def build_admit_map(merged_initial_path):
    print(f"[{nowstr()}] Build admit_map (light read)...")
    mi = pd.read_csv(merged_initial_path, usecols=['subject_id','hadm_id','admittime'], parse_dates=['admittime'], low_memory=False)
    mi['subject_id'] = pd.to_numeric(mi['subject_id'], errors='coerce').astype('Int64')
    mi['hadm_id'] = pd.to_numeric(mi['hadm_id'], errors='coerce').astype('Int64')
    adm = mi.groupby(['subject_id','hadm_id'], dropna=False)['admittime'].first().reset_index().rename(columns={'admittime':'admit_time'})
    adm['admit_date'] = pd.to_datetime(adm['admit_time']).dt.normalize()
    d = dict(zip(zip(adm['subject_id'].astype(int), adm['hadm_id'].astype(int)), adm['admit_date']))
    print(f"[{nowstr()}] admit_map keys: {len(d)}")
    del mi, adm; gc.collect()
    return d

def key_str(s,h,d): return f"{int(s)}|{int(h)}|{int(d)}"

def key_series_from_df(df):
    s = pd.to_numeric(df['subject_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    h = pd.to_numeric(df['hadm_id'], errors='coerce').fillna(-1).astype(int).astype(str)
    d = pd.to_numeric(df['day_index'], errors='coerce').fillna(-1).astype(int).astype(str)
    return s + '|' + h + '|' + d

def _choose_agg_for_col(col_name):
    """
    Heuristic for collapsing duplicates in agg files.
    - counts/totals -> sum
    - '*_max' or 'max' in name -> max
    - numeric-like names containing 'amount' or 'rate' -> sum (for amounts) or max (for rate) depending on keywords
    - otherwise -> take last non-null
    """
    name = col_name.lower()
    if name in ('subject_id','hadm_id','day_index'):
        return 'first'
    if 'count' in name or 'total' in name or name.endswith('_sum'):
        return 'sum'
    if 'max' in name or name.endswith('_max'):
        return 'max'
    
    if 'amount' in name or 'total' in name:
        return 'sum'
    
    if 'rate' in name or name.endswith('_num') or 'val' in name or 'value' in name:
        return 'max'
    
    return lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] > 0 else pd.NA

def collect_matching_rows(agg_path, keys_set, usecols=None, parse_dates=None, chunksize=MATCH_CHUNK):
    """
    Read agg_path in chunks and collect rows matching keys_set.
    Then collapse duplicate keys by grouping and applying heuristics for aggregation.
    Returns a DataFrame with unique keys.
    """
    if not agg_path.exists() or not keys_set:
        return pd.DataFrame(columns=(usecols or []))

    
    keys_s = set(key_str(s,h,d) for (s,h,d) in keys_set)

    matches = []
    cols_seen = None
    for chunk in pd.read_csv(agg_path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize, low_memory=True):
        
        if not {'subject_id','hadm_id','day_index'}.issubset(set(chunk.columns)):
            continue
        
        chunk['subject_id'] = pd.to_numeric(chunk['subject_id'], errors='coerce').astype('Int64')
        chunk['hadm_id'] = pd.to_numeric(chunk['hadm_id'], errors='coerce').astype('Int64')
        chunk['day_index'] = pd.to_numeric(chunk['day_index'], errors='coerce').astype('Int64')

        ks = key_series_from_df(chunk)
        mask = ks.isin(keys_s)
        sel = chunk.loc[mask]
        if not sel.empty:
            matches.append(sel)
            if cols_seen is None:
                cols_seen = sel.columns.tolist()
        del chunk, ks, mask; gc.collect()

    if not matches:
        return pd.DataFrame(columns=(usecols or []))

    df = pd.concat(matches, ignore_index=True)

    
    df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce').astype('Int64')
    df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce').astype('Int64')
    df['day_index'] = pd.to_numeric(df['day_index'], errors='coerce').astype('Int64')

    
    agg_dict = {}
    for col in df.columns:
        agg_dict[col] = _choose_agg_for_col(col)

    
    grouped = df.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(agg_dict)

    
    grouped['subject_id'] = pd.to_numeric(grouped['subject_id'], errors='coerce').astype('Int64')
    grouped['hadm_id'] = pd.to_numeric(grouped['hadm_id'], errors='coerce').astype('Int64')
    grouped['day_index'] = pd.to_numeric(grouped['day_index'], errors='coerce').astype('Int64')

    return grouped


admit_dict = build_admit_map(MERGED_INITIAL_PATH)


print(f"[{nowstr()}] Phase A: aggregate ingredientevents -> {AGG_ING_PATH}")
if AGG_ING_PATH.exists(): AGG_ING_PATH.unlink()
hdr = read_header(ING_PATH)
use_want = ['subject_id','hadm_id','starttime','endtime','storetime','itemid','amount','amountuom','rate','rateuom','orderid','statusdescription','originalamount','originalrate']
usecols = [c for c in use_want if c in hdr]
parse_dates = [c for c in ['starttime','endtime','storetime'] if c in hdr]
first_out=True; total=0
for i, chunk in enumerate(pd.read_csv(ING_PATH, usecols=usecols, parse_dates=parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0=time.time(); print(f"[{nowstr()}] ing chunk {i} rows={len(chunk):,}")
    chunk['subject_id']=pd.to_numeric(chunk['subject_id'],errors='coerce').astype('Int64')
    chunk['hadm_id']=pd.to_numeric(chunk['hadm_id'],errors='coerce').astype('Int64')
    keys=list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date']=[admit_dict.get((int(s),int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before=len(chunk); chunk=chunk[chunk['admit_date'].notna()].copy(); dropped=before-len(chunk)
    if dropped: print(f"  -> dropped {dropped:,} ing rows w/o admit")
    if chunk.empty: del chunk; gc.collect(); continue
    
    chunk['event_time']=pd.to_datetime(chunk.get('starttime', pd.NaT),errors='coerce').fillna(pd.to_datetime(chunk.get('storetime', pd.NaT),errors='coerce')).fillna(pd.to_datetime(chunk.get('endtime', pd.NaT),errors='coerce'))
    chunk['chart_date']=chunk['event_time'].dt.normalize()
    chunk['day_index']= (chunk['chart_date'] - chunk['admit_date']).dt.days.fillna(0).astype(int); chunk.loc[chunk['day_index']<0,'day_index']=0
    
    chunk['amount_num']=chunk['amount'].apply(parse_numeric) if 'amount' in chunk.columns else np.nan
    chunk['rate_num']=chunk['rate'].apply(parse_numeric) if 'rate' in chunk.columns else np.nan
    
    chunk['item_text']=chunk['itemid'].astype(str).str.strip()
    
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        ing_events_count = ('item_text','count'),
        ing_total_amount = ('amount_num','sum'),
        ing_max_rate = ('rate_num','max'),
        ing_last_itemid = ('item_text', lambda s: s.dropna().astype(str).iloc[-1] if len(s.dropna())>0 else pd.NA),
        ing_last_status = ('statusdescription', lambda s: s.dropna().iloc[-1] if 'statusdescription' in chunk.columns and len(s.dropna())>0 else pd.NA)
    )
    agg['subject_id']=agg['subject_id'].astype('Int64'); agg['hadm_id']=agg['hadm_id'].astype('Int64'); agg['day_index']=agg['day_index'].astype('Int64')
    agg.to_csv(AGG_ING_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out=False; total+=len(agg)
    print(f"[{nowstr()}] ing chunk {i} -> wrote {len(agg):,} agg rows (total {total:,}) took {time.time()-t0:.1f}s")
    del chunk, agg; gc.collect()
print(f"[{nowstr()}] ingredient aggregation done. total agg rows (raw appended): {total}")


print(f"[{nowstr()}] Phase B: aggregate inputevents -> {AGG_INP_PATH}")
if AGG_INP_PATH.exists(): AGG_INP_PATH.unlink()
hdr = read_header(INP_PATH)
use_want = ['subject_id','hadm_id','starttime','endtime','storetime','itemid','amount','amountuom','rate','rateuom','orderid','ordercategoryname','ordercomponenttypedescription','totalamount','totalamountuom','isopenbag','statusdescription']
usecols = [c for c in use_want if c in hdr]
parse_dates = [c for c in ['starttime','endtime','storetime'] if c in hdr]
first_out=True; total=0
for i, chunk in enumerate(pd.read_csv(INP_PATH, usecols=usecols, parse_dates=parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0=time.time(); print(f"[{nowstr()}] inp chunk {i} rows={len(chunk):,}")
    chunk['subject_id']=pd.to_numeric(chunk['subject_id'],errors='coerce').astype('Int64')
    chunk['hadm_id']=pd.to_numeric(chunk['hadm_id'],errors='coerce').astype('Int64')
    keys=list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date']=[admit_dict.get((int(s),int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before=len(chunk); chunk=chunk[chunk['admit_date'].notna()].copy(); dropped=before-len(chunk)
    if dropped: print(f"  -> dropped {dropped:,} input rows w/o admit")
    if chunk.empty: del chunk; gc.collect(); continue
    chunk['event_time']=pd.to_datetime(chunk.get('starttime', pd.NaT),errors='coerce').fillna(pd.to_datetime(chunk.get('storetime', pd.NaT),errors='coerce')).fillna(pd.to_datetime(chunk.get('endtime', pd.NaT),errors='coerce'))
    chunk['chart_date']=chunk['event_time'].dt.normalize()
    chunk['day_index']=(chunk['chart_date']-chunk['admit_date']).dt.days.fillna(0).astype(int); chunk.loc[chunk['day_index']<0,'day_index']=0
    chunk['amount_num']=chunk['amount'].apply(parse_numeric) if 'amount' in chunk.columns else np.nan
    chunk['rate_num']=chunk['rate'].apply(parse_numeric) if 'rate' in chunk.columns else np.nan
    chunk['totalamount_num']=chunk['totalamount'].apply(parse_numeric) if 'totalamount' in chunk.columns else np.nan
    chunk['item_text']=chunk['itemid'].astype(str)
    chunk['ordercat_text']=chunk['ordercategoryname'].astype(str) if 'ordercategoryname' in chunk.columns else pd.NA
    
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        input_events_count = ('item_text','count'),
        input_total_amount = ('amount_num','sum'),
        input_totalamount_field = ('totalamount_num','sum'),
        input_max_rate = ('rate_num','max'),
        input_last_ordercat = ('ordercat_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        input_last_status = ('statusdescription', lambda s: s.dropna().iloc[-1] if 'statusdescription' in chunk.columns and len(s.dropna())>0 else pd.NA)
    )
    agg['subject_id']=agg['subject_id'].astype('Int64'); agg['hadm_id']=agg['hadm_id'].astype('Int64'); agg['day_index']=agg['day_index'].astype('Int64')
    agg.to_csv(AGG_INP_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out=False; total+=len(agg)
    print(f"[{nowstr()}] inp chunk {i} -> wrote {len(agg):,} agg rows (total {total:,}) took {time.time()-t0:.1f}s")
    del chunk, agg; gc.collect()
print(f"[{nowstr()}] inputevents aggregation done. total agg rows (raw appended): {total}")


print(f"[{nowstr()}] Phase C: aggregate procedureevents -> {AGG_PROC_PATH}")
if AGG_PROC_PATH.exists(): AGG_PROC_PATH.unlink()
hdr = read_header(PROC_PATH)
use_want = ['subject_id','hadm_id','starttime','endtime','storetime','itemid','value','valueuom','location','locationcategory','ordercategoryname','statusdescription']
usecols = [c for c in use_want if c in hdr]
parse_dates = [c for c in ['starttime','endtime','storetime'] if c in hdr]
first_out=True; total=0
for i, chunk in enumerate(pd.read_csv(PROC_PATH, usecols=usecols, parse_dates=parse_dates, chunksize=SRC_CHUNKSIZE, low_memory=True), start=1):
    t0=time.time(); print(f"[{nowstr()}] proc chunk {i} rows={len(chunk):,}")
    chunk['subject_id']=pd.to_numeric(chunk['subject_id'],errors='coerce').astype('Int64')
    chunk['hadm_id']=pd.to_numeric(chunk['hadm_id'],errors='coerce').astype('Int64')
    keys=list(zip(chunk['subject_id'].astype('Int64').astype(object), chunk['hadm_id'].astype('Int64').astype(object)))
    chunk['admit_date']=[admit_dict.get((int(s),int(h)), pd.NaT) if not (pd.isna(s) or pd.isna(h)) else pd.NaT for s,h in keys]
    before=len(chunk); chunk=chunk[chunk['admit_date'].notna()].copy(); dropped=before-len(chunk)
    if dropped: print(f"  -> dropped {dropped:,} proc rows w/o admit")
    if chunk.empty: del chunk; gc.collect(); continue
    chunk['event_time']=pd.to_datetime(chunk.get('starttime', pd.NaT),errors='coerce').fillna(pd.to_datetime(chunk.get('storetime', pd.NaT),errors='coerce')).fillna(pd.to_datetime(chunk.get('endtime', pd.NaT),errors='coerce'))
    chunk['chart_date']=chunk['event_time'].dt.normalize()
    chunk['day_index']=(chunk['chart_date']-chunk['admit_date']).dt.days.fillna(0).astype(int); chunk.loc[chunk['day_index']<0,'day_index']=0
    chunk['val_num']=chunk['value'].apply(parse_numeric) if 'value' in chunk.columns else np.nan
    chunk['item_text']=chunk['itemid'].astype(str)
    chunk['loc_text']=chunk['location'].astype(str) if 'location' in chunk.columns else pd.NA
    agg = chunk.groupby(['subject_id','hadm_id','day_index'], as_index=False).agg(
        proc_events_count = ('item_text','count'),
        proc_max_value = ('val_num','max'),
        proc_last_valueuom = ('valueuom', lambda s: s.dropna().iloc[-1] if 'valueuom' in chunk.columns and len(s.dropna())>0 else pd.NA),
        proc_last_location = ('loc_text', lambda s: s.dropna().iloc[-1] if len(s.dropna())>0 else pd.NA),
        proc_last_category = ('ordercategoryname', lambda s: s.dropna().iloc[-1] if 'ordercategoryname' in chunk.columns and len(s.dropna())>0 else pd.NA)
    )
    agg['subject_id']=agg['subject_id'].astype('Int64'); agg['hadm_id']=agg['hadm_id'].astype('Int64'); agg['day_index']=agg['day_index'].astype('Int64')
    agg.to_csv(AGG_PROC_PATH, mode='w' if first_out else 'a', index=False, header=first_out)
    first_out=False; total+=len(agg)
    print(f"[{nowstr()}] proc chunk {i} -> wrote {len(agg):,} agg rows (total {total:,}) took {time.time()-t0:.1f}s")
    del chunk, agg; gc.collect()
print(f"[{nowstr()}] procedure aggregation done. total agg rows (raw appended): {total}")


OUT = BASE / "merged_with_inputs_procs.csv"
if OUT.exists(): OUT.unlink()
print(f"[{nowstr()}] Phase D: merge aggregated files into {OUT} in chunks (write_chunk={WRITE_CHUNK})")
first_write=True; mchunk_no=0
for mchunk in pd.read_csv(MERGED_INITIAL_PATH, chunksize=WRITE_CHUNK, parse_dates=['admittime','dischtime','deathtime','edregtime','edouttime'], low_memory=False):
    mchunk_no+=1; t0=time.time()
    print(f"[{nowstr()}] merged chunk {mchunk_no} rows={len(mchunk):,}")
    mchunk['subject_id']=pd.to_numeric(mchunk['subject_id'],errors='coerce').astype('Int64')
    mchunk['hadm_id']=pd.to_numeric(mchunk['hadm_id'],errors='coerce').astype('Int64')
    mchunk['day_index']=pd.to_numeric(mchunk['day_index'],errors='coerce').astype('Int64')

    
    keys = set((int(r['subject_id']), int(r['hadm_id']), int(r['day_index'])) for _,r in mchunk[['subject_id','hadm_id','day_index']].iterrows())

    
    ing_match = collect_matching_rows(AGG_ING_PATH, keys, usecols=None) if AGG_ING_PATH.exists() else pd.DataFrame()
    inp_match = collect_matching_rows(AGG_INP_PATH, keys, usecols=None) if AGG_INP_PATH.exists() else pd.DataFrame()
    proc_match = collect_matching_rows(AGG_PROC_PATH, keys, usecols=None) if AGG_PROC_PATH.exists() else pd.DataFrame()

    
    if not ing_match.empty:
        mchunk = mchunk.merge(ing_match, on=['subject_id','hadm_id','day_index'], how='left')
    if not inp_match.empty:
        mchunk = mchunk.merge(inp_match, on=['subject_id','hadm_id','day_index'], how='left')
    if not proc_match.empty:
        mchunk = mchunk.merge(proc_match, on=['subject_id','hadm_id','day_index'], how='left')

    mchunk.to_csv(OUT, mode='w' if first_write else 'a', index=False, header=first_write)
    first_write=False
    print(f"[{nowstr()}] Written merged chunk {mchunk_no} (took {time.time()-t0:.1f}s)")
    del mchunk, ing_match, inp_match, proc_match; gc.collect()

print(f"[{nowstr()}] All done. Output: {OUT}")
