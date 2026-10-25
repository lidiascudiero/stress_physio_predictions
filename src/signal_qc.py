import numpy as np
import pandas as pd
from .wesad_io import RATES

def alignment(record,tolerance=1.0):
    duration=len(record['labels'])/700
    rows=[]
    for site,channels in RATES.items():
        for name,fs in channels.items():
            a=record['signal'].get(site,{}).get(name)
            actual=len(a)/fs if a is not None else np.nan
            rows.append({'subject_id':record['subject_id'],'site':site,'channel':name,
                         'sampling_hz':fs,'label_duration':duration,'signal_duration':actual,
                         'duration_ok':bool(np.isfinite(actual) and abs(actual-duration)<=tolerance),
                         'synchronization_verified':False,
                         'evidence':record.get('alignment_evidence','not documented')})
    return pd.DataFrame(rows)

def assess(x,fs,name,c):
    x=np.asarray(x,dtype=float); x=x[:,None] if x.ndim==1 else x
    finite=np.all(np.isfinite(x),axis=1)
    ratio=float(finite.mean()) if len(x) else 0.
    usable=x[finite]
    flat=not len(usable) or np.any(np.std(usable,axis=0)<c['qc']['flat_std'])
    in_range=finite.copy()
    if name.upper()=='TEMP':
        lo,hi=c['qc']['temp_range_c'];in_range &= np.all((x>=lo)&(x<=hi),axis=1)
    if name=='EDA':
        lo,hi=c['qc']['eda_range_us'];in_range &= np.all((x>=lo)&(x<=hi),axis=1)
    valid_ratio=float(in_range.mean()) if len(x) else 0.
    # ACC units must be reviewed against README; warning is not an exclusion.
    movement=bool(name=='ACC' and len(usable) and np.sqrt(np.var(usable,axis=0).sum())>c['qc']['acc_movement_warning_g'])
    return {'finite_ratio':ratio,'valid_ratio':valid_ratio,'flat':bool(flat),
            'movement_warning':movement,
            'valid':bool(valid_ratio>=c['features']['min_valid_ratio'] and not flat),
            'reason':'ok' if valid_ratio>=c['features']['min_valid_ratio'] and not flat else 'flat_or_missing_or_out_of_range'}

def fill_short_missing(x):
    x=np.asarray(x,float).copy()
    if x.ndim==1:x=x[:,None]
    for j in range(x.shape[1]):
        good=np.isfinite(x[:,j])
        if not good.any():return None
        # Only short interior gaps, never long missing intervals or edge extrapolation.
        missing=np.flatnonzero(~good)
        runs=np.split(missing,np.flatnonzero(np.diff(missing)>1)+1)
        for run in runs:
            if not len(run):continue
            if run[0]==0 or run[-1]==len(x)-1 or len(run)>2:return None
        x[:,j]=np.interp(np.arange(len(x)),np.flatnonzero(good),x[good,j])
    return x
