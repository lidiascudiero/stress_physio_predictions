import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import signal
from .wesad_io import RATES, META, episodes
from .signal_qc import assess, fill_short_missing, alignment

def _filtered(x,fs,lo,hi):
    sos=signal.butter(3,[lo,hi],btype='bandpass',fs=fs,output='sos')
    return signal.sosfiltfilt(sos,x)

def beat_features(x,fs,c,ppg=False):
    x=np.asarray(x).reshape(-1)
    y=_filtered(x,fs,0.5 if ppg else 5,5 if ppg else 25)
    # ECG polarity chosen from signal shape only, never from class labels.
    if not ppg and abs(np.quantile(y,.01))>abs(np.quantile(y,.99)):y=-y
    peaks,_=signal.find_peaks(y,distance=max(1,int(fs*.3)),prominence=max(np.std(y)*.5,1e-8))
    rr=np.diff(peaks)/fs
    plausible=(rr>=.3)&(rr<=2.)
    good=bool(len(peaks)>=c['features']['min_valid_beats'] and len(rr)>=c['features']['min_beat_intervals']
              and plausible.mean()>=c['features']['min_plausible_interval_ratio'])
    result={'beat_quality':float(plausible.mean()) if len(rr) else 0.,'beat_count':float(len(peaks))}
    if not good:return result,False
    valid=rr[plausible]*1000
    # Consecutive differences only when both original intervals were valid.
    differences=np.diff(rr)[plausible[:-1]&plausible[1:]]*1000
    prefix='prv_' if ppg else 'hrv_'
    result.update({('mean_pulse_rate' if ppg else 'mean_hr'):60/np.mean(valid/1000),
                   prefix+'sdnn':float(np.std(valid,ddof=1)),
                   prefix+'rmssd':float(np.sqrt(np.mean(differences**2))) if len(differences) else np.nan,
                   prefix+'pnn50':float(np.mean(abs(differences)>50)) if len(differences) else np.nan})
    return result,True

def channel_features(x,fs,name,c):
    x=np.asarray(x);a=x[:,0];duration=len(a)/fs
    if name in ['ECG','BVP']:return beat_features(a,fs,c,ppg=name=='BVP')
    if name=='ACC':
        mag=np.linalg.norm(x,axis=1)
        return {'activity_level':float(np.std(mag)), 'movement_variance':float(np.var(x,axis=0).sum()),
                **{f'posture_axis_{i}':float(v) for i,v in enumerate(x.mean(axis=0))}},True
    out={'mean':float(a.mean()),'std':float(a.std())}
    if name.upper()=='TEMP':out['slope_per_second']=float(np.polyfit(np.arange(len(a))/fs,a,1)[0])
    if name=='EDA':
        # Explicit low-pass tonic proxy, not a validated physiological decomposition.
        tonic=signal.sosfiltfilt(signal.butter(2,.05,fs=fs,output='sos'),a)
        phasic=a-tonic
        peaks,props=signal.find_peaks(phasic,prominence=.05,distance=max(1,int(fs)))
        left=props.get('left_bases',np.array([],int))
        out.update(tonic_proxy=float(tonic.mean()),scr_proxy_rate_per_minute=len(peaks)/duration*60,
                   rise_time_proxy=float(np.mean((peaks-left)/fs)) if len(peaks) else np.nan)
    if name=='Resp':
        y=_filtered(a,fs,.05,1.)
        peaks,_=signal.find_peaks(y,distance=int(fs),prominence=max(np.std(y)*.3,1e-8))
        out.update(respiratory_rate=len(peaks)/duration*60,amplitude_variability=float(np.std(y)))
    if name=='EMG':out.update(rms=float(np.sqrt(np.mean(a*a))))
    return out,True

def extract(records,c,classes=(1,2,3)):
    tables={'chest':[],'wrist':[]};qc=[]
    w=c['features']['window_seconds'];step=w*(1-c['features']['overlap'])
    for rec in records:
        checks=alignment(rec,c['qc']['max_duration_error_seconds'])
        for ep,label,a,b in episodes(rec['labels']):
            if label not in classes:continue
            start=a/700
            while start+w<=b/700+1e-9:
                end=start+w;sid=rec['subject_id']
                meta=dict(window_id=f'S{sid}_E{ep}_{round(start*700)}',subject_id=sid,
                          episode_id=f'S{sid}_E{ep}',label=label,start=start,end=end)
                for site in tables:
                    row=meta.copy();valid=True
                    for name,fs in RATES[site].items():
                        if name=='ACC' and not c['features']['include_acc']:continue
                        x=rec['signal'].get(site,{}).get(name)
                        dur_ok=bool(checks.loc[(checks.site==site)&(checks.channel==name),'duration_ok'].iloc[0])
                        z=x[int(round(start*fs)):int(round(end*fs))] if x is not None else np.empty((0,1))
                        q=assess(z,fs,name,c)
                        sanitized=z.copy()
                        if name.upper()=='TEMP' and len(z):
                            lo,hi=c['qc']['temp_range_c']
                            sanitized[(sanitized<lo)|(sanitized>hi)]=np.nan
                        if name=='EDA' and len(z):
                            lo,hi=c['qc']['eda_range_us']
                            sanitized[(sanitized<lo)|(sanitized>hi)]=np.nan
                        clean=fill_short_missing(sanitized) if q['valid'] and len(z) else None
                        f={};beat_ok=True
                        if clean is not None and dur_ok:
                            f,beat_ok=channel_features(clean,fs,name,c)
                        ok=q['valid'] and clean is not None and dur_ok and beat_ok
                        valid &= ok
                        qc.append({**meta,'site':site,'channel':name,**q,'valid':ok,
                                   'duration_ok':dur_ok,'beat_quality_ok':beat_ok,
                                   'reason':q['reason'] if q['reason']!='ok' else ('ok' if ok else 'duration_gap_or_beats')})
                        row.update({f'{site}__{name}__{k}':v for k,v in f.items()})
                    row['valid']=valid;tables[site].append(row)
                start+=step
    return {k:pd.DataFrame(v) for k,v in tables.items()},pd.DataFrame(qc)

def save_tables(tables,c):
    from .fusion import paired
    root=Path(c['output']['root'])/'features';root.mkdir(parents=True,exist_ok=True)
    data={**tables,'fused':paired(tables['chest'],tables['wrist'])}
    fmt=c['output']['feature_format'];names={}
    for site,frame in data.items():
        if frame.empty:raise ValueError(f'No {site} windows')
        if fmt=='parquet':frame.to_parquet(root/f'{site}_features.parquet',index=False)
        elif fmt=='csv':frame.to_csv(root/f'{site}_features.csv',index=False)
        else:raise ValueError('feature_format must be parquet or csv')
        names[site]=[x for x in frame if '__' in x]
    (root/'feature_names.json').write_text(json.dumps(names,indent=2))
    return data
