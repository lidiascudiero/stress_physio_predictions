"""Readers for documented original WESAD raw formats. Never assume shared clocks."""
from pathlib import Path
from decimal import Decimal
import json
import numpy as np
import pandas as pd
from .wesad_io import read_questionnaire

E4_RATES={'ACC':32,'BVP':64,'EDA':4,'TEMP':4}
CONDITIONS={'Base':1,'TSST':2,'Fun':3,'Medi 1':4,'Medi 2':4}
IGNORED_PROTOCOL={'bRead','fRead','sRead'}

def protocol_seconds(text):
    """Minutes.seconds notation: 7.08=428 s; 50.3=50m30s, never decimal minutes."""
    value=Decimal(str(text).strip())
    minutes=int(value);seconds=(value-minutes)*100
    if value<0 or seconds!=int(seconds) or seconds>=60:
        raise ValueError(f'Invalid minutes.seconds value: {text}')
    return minutes*60+int(seconds)

def schedule(path):
    q=read_questionnaire(path)
    if not all(len(q[k])==1 for k in ['order_rows','start_rows','end_rows']):
        raise ValueError('Expected one ORDER, START and END row')
    order,start,end=(q[k][0][1:] for k in ['order_rows','start_rows','end_rows'])
    rows=[]
    for condition,a,b in zip(order,start,end):
        condition=condition.strip()
        if not condition:continue
        if condition not in CONDITIONS and condition not in IGNORED_PROTOCOL:
            raise ValueError(f'Unknown protocol condition {condition}')
        begin=protocol_seconds(a);finish=protocol_seconds(b)
        if finish<=begin:raise ValueError('Nonpositive episode duration')
        rows.append({'condition':condition,'label':CONDITIONS.get(condition),
                     'start_seconds':begin,'end_seconds':finish,'duration_seconds':finish-begin,
                     'include_primary':condition in ['Base','TSST','Fun'],
                     'ignored':condition in IGNORED_PROTOCOL,'clock':'RespiBAN recording origin'})
    for i,a in enumerate(rows):
        for b in rows[i+1:]:
            if a['start_seconds']<b['end_seconds'] and b['start_seconds']<a['end_seconds']:
                raise ValueError('Overlapping questionnaire intervals')
    return pd.DataFrame(rows)

def load_e4(path,channel):
    if channel not in E4_RATES:raise ValueError('HR/IBI/tags are derived; excluded from raw feature input')
    with Path(path).open() as f:
        timestamp=np.asarray([float(v.strip()) for v in f.readline().split(',')])
        rate=np.asarray([float(v.strip()) for v in f.readline().split(',')])
    if not np.all(timestamp==timestamp[0]) or not np.all(rate==rate[0]):
        raise ValueError('Inconsistent E4 header values')
    fs=float(rate[0])
    if fs!=E4_RATES[channel]:raise ValueError('Unexpected sampling frequency')
    values=np.loadtxt(path,delimiter=',',skiprows=2,ndmin=2)
    if values.shape[1]!=(3 if channel=='ACC' else 1):raise ValueError('Unexpected channel count')
    if channel=='ACC':values=values/64.0
    return {'channel':channel,'values':values,'sampling_hz':fs,
            'start_unix_utc':float(timestamp[0]),'duration_seconds':len(values)/fs,
            'unit':{'ACC':'g','TEMP':'degC','EDA':'uS','BVP':'device units'}[channel]}

def respiban_header(path):
    with Path(path).open() as f:
        first=f.readline();line=f.readline()
    if not first.startswith('# OpenSignals'):raise ValueError('Not an OpenSignals file')
    obj=json.loads(line.lstrip('# ').strip())
    if len(obj)!=1:raise ValueError('Expected exactly one device')
    metadata=next(iter(obj.values()))
    expected=['ECG','EDA','EMG','TEMP','XYZ','XYZ','XYZ','RESPIRATION']
    if metadata['sensor']!=expected or metadata['sampling rate']!=700:
        raise ValueError('Unexpected RespiBAN sensor order or sampling rate')
    if metadata['column']!=['nSeq','DI']+[f'CH{i}' for i in range(1,9)]:
        raise ValueError('Unexpected raw columns')
    return metadata

def convert_respiban(raw):
    a=np.asarray(raw,dtype=float)
    if a.ndim!=2 or a.shape[1]!=8:raise ValueError('Eight raw channels required')
    vcc=3.;bit=2.**16
    vout=a[:,3]*vcc/(bit-1.)
    with np.errstate(divide='ignore',invalid='ignore'):
        resistance=1e4*vout/(vcc-vout);logr=np.log(resistance)
        temp=-273.15+1/(1.12764514e-3+2.34282709e-4*logr+8.77303013e-8*logr**3)
        temp[(vout<=0)|(vout>=vcc)]=np.nan
    return {'ECG':(a[:,0:1]/bit-.5)*vcc,'EDA':a[:,1:2]/bit*vcc/.12,
            'EMG':(a[:,2:3]/bit-.5)*vcc,'Temp':temp[:,None],
            'ACC':(a[:,4:7]-28000)/(38000-28000)*2-1,
            'Resp':(a[:,7:8]/bit-.5)*100}

def _accumulate(stats,name,a,c):
    a=np.asarray(a);finite=np.isfinite(a)
    lo,hi=stats.get(name,{}).get('minimum',np.inf),stats.get(name,{}).get('maximum',-np.inf)
    good=a[finite]
    entry=stats.setdefault(name,{'n_rows':0,'n_values':0,'nonfinite':0,'out_of_range':0})
    entry['n_rows']+=len(a);entry['n_values']+=a.size;entry['nonfinite']+=int((~finite).sum())
    entry['minimum']=min(lo,float(good.min())) if len(good) else None
    entry['maximum']=max(hi,float(good.max())) if len(good) else None
    limits=c['qc']['temp_range_c'] if name in ['Temp','TEMP'] else c['qc']['eda_range_us'] if name=='EDA' else None
    if limits:entry['out_of_range']+=int(((a<limits[0])|(a>limits[1])).sum())

def audit_respiban(path,c,chunksize=200000):
    meta=respiban_header(path);stats={};n=0;wraps=0;unexpected_jumps=0;previous=None
    # Stream the 250 MB text instead of materializing all 10 columns.
    for chunk in pd.read_csv(path,sep='\t',comment='#',header=None,chunksize=chunksize,dtype=np.float64):
        if chunk.shape[1]>10 and chunk.iloc[:,10:].isna().all().all():chunk=chunk.iloc[:,:10]
        if chunk.shape[1]!=10:raise ValueError('Expected ten raw columns (optional trailing delimiter allowed)')
        values=chunk.to_numpy();seq=values[:,0]
        d=np.diff(np.r_[previous,seq] if previous is not None else seq)
        wraps+=int((d<0).sum())
        # OpenSignals nSeq commonly wraps; count unusual positive jumps separately.
        unexpected_jumps+=int(((d>0)&(d!=1)).sum());previous=seq[-1]
        converted=convert_respiban(values[:,2:]);n+=len(values)
        for name,a in converted.items():_accumulate(stats,name,a,c)
    return {'sampling_hz':700,'n_samples':n,'duration_seconds':n/700,
            'raw_sensor_order':meta['sensor'],'header_date':meta['date'],'header_time':meta['time'],
            'timezone':'not documented; not assumed UTC','sequence_wraps':wraps,
            'sequence_unusual_positive_jumps':unexpected_jumps,'channels':stats,
            'conversion_reference':'wesad_readme.pdf section II.1'}

def _find(root,pattern):
    matches=list(Path(root).rglob(pattern))
    if len(matches)!=1:raise FileNotFoundError(f'Expected one {pattern}; found {len(matches)}')
    return matches[0]

def audit_raw_subject(subject_dir,c,subject='S2'):
    root=Path(subject_dir)
    protocol=schedule(_find(root,f'{subject}_quest.csv'))
    chest=audit_respiban(_find(root,f'{subject}_respiban.txt'),c)
    wrist={}
    for channel in E4_RATES:
        p=_find(root,f'{channel}.csv');e4=load_e4(p,channel);stats={}
        _accumulate(stats,channel,e4['values'],c)
        wrist[channel]={k:v for k,v in e4.items() if k!='values'}
        wrist[channel].update(stats[channel])
    notes=_find(root,f'{subject}_readme.txt').read_text(errors='replace')
    quality_notes=notes.split('### Additional notes ###')[-1].strip() if '### Additional notes ###' in notes else ''
    return {'subject':subject,'data_kind':'real_raw_unaligned','chest':chest,'wrist':wrist,
            'schedule':protocol.astype(object).where(pd.notna(protocol),None).to_dict(orient='records'),
            'quality_note':quality_notes,
            'synchronized_pickle_present':(root/f'{subject}.pkl').exists(),
            'excluded_derived_files':['HR.csv','IBI.csv','tags.csv'],
            'synchronization_status':'NOT VERIFIED: raw devices require double-tap alignment or original synchronized pickle',
            'fusion_status':'NOT EXECUTABLE until cross-device alignment is validated',
            'loso_status':'NOT EXECUTABLE with only one subject',
            'timing_reference':'Questionnaire minutes.seconds, relative to RespiBAN start (README III.1)'}
