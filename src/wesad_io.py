from pathlib import Path
import csv, hashlib, pickle, re
import numpy as np
import pandas as pd

RATES={'chest':{'ECG':700,'EDA':700,'EMG':700,'Resp':700,'Temp':700,'ACC':700},
       'wrist':{'BVP':64,'EDA':4,'TEMP':4,'ACC':32}}
LABELS={1:'baseline',2:'stress',3:'amusement',4:'meditation'}
META=['window_id','subject_id','episode_id','label','start','end']

def discover(root):
    files=sorted(Path(root).rglob('S*.pkl'))
    found={}
    for p in files:
        if re.fullmatch(r'S\d+',p.stem):
            sid=int(p.stem[1:])
            if sid in found: raise ValueError(f'Duplicate subject {sid}')
            found[sid]=p
    if not found: raise FileNotFoundError('No SX.pkl files found; download official WESAD first')
    return found

def load_subject(path, trusted=False):
    if not trusted: raise ValueError('Pickle executes code: enable trusted_pickle only for verified official files')
    with Path(path).open('rb') as f: obj=pickle.load(f,encoding='latin1')
    sid=int(Path(path).stem[1:]); labels=np.asarray(obj['label']).reshape(-1)
    if not np.all(np.isin(labels,[0,1,2,3,4,5,6,7])):
        raise ValueError('Unknown label codes; inspect README')
    signals={}
    for site,channels in RATES.items():
        signals[site]={}
        for name in channels:
            channels_in=obj['signal'].get(site,{})
            aliases={'Resp':['Resp','RESP','RESPIRATION'],'Temp':['Temp','TEMP']}
            actual=next((key for key in aliases.get(name,[name]) if key in channels_in),None)
            if actual is not None:
                a=np.asarray(channels_in[actual],dtype=float)
                signals[site][name]=a[:,None] if a.ndim==1 else a
    return {'subject_id':sid,'labels':labels.astype(int),'signal':signals,
            'source':str(path),'alignment_evidence':'WESAD supplied synchronized pickle; duration check only'}

def episodes(labels):
    labels=np.asarray(labels)
    edges=np.r_[0,np.flatnonzero(np.diff(labels)!=0)+1,len(labels)]
    return [(i,int(labels[a]),int(a),int(b)) for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]

def read_questionnaire(path):
    # Preserve row names and original values; do not guess undocumented timing units.
    raw=Path(path).read_text(encoding='utf-8-sig',errors='replace')
    sep=';' if raw.count(';')>raw.count(',') else ','
    rows=list(csv.reader(raw.splitlines(),delimiter=sep))
    def select(token): return [r for r in rows if r and token in r[0].strip().upper()]
    return {'path':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'rows':rows,'order_rows':select('ORDER'),'start_rows':select('START'),
            'end_rows':select('END'),
            'self_report_rows':[r for r in rows if r and any(x in r[0].upper() for x in ['PANAS','STAI','SAM','DIM','SSSQ'])],
            'timing_status':'raw rows retained; units and correspondence require original README review'}

def audit(records,c):
    rows=[]
    expected=set(c['dataset']['expected_subjects'])
    for r in records:
        labels=r['labels'];sid=r['subject_id'];p=Path(r.get('source','synthetic'))
        quest=list(p.parent.glob(f'S{sid}_quest.*')) if p.exists() else []
        readmes=list(p.parent.glob('*readme*')) if p.exists() else []
        for label in np.unique(labels):
            runs=[(a,b) for _,l,a,b in episodes(labels) if l==label]
            w=c['features']['window_seconds'];step=w*(1-c['features']['overlap'])
            rows.append({'subject_id':sid,'label':int(label),'condition':LABELS.get(int(label),'ignored'),
                         'seconds':sum(b-a for a,b in runs)/700,'episodes':len(runs),
                         'candidate_windows':sum(max(0,int(np.floor(((b-a)/700-w)/step))+1) for a,b in runs),
                         'label_samples':int((labels==label).sum()),
                         'missing_signal_values':sum(int((~np.isfinite(a)).sum()) for ch in r['signal'].values() for a in ch.values()),
                         'questionnaire_present':bool(quest),'readme_present':bool(readmes),
                         'unexpected_subject':sid not in expected})
    return pd.DataFrame(rows)
