"""Artificial waveforms for execution tests only, never physiological validation."""
import numpy as np
from .wesad_io import RATES

def make_records(n_subjects=6,seconds_per_class=120,seed=42,include_meditation=False):
    classes=[1,2,3,4] if include_meditation else [1,2,3]
    records=[];rng=np.random.default_rng(seed)
    for sid in range(2,2+n_subjects):
        labels=np.repeat(classes,int(seconds_per_class*700));signals={}
        for site,channels in RATES.items():
            signals[site]={}
            for name,fs in channels.items():
                parts=[]
                for label in classes:
                    t=np.arange(int(seconds_per_class*fs))/fs
                    hr={1:65,2:90,3:75,4:60}[label]+sid*.3
                    if name=='ECG':
                        period=60/hr;beats=np.arange(.3,seconds_per_class,period)
                        beats+=rng.normal(0,.012,len(beats))
                        a=rng.normal(0,.008,len(t))
                        for beat in beats:
                            lo=max(0,int((beat-.07)*fs));hi=min(len(t),int((beat+.07)*fs))
                            a[lo:hi]+=np.exp(-((t[lo:hi]-beat)/.012)**2)
                    elif name=='BVP':a=np.sin(2*np.pi*hr/60*t)+.05*rng.normal(size=len(t))
                    elif name=='EDA':a=1+label*.2+.06*np.sin(2*np.pi*.12*t)+.005*rng.normal(size=len(t))
                    elif name.upper()=='TEMP':a=32+sid*.02+.03*np.sin(t/20)+.001*rng.normal(size=len(t))
                    elif name=='Resp':a=np.sin(2*np.pi*(.2+label*.02)*t)+.02*rng.normal(size=len(t))
                    elif name=='ACC':a=np.column_stack([.02*rng.normal(size=len(t)),.02*rng.normal(size=len(t)),1+.02*rng.normal(size=len(t))])
                    else:a=.02*rng.normal(size=len(t))
                    parts.append(a[:,None] if a.ndim==1 else a)
                signals[site][name]=np.concatenate(parts)
        records.append({'subject_id':sid,'labels':labels,'signal':signals,
                        'source':'synthetic','alignment_evidence':'generated shared clock; not real WESAD'})
    return records

def make_questionnaires(root,n_subjects=6,seed=42):
    """Artificial questionnaire fixtures, stored separately from real questionnaires."""
    from pathlib import Path
    root=Path(root);root.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed)
    for sid in range(2,2+n_subjects):
        lines=[f'# Subj;S{sid}','# ORDER;Base;TSST;Fun;Medi 1;Medi 2',
               '# START;0.00;2.00;4.00;6.00;8.00','# END;2.00;4.00;6.00;8.00;10.00']
        for condition in range(5):
            length=26 if condition==1 else 24
            values=rng.integers(1,6,size=length)
            lines.append('# PANAS;'+';'.join(map(str,values)))
        for _ in range(5):lines.append('# STAI;'+';'.join(map(str,rng.integers(1,5,size=6))))
        for _ in range(5):lines.append('# DIM;'+';'.join(map(str,rng.integers(1,10,size=2))))
        lines.append('# SSSQ;'+';'.join(map(str,rng.integers(1,6,size=6))))
        (root/f'S{sid}_quest.csv').write_text('\n'.join(lines)+'\n')
    return root
