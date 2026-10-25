import json
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,f1_score,recall_score,
                             confusion_matrix,roc_auc_score,average_precision_score,precision_recall_curve,auc)
from .splits import loso
from .models import tune_fit
from .config import fingerprint

PERMUTATION_STATUS='Permutation test not applicable without a validated exchangeability scheme.'

def metrics(y,p,classes):
    classes=list(classes);y=np.asarray(y);pred=np.asarray(classes)[np.argmax(p,axis=1)]
    out={'accuracy':float(accuracy_score(y,pred)), 'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
         'macro_f1':float(f1_score(y,pred,labels=classes,average='macro',zero_division=0)),
         'per_class_recall':recall_score(y,pred,labels=classes,average=None,zero_division=0).tolist(),
         'confusion_matrix':confusion_matrix(y,pred,labels=classes).tolist(),
         'class_counts':{str(k):int((y==k).sum()) for k in classes}}
    roc={};ap={};pr={}
    for j,k in enumerate(classes):
        binary=(y==k).astype(int)
        roc[str(k)]=float(roc_auc_score(binary,p[:,j])) if len(np.unique(binary))==2 else None
        ap[str(k)]=float(average_precision_score(binary,p[:,j])) if 0<binary.sum()<len(binary) else None
        if 0<binary.sum()<len(binary):
            precision,recall,_=precision_recall_curve(binary,p[:,j])
            pr[str(k)]=float(auc(recall,precision))
        else:pr[str(k)]=None
    out.update(roc_auc_ovr=roc,average_precision_ovr=ap,pr_auc_trapezoidal_ovr=pr,
               macro_pr_auc_trapezoidal_ovr=float(np.mean(list(pr.values()))) if all(v is not None for v in pr.values()) else None,
               macro_roc_auc_ovr=float(np.mean(list(roc.values()))) if all(v is not None for v in roc.values()) else None,
               macro_average_precision_ovr=float(np.mean(list(ap.values()))) if all(v is not None for v in ap.values()) else None)
    return out

def subject_bootstrap(values,n=2000,seed=42):
    values=np.asarray(values,float)
    if len(values)<2:raise ValueError('CI needs multiple subjects')
    rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(n,len(values)))].mean(axis=1)
    return {'mean':float(values.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'unit':'subject','n_subjects':len(values),'interpretation':'conditional on fitted LOSO predictions; not full retraining uncertainty'}

def run_loso(frame,c,name,site,task='three_class',classes=(1,2,3)):
    fold_results=[];predictions=[];partitions=[]
    for tr,te in loso(frame,classes):
        train=frame.iloc[tr];test=frame.iloc[te];sid=int(test.subject_id.iloc[0])
        model,cols,search=tune_fit(train,name,c)
        if list(model.classes_)!=list(classes):raise ValueError('Unexpected probability class order')
        folder=Path(c['output']['root'])/'models'/f'{task}_{site}_{name}'
        folder.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,folder/f'S{sid}.joblib')
        estimator=model.named_steps['model']
        relevance={}
        if hasattr(estimator,'feature_importances_'):
            relevance=dict(zip(cols,map(float,estimator.feature_importances_)))
        elif hasattr(estimator,'coef_'):
            relevance=dict(zip(cols,map(float,np.mean(abs(estimator.coef_),axis=0))))
        (folder/f'S{sid}_metadata.json').write_text(json.dumps({
            'features':cols,'fit_subjects':sorted(map(int,train.subject_id.unique())),
            'test_subject':sid,'classes':list(map(int,model.classes_)),
            'training_only_relevance':relevance,
            'relevance_note':'descriptive impurity importance or mean absolute standardized coefficient; not causal',
            'search':search},indent=2))
        p=model.predict_proba(test[cols])
        fold_results.append({'subject_id':sid,**metrics(test.label,p,classes),**search,
                             'n_windows':len(test),'n_episodes':test.episode_id.nunique()})
        q=test[['window_id','subject_id','episode_id','start','end','label']].copy()
        for j,k in enumerate(classes):q[f'p_{k}']=p[:,j]
        predictions.append(q)
        partitions.append({'test_subject':sid,'train_subjects':sorted(map(int,train.subject_id.unique()))})
    return summarize_folds(frame,c,name,site,task,classes,fold_results,partitions),pd.concat(predictions,ignore_index=True)

def summarize_folds(frame,c,name,site,task,classes,fold_results,partitions):
    result={'config_sha256':fingerprint(c),'task':task,'site':site,'model':name,'classes':list(classes),
            'chance_balanced_accuracy':1/len(classes),'n_subjects':frame.subject_id.nunique(),
            'n_windows':len(frame),'n_episodes':frame.episode_id.nunique(),
            'folds':fold_results,'partitions':partitions,
            'primary':subject_bootstrap([f['balanced_accuracy'] for f in fold_results],c['splits']['bootstrap_repetitions'],c['splits']['seed']),
            'permutation':{'status':PERMUTATION_STATUS,'p_value':None}}
    for metric in ['macro_f1','accuracy','macro_roc_auc_ovr','macro_average_precision_ovr','macro_pr_auc_trapezoidal_ovr']:
        result[metric]=subject_bootstrap([f[metric] for f in fold_results],c['splits']['bootstrap_repetitions'],c['splits']['seed'])
    result['confusion_matrix_sum']=np.sum([f['confusion_matrix'] for f in fold_results],axis=0).tolist()
    return result

def save_run(result,pred,c,tag):
    root=Path(c['output']['root'])/'results';root.mkdir(parents=True,exist_ok=True)
    (root/f'{tag}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pred.to_csv(root/f'{tag}_predictions.csv',index=False)

def paired_difference(a,b,c):
    a={f['subject_id']:f['balanced_accuracy'] for f in a['folds']}
    b={f['subject_id']:f['balanced_accuracy'] for f in b['folds']}
    if set(a)!=set(b):raise ValueError('Subject sets must match')
    return subject_bootstrap([a[k]-b[k] for k in sorted(a)],c['splits']['bootstrap_repetitions'],c['splits']['seed'])
