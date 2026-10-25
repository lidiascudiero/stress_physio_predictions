"""Exploratory late fusion and training-only modality masking under nested LOSO."""
import joblib
from pathlib import Path
import numpy as np
import pandas as pd
from .models import tune_fit,feature_columns
from .splits import loso,assert_disjoint
from .evaluation import metrics,summarize_folds

SCENARIOS=('both_present','chest_missing','wrist_missing')

def mask_modality(frame,scenario,indicators=True):
    if scenario not in SCENARIOS:raise ValueError('Unknown missing modality scenario')
    f=frame.copy()
    for site in ['chest','wrist']:
        cols=[x for x in feature_columns(f) if x.startswith(site+'__') and '__available' not in x]
        missing=scenario==site+'_missing'
        if missing:f.loc[:,cols]=np.nan
        if indicators:f[site+'__available']=0. if missing else 1.
    return f

def training_views(frame):
    views=[]
    for scenario in SCENARIOS:
        v=mask_modality(frame,scenario)
        v['window_id']=v.window_id.astype(str)+'::'+scenario
        views.append(v)
    return pd.concat(views,ignore_index=True)

def _site_frame(frame,site):
    return frame.drop(columns=[x for x in feature_columns(frame) if not x.startswith(site+'__')])

def run_fusion(frame,c,method,classes=(1,2,3)):
    if method not in ['late_fusion','modality_dropout']:raise ValueError(method)
    fold_results={s:[] for s in (('both_present',) if method=='late_fusion' else SCENARIOS)}
    predictions={s:[] for s in fold_results};partitions=[]
    for tr,te in loso(frame,classes):
        train=frame.iloc[tr];test=frame.iloc[te];sid=int(test.subject_id.iloc[0])
        assert_disjoint(train.subject_id,test.subject_id)
        folder=Path(c['output']['root'])/'models'/method;folder.mkdir(parents=True,exist_ok=True)
        if method=='late_fusion':
            probabilities=[];search={};bundle={}
            for site in ['chest','wrist']:
                model,cols,trace=tune_fit(_site_frame(train,site),'logistic_regression',c)
                if list(model.classes_)!=list(classes):raise ValueError('Probability classes do not align')
                probabilities.append(model.predict_proba(test[cols]));search[site]=trace
                bundle[site]={'model':model,'features':cols}
            # Equal weights fixed before any test results; no learned stacking weights.
            output={'both_present':np.mean(probabilities,axis=0)}
            metadata={'fixed_probability_weights':{'chest':.5,'wrist':.5},'search':search}
        else:
            augmented=training_views(train)
            model,cols,trace=tune_fit(augmented,'logistic_regression',c)
            output={s:model.predict_proba(mask_modality(test,s)[cols]) for s in SCENARIOS}
            bundle={'model':model,'features':cols}
            metadata={'training_views':list(SCENARIOS),'validation_objective':'mean subject BA across all three views',
                      'missing_representation':'NaN then training-fold median imputation, plus availability indicators','search':trace}
        joblib.dump({'bundle':bundle,'metadata':metadata,'fit_subjects':sorted(map(int,train.subject_id.unique()))},folder/f'S{sid}.joblib')
        for scenario,p in output.items():
            fold_results[scenario].append({'subject_id':sid,**metrics(test.label,p,classes),
                                          'n_windows':len(test),'n_episodes':test.episode_id.nunique(),'design':metadata})
            q=test[['window_id','subject_id','episode_id','start','end','label']].copy()
            for j,k in enumerate(classes):q[f'p_{k}']=p[:,j]
            predictions[scenario].append(q)
        partitions.append({'test_subject':sid,'train_subjects':sorted(map(int,train.subject_id.unique()))})
    results=[]
    for scenario,folds in fold_results.items():
        result=summarize_folds(frame,c,'logistic_regression',method,'three_class_'+scenario,classes,folds,partitions)
        result.update(analysis_role='exploratory',scenario=scenario,method=method)
        results.append((result,pd.concat(predictions[scenario],ignore_index=True)))
    return results
