import numpy as np
import warnings
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import balanced_accuracy_score
from .splits import inner

def feature_columns(frame):
    cols=[x for x in frame if x.startswith(('chest__','wrist__'))]
    if not cols:raise ValueError('No sensor feature columns')
    return cols

def pipeline(name,c,seed=None):
    seed=c['splits']['seed'] if seed is None else seed
    if name=='majority':model=DummyClassifier(strategy='most_frequent')
    elif name=='logistic_regression':model=LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed)
    elif name=='random_forest':model=RandomForestClassifier(n_estimators=c['models']['rf_trees'],class_weight='balanced',random_state=seed,n_jobs=1)
    elif name in ['xgboost','lightgbm','mlp']:
        from .advanced_models import advanced_classifier
        model=advanced_classifier(name,c,seed)
    else:raise ValueError(name)
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ).set_output(transform="pandas"),
        ),
        (
            "scaler",
            StandardScaler().set_output(transform="pandas"),
        ),
        ("model", model),
    ])

def grid(name,c):
    if name=='logistic_regression':return {'model__C':c['models']['logistic_c']}
    if name=='random_forest':return {'model__max_depth':c['models']['rf_depth']}
    if name=='xgboost':return {'model__estimator__max_depth':c.get('exploratory',{}).get('xgb_depth',[3,5])}
    if name=='lightgbm':return {'model__num_leaves':c.get('exploratory',{}).get('lgbm_leaves',[7,15])}
    if name=='mlp':return {'model__alpha':c.get('exploratory',{}).get('mlp_alpha',[0.001,0.01])}
    return {}

def subject_mean_score(frame,pred):
    vals=[]
    for sid in sorted(frame.subject_id.unique()):
        mask=frame.subject_id.to_numpy()==sid
        if set(frame.label[mask])!=set(frame.label):raise ValueError('Inner validation subject lacks class')
        vals.append(balanced_accuracy_score(frame.label[mask],np.asarray(pred)[mask]))
    return float(np.mean(vals))

def tune_fit(frame,name,c,seed=None):
    cols=feature_columns(frame);base=pipeline(name,c,seed)
    best=None;best_score=-np.inf;trace=[]
    for params in ParameterGrid(grid(name,c)):
        subject_scores=[];fit_warnings=[]
        for tr,va in inner(frame,c['splits']['inner_folds']):
            model=clone(base).set_params(**params)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                model.fit(frame.iloc[tr][cols],frame.iloc[tr].label)
            fit_warnings.extend({'category':w.category.__name__,'message':str(w.message)} for w in observed)
            v=frame.iloc[va]
            pred=model.predict(v[cols])
            for sid in v.subject_id.unique():
                m=v.subject_id.to_numpy()==sid
                if set(v.label[m])!=set(frame.label):raise ValueError('Validation subject lacks required class')
                subject_scores.append(balanced_accuracy_score(v.label[m],pred[m]))
        score=float(np.mean(subject_scores));trace.append({'params':params,'subject_mean_balanced_accuracy':score,'fit_warnings':fit_warnings})
        if score>best_score:best_score=score;best=params
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter('always')
        fitted=clone(base).set_params(**best).fit(frame[cols],frame.label)
    final_warnings=[{'category':w.category.__name__,'message':str(w.message)} for w in observed]
    return fitted,cols,{'best_params':best,'inner_score':best_score,'search':trace,'final_fit_warnings':final_warnings}
