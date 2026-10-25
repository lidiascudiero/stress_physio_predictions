"""Secondary empirical reliability; optional conformal output is explicitly exploratory."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from .models import tune_fit
from .splits import loso,calibration_split,assert_disjoint

class ProbabilityCalibrator:
    # Multinomial sigmoid calibration on log probabilities. Subject-balanced weights.
    def fit(self,p,y,subjects):
        subjects=np.asarray(subjects)
        weights=np.array([1/np.sum(subjects==s) for s in subjects])
        weights*=len(weights)/weights.sum()
        self.model=LogisticRegression(C=1.,max_iter=2000).fit(np.log(np.clip(p,1e-6,1)),y,sample_weight=weights)
        return self
    def predict(self,p):return self.model.predict_proba(np.log(np.clip(p,1e-6,1)))

def reliability(y,p,classes,bins=10):
    y=np.asarray(y);classes=np.asarray(classes);rows=[]
    for j,k in enumerate(classes):
        for lo in np.linspace(0,1,bins,endpoint=False):
            hi=lo+1/bins;mask=(p[:,j]>=lo)&((p[:,j]<hi) if hi<1-1e-9 else (p[:,j]<=1))
            if mask.any():rows.append({'class':int(k),'bin_low':float(lo),'count':int(mask.sum()),
                'mean_probability':float(p[mask,j].mean()),'observed_frequency':float((y[mask]==k).mean())})
    return pd.DataFrame(rows)

def risk_curve(y,p,classes):
    confidence=p.max(axis=1);pred=np.asarray(classes)[p.argmax(axis=1)];rows=[]
    for threshold in np.linspace(0,1,51):
        m=confidence>=threshold
        rows.append({'threshold':float(threshold),'acceptance':float(m.mean()),
                     'risk':float((pred[m]!=np.asarray(y)[m]).mean()) if m.any() else None})
    return pd.DataFrame(rows)

def choose_threshold(y,p,classes,target,min_n):
    # Chosen solely on calibration data; performance is descriptive there.
    pred=np.asarray(classes)[p.argmax(axis=1)];confidence=p.max(axis=1)
    for t in sorted(set(confidence)):
        m=confidence>=t
        if m.sum()>=min_n and np.mean(pred[m]!=np.asarray(y)[m])<=target:return float(t)
    return 1.000001  # Explicit abstain-all if no eligible threshold.

def conformal_sets(cal_y,cal_p,test_p,classes,alpha):
    # Window-pooled experimental score: no exchangeability guarantee claimed.
    lookup={k:i for i,k in enumerate(classes)}
    scores=1-cal_p[np.arange(len(cal_y)),[lookup[k] for k in cal_y]]
    rank=int(np.ceil((len(scores)+1)*(1-alpha)))
    q=float(np.sort(scores)[rank-1]) if rank<=len(scores) else float('inf')
    return 1-test_p<=q

def run_uncertainty(frame,c,classes=(1,2,3)):
    outputs=[];logs=[];curves=[];diagrams=[]
    for tr,te in loso(frame,classes):
        outer=frame.iloc[tr];test=frame.iloc[te];sid=int(test.subject_id.iloc[0])
        fit,cal=calibration_split(outer,c['uncertainty']['calibration_subjects'],c['splits']['seed']+sid)
        assert_disjoint(fit.subject_id,cal.subject_id,test.subject_id)
        fitted,cols,search=tune_fit(fit,'random_forest',c)
        pc=fitted.predict_proba(cal[cols]);pt=fitted.predict_proba(test[cols])
        calibrator=ProbabilityCalibrator().fit(pc,cal.label,cal.subject_id)
        p=calibrator.predict(pt)
        # Threshold is fitted on raw-model calibration probabilities, independently of test.
        threshold=choose_threshold(cal.label,pc,classes,c['uncertainty']['target_calibration_error'],c['uncertainty']['min_accepted_calibration'])
        members=[]
        for member in range(c['uncertainty']['ensemble_members']):
            m=clone(fitted).set_params(model__random_state=c['splits']['seed']+member)
            m.fit(fit[cols],fit.label);members.append(m.predict_proba(test[cols]))
        disagreement=np.var(np.stack(members),axis=0).mean(axis=1)
        o=test[['window_id','subject_id','label']].copy()
        o['disagreement']=disagreement;o['accepted']=pt.max(axis=1)>=threshold
        o['prediction_raw']=np.asarray(classes)[pt.argmax(axis=1)]
        for j,k in enumerate(classes):o[f'p_calibrated_{k}']=p[:,j]
        if c['uncertainty']['conformal_exploratory']:
            sets=conformal_sets(cal.label.to_numpy(),pc,pt,classes,c['uncertainty']['alpha'])
            o['set_size']=sets.sum(axis=1)
            o['set_covers']=sets[np.arange(len(test)),[list(classes).index(k) for k in test.label]]
        outputs.append(o)
        d=reliability(test.label,p,classes);d['subject_id']=sid;diagrams.append(d)
        curve=risk_curve(test.label,pt,classes);curve['subject_id']=sid;curves.append(curve)
        logs.append({'test_subject':sid,'fit_subjects':sorted(map(int,fit.subject_id.unique())),
                     'calibration_subjects':sorted(map(int,cal.subject_id.unique())),
                     'threshold_on_raw_probabilities':threshold,'search':search,
                     'calibration_note':'Probability calibrator and raw-score abstention threshold share calibration subjects; neither uses test',
                     'conformal_status':'exploratory window-pooled; no guaranteed new-subject coverage' if c['uncertainty']['conformal_exploratory'] else 'not run'})
    return pd.concat(outputs,ignore_index=True),logs,pd.concat(diagrams,ignore_index=True),pd.concat(curves,ignore_index=True)
