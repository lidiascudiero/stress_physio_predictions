import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold

def assert_disjoint(*groups):
    sets=[set(g) for g in groups]
    for i,a in enumerate(sets):
        for b in sets[i+1:]:
            if a & b:raise AssertionError('Subject leakage')

def loso(frame,classes):
    if frame.window_id.duplicated().any():raise ValueError('Duplicate window_id')
    if not len(frame):raise ValueError('No valid windows')
    if frame.subject_id.nunique()<3:raise ValueError('Too few subjects')
    for tr,te in LeaveOneGroupOut().split(frame,frame.label,frame.subject_id):
        assert_disjoint(frame.iloc[tr].subject_id,frame.iloc[te].subject_id)
        if set(frame.iloc[tr].label)!=set(classes) or set(frame.iloc[te].label)!=set(classes):
            raise ValueError('A LOSO partition lacks a required class; report as not executable')
        yield tr,te

def inner(frame,n):
    n=min(n,frame.subject_id.nunique())
    if n<2:raise ValueError('At least two training subject groups required')
    for tr,va in GroupKFold(n_splits=n).split(frame,frame.label,frame.subject_id):
        assert_disjoint(frame.iloc[tr].subject_id,frame.iloc[va].subject_id)
        if set(frame.iloc[tr].label)!=set(frame.label):raise ValueError('Inner training class missing')
        yield tr,va

def calibration_split(frame,n,seed):
    subjects=np.array(sorted(frame.subject_id.unique()))
    if n<1 or len(subjects)-n<2:raise ValueError('Insufficient subjects for separate calibration and inner CV')
    chosen=np.random.default_rng(seed).choice(subjects,n,replace=False)
    cal=frame[frame.subject_id.isin(chosen)].copy();fit=frame[~frame.subject_id.isin(chosen)].copy()
    assert_disjoint(fit.subject_id,cal.subject_id)
    if set(cal.label)!=set(frame.label) or set(fit.label)!=set(frame.label):
        raise ValueError('Calibration or fit class missing')
    return fit,cal

def assert_no_temporal_overlap(train,test):
    for sid in set(train.subject_id)&set(test.subject_id):
        a=train[train.subject_id==sid];b=test[test.subject_id==sid]
        for r in a.itertuples():
            if ((b.start<r.end)&(b.end>r.start)).any():raise AssertionError('Shared raw samples')
