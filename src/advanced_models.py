"""Exploratory estimators; no random window-level early stopping split."""
import numpy as np
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight

class EncodedClassifier(ClassifierMixin,BaseEstimator):
    """Encode WESAD labels 1..K as 0..K-1 inside each training fit (XGBoost)."""
    def __init__(self,estimator):self.estimator=estimator
    def fit(self,X,y):
        self.encoder_=LabelEncoder().fit(y);self.classes_=self.encoder_.classes_
        encoded=self.encoder_.transform(y)
        self.estimator_=clone(self.estimator)
        self.estimator_.fit(X,encoded,sample_weight=compute_sample_weight('balanced',encoded))
        self.n_features_in_=X.shape[1]
        return self
    def predict(self,X):return self.encoder_.inverse_transform(self.estimator_.predict(X).astype(int))
    def predict_proba(self,X):return self.estimator_.predict_proba(X)

def advanced_classifier(name,c,seed):
    e=c.get('exploratory',{});trees=e.get('boosting_trees',200)
    if name=='xgboost':
        try:from xgboost import XGBClassifier
        except ImportError as ex:raise ImportError('Install requirements-exploratory.txt to run XGBoost') from ex
        return EncodedClassifier(XGBClassifier(n_estimators=trees,max_depth=3,learning_rate=.05,
                    tree_method='hist',n_jobs=1,random_state=seed,verbosity=0))
    if name=='lightgbm':
        try:from lightgbm import LGBMClassifier
        except ImportError as ex:raise ImportError('Install requirements-exploratory.txt to run LightGBM') from ex
        return LGBMClassifier(n_estimators=trees,num_leaves=7,learning_rate=.05,
                    min_child_samples=5,class_weight='balanced',n_jobs=1,random_state=seed,verbosity=-1)
    if name=='mlp':
        return MLPClassifier(hidden_layer_sizes=(16,),alpha=.001,solver='lbfgs',
                    max_iter=e.get('mlp_max_iter',500),early_stopping=False,random_state=seed)
    raise ValueError(name)
