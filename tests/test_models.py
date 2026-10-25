import unittest
from tempfile import TemporaryDirectory
import numpy as np
from src.config import load,smoke_config
from src.models import pipeline,tune_fit,feature_columns
from src.evaluation import run_loso,subject_bootstrap,paired_difference
from src.uncertainty import ProbabilityCalibrator,run_uncertainty,conformal_sets
from tests.test_splits import frame

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=smoke_config(load());self.c['output']['root']=self.tmp.name;self.f=frame()
    def test_training_only_scaler_and_imputer(self):
        train=self.f[self.f.subject_id<4];test=self.f[self.f.subject_id>=4].copy()
        test['chest__x']=1e9
        cols=feature_columns(train);m=pipeline('logistic_regression',self.c).fit(train[cols],train.label)
        mean=m.named_steps['scaler'].mean_.copy()
        m.predict(test[cols]);np.testing.assert_allclose(mean,train[cols].mean())
        np.testing.assert_allclose(m.named_steps['imputer'].statistics_,train[cols].median())
        self.assertNotIn('label',cols);self.assertNotIn('subject_id',cols)
        self.assertEqual(list(m.named_steps),['imputer','scaler','model']) # no global feature selector
    def test_results_baseline_ci_permutation(self):
        r,p=run_loso(self.f,self.c,'majority','chest')
        self.assertAlmostEqual(r['primary']['mean'],1/3)
        self.assertEqual(r['primary']['unit'],'subject')
        self.assertIsNone(r['permutation']['p_value'])
        self.assertEqual(len(p),len(self.f));self.assertEqual(r['n_subjects'],6)
    def test_bootstrap_subject_values(self):
        r=subject_bootstrap([0.,1.],100,42)
        self.assertEqual(r['n_subjects'],2);self.assertEqual(r['mean'],.5)
    def test_calibrator_probability_simplex(self):
        p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]*4);y=np.array([1,2,3]*4)
        m=ProbabilityCalibrator().fit(p,y,np.repeat([1,2],6));q=m.predict(p)
        np.testing.assert_allclose(q.sum(axis=1),1)
        self.assertTrue(np.isfinite(q).all()) # no promise of empirical calibration perfection
    def test_uncertainty_separates_subjects(self):
        p,logs,d,curves=run_uncertainty(self.f,self.c)
        for row in logs:
            self.assertNotIn(row['test_subject'],row['fit_subjects']+row['calibration_subjects'])
            self.assertFalse(set(row['fit_subjects'])&set(row['calibration_subjects']))
        self.assertEqual(len(p),len(self.f));self.assertNotIn('set_size',p)
    def test_conformal_finite_sample_rank(self):
        sets=conformal_sets(np.array([1,2]),np.array([[.8,.2],[.2,.8]]),np.array([[.5,.5]]),(1,2),.01)
        self.assertTrue(sets.all())
