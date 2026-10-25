import json,unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.config import load,smoke_config,fingerprint,setup
from src.evaluation import metrics,run_loso
from src.models import tune_fit
from src.advanced_models import EncodedClassifier
from src.exploratory_fusion import training_views,mask_modality,run_fusion
from src.exploratory import require_confirmatory,four_class_eligibility
from src.uncertainty import run_uncertainty
from src.synthetic import make_questionnaires,make_records
from src.self_reports import parse_self_reports,analyze
from tests.test_splits import frame

class ExploratoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.c=smoke_config(load());self.c['output']['root']=self.tmp.name;setup(self.c)
        self.f=frame();self.f['wrist__x']=self.f['chest__x']*.8
    def test_trapezoid_and_ap_are_distinct(self):
        y=np.array([1,2,1,2]);p=np.array([[.9,.1],[.8,.2],[.7,.3],[.1,.9]])
        r=metrics(y,p,(1,2))
        self.assertAlmostEqual(r['average_precision_ovr']['1'],5/6)
        self.assertAlmostEqual(r['pr_auc_trapezoidal_ovr']['1'],19/24)
        self.assertNotEqual(r['macro_pr_auc_trapezoidal_ovr'],r['macro_average_precision_ovr'])
    def test_constant_score_has_different_pr_baselines(self):
        r=metrics(np.array([1,2,3]*3),np.tile([1.,0.,0.],(9,1)),(1,2,3))
        self.assertAlmostEqual(r['macro_average_precision_ovr'],1/3)
        self.assertAlmostEqual(r['macro_pr_auc_trapezoidal_ovr'],2/3)
    def test_encoded_classifier_preserves_labels_and_probabilities(self):
        model=EncodedClassifier(RandomForestClassifier(n_estimators=5,random_state=42)).fit(self.f[['chest__x']],self.f.label)
        self.assertEqual(list(model.classes_),[1,2,3])
        self.assertTrue(set(model.predict(self.f[['chest__x']]))<={1,2,3})
        np.testing.assert_allclose(model.predict_proba(self.f[['chest__x']]).sum(axis=1),1)
    def test_gate_blocks_missing_or_mismatched_confirmatory_runs(self):
        with self.assertRaises(ValueError):require_confirmatory(self.c)
        root=Path(self.tmp.name);(root/'protocol_lock.json').write_text(json.dumps({'config_sha256':fingerprint(self.c)}))
        with self.assertRaises(ValueError):require_confirmatory(self.c)
    def test_modality_training_views_keep_groups(self):
        aug=training_views(self.f)
        self.assertEqual(set(aug.subject_id),set(self.f.subject_id))
        self.assertEqual(len(aug),len(self.f)*3)
        self.assertFalse(aug.window_id.duplicated().any())
        masked=mask_modality(self.f,'chest_missing')
        self.assertTrue(masked.chest__x.isna().all());self.assertTrue((masked.chest__available==0).all())
        self.assertFalse(masked.wrist__x.isna().any())
        self.assertFalse(self.f.chest__x.isna().any())
    def test_late_fusion_and_dropout_nested_loso(self):
        for method in ['late_fusion','modality_dropout']:
            outputs=run_fusion(self.f,self.c,method)
            self.assertEqual(len(outputs),1 if method=='late_fusion' else 3)
            for result,pred in outputs:
                self.assertEqual(result['analysis_role'],'exploratory')
                self.assertEqual(len(pred),len(self.f))
                self.assertIn('macro_pr_auc_trapezoidal_ovr',result)
                for part in result['partitions']:
                    self.assertNotIn(part['test_subject'],part['train_subjects'])
                np.testing.assert_allclose(pred[['p_1','p_2','p_3']].sum(axis=1),1)
    def test_four_class_requires_each_subject_each_class(self):
        self.assertFalse(four_class_eligibility(self.f,self.c)['eligible'])
        extra=self.f[self.f.label==3].copy();extra['label']=4;extra.window_id+='four'
        f=pd.concat([self.f,extra],ignore_index=True)
        self.assertTrue(four_class_eligibility(f,self.c)['eligible'])
        r,p=run_loso(f,self.c,'logistic_regression','early_fusion','four_class',(1,2,3,4))
        self.assertEqual(r['classes'],[1,2,3,4]);self.assertEqual(r['chance_balanced_accuracy'],.25)
    def test_questionnaires_are_episode_level_and_synthetic_separate(self):
        raw=make_questionnaires(Path(self.tmp.name)/'synthetic_questionnaires')
        q=parse_self_reports(raw/'S2_quest.csv')
        self.assertEqual(len(q),5);self.assertEqual(set(q.label),{1,2,3,4})
        self.assertTrue(q.stai6_keyed_sum.between(6,24).all())
        data,summary,corr=analyze(raw,None)
        self.assertEqual(len(data),30);self.assertEqual(len(summary),5);self.assertTrue(corr.empty)
        self.assertNotIn('window_id',data)
    def test_conformal_produces_per_subject_sets_without_test_calibration(self):
        self.c['uncertainty']['conformal_exploratory']=True
        pred,logs,_,_=run_uncertainty(self.f,self.c)
        self.assertIn('set_covers',pred);self.assertIn('set_size',pred)
        self.assertTrue(pred.set_size.between(0,3).all())
        for item in logs:
            self.assertNotIn(item['test_subject'],item['fit_subjects']+item['calibration_subjects'])
    def _check_native(self,name):
        model,cols,trace=tune_fit(self.f,name,self.c)
        p=model.predict_proba(self.f[cols])
        self.assertEqual(p.shape,(len(self.f),3));np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-6)
        self.assertEqual(list(model.classes_),[1,2,3]);self.assertTrue(trace['search'])
    def test_native_xgboost(self):
        try:import xgboost
        except ImportError:self.skipTest('Optional XGBoost dependency is unavailable')
        self._check_native('xgboost')
    def test_native_lightgbm(self):
        try:import lightgbm
        except ImportError:self.skipTest('Optional LightGBM dependency is unavailable')
        self._check_native('lightgbm')
    def test_native_mlp(self):self._check_native('mlp')
