import unittest
import numpy as np
from src.config import load,smoke_config
from src.synthetic import make_records
from src.features import extract,beat_features
from src.fusion import paired

class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=smoke_config(load());cls.records=make_records(1,120)
        cls.tables,cls.qc=extract(cls.records,cls.c)
    def test_boundaries_and_no_duplicates(self):
        f=self.tables['chest']
        self.assertFalse(f.window_id.duplicated().any())
        self.assertEqual(set(f.label),{1,2,3})
        self.assertTrue(np.allclose(f.end-f.start,60))
        for row in f.itertuples():
            labels=self.records[0]['labels'][round(row.start*700):round(row.end*700)]
            self.assertTrue(np.all(labels==row.label))
        self.assertEqual(len(f),9)
    def test_no_forbidden_features(self):
        names=' '.join(self.tables['chest'].columns).lower()
        self.assertNotIn('lf_hf',names);self.assertNotIn('entropy',names)
        self.assertTrue(any('prv_' in x for x in self.tables['wrist']))
    def test_pairing_and_invalid_window(self):
        chest=self.tables['chest'].copy();wrist=self.tables['wrist'].copy()
        wrist.loc[0,'valid']=False
        p=paired(chest,wrist)
        self.assertNotIn(wrist.iloc[0].window_id,set(p.window_id))
        self.assertEqual(len(p),8)
    def test_peak_intervals_known(self):
        fs=700;t=np.arange(60*fs)/fs;x=np.zeros(len(t))
        for beat in np.arange(.5,60,1):x+=np.exp(-((t-beat)/.012)**2)
        f,ok=beat_features(x,fs,self.c)
        self.assertTrue(ok);self.assertAlmostEqual(f['mean_hr'],60,delta=.5)
