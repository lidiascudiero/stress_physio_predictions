import unittest
import numpy as np
import pandas as pd
from src.splits import loso,inner,calibration_split,assert_disjoint,assert_no_temporal_overlap


def frame():
    return pd.DataFrame([dict(window_id=f'{s}_{k}_{i}',subject_id=s,episode_id=f'{s}_{k}',
                             label=k,start=k*200+i*30,end=k*200+i*30+60,
                             chest__x=float(k+i+s/20),valid=True)
                         for s in range(6) for k in [1,2,3] for i in range(3)])

class SplitTests(unittest.TestCase):
    def test_loso_test_once_training_repeats(self):
        f=frame();seen=[]
        for tr,te in loso(f,(1,2,3)):
            assert_disjoint(f.iloc[tr].subject_id,f.iloc[te].subject_id)
            assert_no_temporal_overlap(f.iloc[tr],f.iloc[te])
            seen.extend(f.iloc[te].window_id)
            for a,b in inner(f.iloc[tr],2):
                assert_disjoint(f.iloc[tr].iloc[a].subject_id,f.iloc[tr].iloc[b].subject_id)
        self.assertEqual(sorted(seen),sorted(f.window_id))
    def test_calibration_is_disjoint(self):
        f=frame();tr,te=next(loso(f,(1,2,3)))
        fit,cal=calibration_split(f.iloc[tr],2,42)
        assert_disjoint(fit.subject_id,cal.subject_id,f.iloc[te].subject_id)
    def test_shared_samples_rejected(self):
        f=frame();a=f.iloc[[0]];b=f.iloc[[1]]
        with self.assertRaises(AssertionError):assert_no_temporal_overlap(a,b)
    def test_duplicate_and_missing_class(self):
        f=frame()
        with self.assertRaises(ValueError):list(loso(pd.concat([f,f.iloc[[0]]]),(1,2,3)))
        f=f[~((f.subject_id==0)&(f.label==3))]
        with self.assertRaises(ValueError):list(loso(f,(1,2,3)))
