import unittest
import numpy as np
from src.config import load,smoke_config
from src.synthetic import make_records
from src.signal_qc import alignment,assess
from src.wesad_io import LABELS,read_questionnaire
from tempfile import TemporaryDirectory
from pathlib import Path

class AlignmentTests(unittest.TestCase):
    def test_native_durations(self):
        r=make_records(1,60)[0]
        self.assertTrue(alignment(r).duration_ok.all())
        self.assertFalse(alignment(r).synchronization_verified.any())
        r['signal']['wrist']['BVP']=r['signal']['wrist']['BVP'][:-128]
        self.assertFalse(alignment(r).duration_ok.all())
    def test_codes(self):
        # Contract test only; cannot verify an unseen external README.
        self.assertEqual(LABELS,{1:'baseline',2:'stress',3:'amusement',4:'meditation'})
        self.assertEqual(load()['dataset']['labels'],LABELS)
    def test_qc(self):
        c=load()
        self.assertFalse(assess(np.zeros(240),4,'EDA',c)['valid'])
        self.assertFalse(assess(np.full(240,100.),4,'TEMP',c)['valid'])
    def test_questionnaire_retains_units(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'S2_quest.csv';p.write_text('# ORDER;Base;TSST\n# START;0;20\n# END;20;30\n# PANAS;1;2\n')
            q=read_questionnaire(p)
            self.assertEqual(q['start_rows'][0][1],'0')
            self.assertEqual(len(q['order_rows']),1)
