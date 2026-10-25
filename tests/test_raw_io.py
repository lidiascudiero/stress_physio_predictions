import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from src.raw_io import protocol_seconds,convert_respiban,load_e4,schedule
from src.import_local import normalized_name,import_subject
from src.paths import project_root,project_path

class RawFormatTests(unittest.TestCase):
    def test_minutes_seconds(self):
        self.assertEqual(protocol_seconds('7.08'),428)
        self.assertEqual(protocol_seconds('50.3'),3030)
        self.assertEqual(protocol_seconds('77.1'),4630)
        with self.assertRaises(ValueError):protocol_seconds('7.60')
    def test_e4_headers_and_acc_units(self):
        with TemporaryDirectory() as d:
            p=Path(d)/'ACC.csv';p.write_text('1495437325,1495437325,1495437325\n32,32,32\n64,0,-64\n')
            a=load_e4(p,'ACC');np.testing.assert_allclose(a['values'],[[1,0,-1]])
            self.assertEqual(a['duration_seconds'],1/32)
            with self.assertRaises(ValueError):load_e4(p,'IBI')
    def test_respiban_formula_units(self):
        raw=np.full((1,8),32768.)
        a=convert_respiban(raw)
        self.assertAlmostEqual(a['ECG'][0,0],0)
        self.assertAlmostEqual(a['EDA'][0,0],12.5)
        self.assertAlmostEqual(a['Resp'][0,0],0)
        self.assertTrue(np.isfinite(a['Temp']).all())
        np.testing.assert_allclose(a['ACC'],np.full((1,3),-.0464))
    def test_names_and_import(self):
        self.assertEqual(normalized_name('S2_quest(1).csv'),'S2_quest.csv')
        with TemporaryDirectory() as d:
            root=Path(d);source=root/'source';source.mkdir();dest=root/'data'
            (source/'S2_quest(1).csv').write_text('original')
            imported=import_subject(source,dest,'S2')
            self.assertEqual((dest/'S2/S2_quest.csv').read_text(),'original')
            self.assertEqual(len(imported),1)
            (source/'S2_quest(1).csv').write_text('changed')
            with self.assertRaises(FileExistsError):import_subject(source,dest,'S2')
    def test_portable_paths(self):
        self.assertTrue((project_root()/'configs/config.yaml').exists())
        self.assertTrue(project_path('data/raw').is_relative_to(project_root()))
        with self.assertRaises(ValueError):project_path('../outside')

    def test_raw_trailing_delimiter(self):
        import json
        from src.raw_io import audit_respiban
        from src.config import load
        meta={'sensor':['ECG','EDA','EMG','TEMP','XYZ','XYZ','XYZ','RESPIRATION'],
              'sampling rate':700,'column':['nSeq','DI']+[f'CH{i}' for i in range(1,9)],
              'date':'2017-5-22','time':'9:39:1.0'}
        with TemporaryDirectory() as d:
            p=Path(d)/'S2_respiban.txt'
            data='\t'.join(map(str,[0,0]+[32768]*8))+'\t\n'
            p.write_text('# OpenSignals Text File Format\n# '+json.dumps({'device':meta})+'\n# EndOfHeader\n'+data)
            result=audit_respiban(p,load(),chunksize=1)
            self.assertEqual(result['n_samples'],1)
            self.assertEqual(result['sequence_unusual_positive_jumps'],0)
