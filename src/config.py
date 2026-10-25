from pathlib import Path
import copy, hashlib, json, platform, random
import numpy as np
import yaml

def load(path='configs/config.yaml'):
    c = yaml.safe_load(Path(path).read_text())
    if not 0 <= c['features']['overlap'] < 1:
        raise ValueError('overlap must be in [0,1)')
    if c['features']['window_seconds'] <= 0:
        raise ValueError('window must be positive')
    if c['dataset']['labels'] != {1:'baseline',2:'stress',3:'amusement',4:'meditation'}:
        raise ValueError('Unexpected label mapping; inspect original WESAD README')
    return c

def setup(c):
    random.seed(c['splits']['seed']); np.random.seed(c['splits']['seed'])
    for d in ['qc','features','models','results','figures','reports']:
        (Path(c['output']['root'])/d).mkdir(parents=True,exist_ok=True)
    for d in ['data/raw','data/interim','data/processed']:
        Path(d).mkdir(parents=True,exist_ok=True)

def smoke_config(c):
    c=copy.deepcopy(c)
    c['splits'].update(inner_folds=2,bootstrap_repetitions=100)
    c['models'].update(logistic_c=[1.0],rf_depth=[5],rf_trees=20)
    c['uncertainty'].update(calibration_subjects=2,ensemble_members=3,min_accepted_calibration=5)
    c.setdefault('exploratory',{}).update(boosting_trees=10,xgb_depth=[3],lgbm_leaves=[7],mlp_alpha=[0.01],mlp_max_iter=300,four_class_min_subjects=4)
    c['output'].update(root='artifacts/synthetic',feature_format='csv')
    return c

def fingerprint(c):
    return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()

def environment():
    from importlib.metadata import version, PackageNotFoundError
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn',
                 'neurokit2','heartpy','tqdm','pyyaml','pytest','pyarrow']:
        try: packages[name]=version(name)
        except PackageNotFoundError: packages[name]='not installed'
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages}
