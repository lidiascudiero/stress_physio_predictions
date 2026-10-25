"""Separate, gated exploratory workflow after all nine confirmatory comparisons."""
from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
from .config import fingerprint,setup
from .fusion import comparison_tables
from .evaluation import run_loso,save_run
from .exploratory_fusion import run_fusion
from .features import extract,save_tables


def require_confirmatory(c):
    root=Path(c['output']['root']);lock=root/'protocol_lock.json'
    if not lock.exists() or json.loads(lock.read_text())['config_sha256']!=fingerprint(c):
        raise ValueError('Complete confirmatory analysis with this frozen configuration first')
    for site in ['wrist','chest','early_fusion']:
        for model in ['majority','logistic_regression','random_forest']:
            p=root/'results'/f'confirmatory_{site}_{model}.json'
            pred=p.with_name(p.stem+'_predictions.csv')
            if not p.exists() or not pred.exists():raise ValueError('Nine confirmatory runs and predictions must exist before exploration')
            result=json.loads(p.read_text())
            if result.get('config_sha256')!=fingerprint(c):raise ValueError('Confirmatory provenance does not match configuration')
            if result['classes']!=[1,2,3] or len(result['folds'])!=result['n_subjects']:
                raise ValueError('Incomplete confirmatory LOSO results')


def four_class_eligibility(frame,c):
    e=c.get('exploratory',{});counts=frame.groupby(['subject_id','label']).size().unstack(fill_value=0).reindex(columns=[1,2,3,4],fill_value=0)
    minimum=e.get('four_class_min_windows_per_subject_class',3)
    bad=counts.index[(counts<minimum).any(axis=1)].tolist()
    enough=len(counts)>=e.get('four_class_min_subjects',10)
    return {'eligible':bool(enough and not bad),'n_subjects':len(counts),'insufficient_subjects':list(map(int,bad)),
            'minimum_subjects':e.get('four_class_min_subjects',10),'minimum_windows_per_subject_class':minimum,
            'counts':{str(s):{str(k):int(v) for k,v in row.items()} for s,row in counts.iterrows()}}


def run(tables,c,records=None,synthetic=False):
    require_confirmatory(c)
    root=Path(c['output']['root'])/'exploratory';root.mkdir(parents=True,exist_ok=True)
    ec=copy.deepcopy(c);ec['output']['root']=str(root);setup(ec)
    status={};frame=comparison_tables(tables,True)['early_fusion']
    def record(name,state,reason=None):
        status[name]={'status':state,'reason':reason,'analysis_role':'exploratory'}
        (root/'results/status.json').write_text(json.dumps(status,indent=2))
    def execute(name,fn):
        try:fn();record(name,'completed')
        except (ImportError,FileNotFoundError,ValueError) as ex:
            record(name,'not_executable',f'{type(ex).__name__}: {ex}')
    e=c.get('exploratory',{})
    for model in e.get('advanced_models',['xgboost','lightgbm','mlp']):
        def fit_advanced(model=model):
            # Each advanced model is compared at all three sites; predictions remain exploratory.
            for site,f in comparison_tables(tables,True).items():
                result,pred=run_loso(f,ec,model,site)
                result['analysis_role']='exploratory'
                save_run(result,pred,ec,f'advanced_{site}_{model}')
        execute(model,fit_advanced)
    for method in ['late_fusion','modality_dropout']:
        def fit_fusion(method=method):
            for result,pred in run_fusion(frame,ec,method):save_run(result,pred,ec,f'{method}_{result["scenario"]}')
        execute(method,fit_fusion)
    def four_class():
        nonlocal records
        if records is None:
            if synthetic:
                from .synthetic import make_records
                records=make_records(seed=c['splits']['seed'],include_meditation=True)
            else:
                from .workflow import load_records
                records=load_records(c)
        ft,qc=extract(records,c,classes=(1,2,3,4))
        all_frame=comparison_tables(ft,True)['early_fusion']
        eligibility=four_class_eligibility(all_frame,c)
        # Every observed original subject must remain represented; no silent favorable subset.
        observed=set(r['subject_id'] for r in records)
        missing=observed-set(all_frame.subject_id)
        if missing:eligibility.update(eligible=False,absent_subjects=sorted(missing))
        (root/'qc/four_class_eligibility.json').write_text(json.dumps(eligibility,indent=2))
        qc.to_csv(root/'qc/four_class_signal_qc.csv',index=False)
        if not eligibility['eligible']:raise ValueError('Insufficient four-class data; inspect qc/four_class_eligibility.json')
        fc=copy.deepcopy(ec);fc['output']['root']=str(root/'four_class');setup(fc);save_tables(ft,fc)
        for site,f in comparison_tables(ft,True).items():
            for model in ['majority','logistic_regression','random_forest']:
                result,pred=run_loso(f,fc,model,site,'four_class',(1,2,3,4))
                result['analysis_role']='exploratory';save_run(result,pred,fc,f'four_class_{site}_{model}')
    execute('four_class',four_class)
    def conformal():
        from .workflow import uncertainty_stage
        cc=copy.deepcopy(ec);cc['output']['root']=str(root/'conformal');cc['uncertainty']['conformal_exploratory']=True;setup(cc)
        uncertainty_stage(tables,cc)
    execute('conformal',conformal)
    def reports():
        from .self_reports import analyze
        if synthetic:
            from .synthetic import make_questionnaires
            questionnaire_root=make_questionnaires(root/'qc/synthetic_questionnaires',seed=c['splits']['seed'])
        else:questionnaire_root=c['dataset']['raw_dir']
        raw,summary,correlations=analyze(questionnaire_root,frame)
        raw.to_csv(root/'results/self_report_episode_scores.csv',index=False)
        summary.to_csv(root/'results/self_report_condition_summary.csv',index=False)
        correlations.to_csv(root/'results/self_report_feature_correlations.csv',index=False)
        (root/'results/self_report_design.json').write_text(json.dumps({'unit':'subject and experimental episode',
            'no_window_label_broadcast':True,'clinical_thresholds':False,
            'sample_note':'only questionnaires actually available; not assumed complete for all subjects',
            'feature_correlations':'one subject-level median per condition and feature; descriptive rho only'},indent=2))
    execute('self_reports',reports)
    write_report(root,status,synthetic)
    return status


def write_report(root,status,synthetic=False):
    root=Path(root)
    lines=['# Exploratory analyses','',
           '**SYNTHETIC EXECUTION ONLY: no real WESAD performance claim.**' if synthetic else 'Exploratory WESAD analyses; separate from confirmatory results.',
           '', '| Analysis | Status | Reason |','|---|---|---|']
    lines += [f'| {name} | {s["status"]} | {s["reason"] or "See dedicated outputs"} |' for name,s in status.items()]
    lines += ['', '## Exploratory classification results','',
              '| Task | Site | Model | Subject mean BA | PR AUC trapezoidal | Average precision |',
              '|---|---|---|---:|---:|---:|']
    paths=sorted((root/'results').glob('*.json'))+sorted((root/'four_class/results').glob('*.json'))
    for path in paths:
        value=json.loads(path.read_text())
        if not isinstance(value,dict) or 'primary' not in value:continue
        lines.append(f"| {value['task']} | {value['site']} | {value['model']} | {value['primary']['mean']:.3f} | {value['macro_pr_auc_trapezoidal_ovr']['mean']:.3f} | {value['macro_average_precision_ovr']['mean']:.3f} |")
    lines += ['', '## Interpretation',
              'All intervals and per-subject predictions are saved in the corresponding JSON/CSV files. No model is promoted retrospectively to the confirmatory result.',
              'Conformal coverage/set size are under conformal/results; questionnaire outputs remain episode-level under results/self_report_*. No individual guarantee or clinical threshold is applied.']
    (root/'reports/exploratory_report.md').write_text('\n'.join(lines)+'\n')
