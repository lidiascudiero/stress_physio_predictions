"""Notebook and CLI stages share this implementation."""
from pathlib import Path
import copy, hashlib, json
import numpy as np
import pandas as pd
from .config import setup, fingerprint, environment
from .wesad_io import discover, load_subject, read_questionnaire, audit
from .signal_qc import alignment
from .features import extract, save_tables
from .fusion import comparison_tables
from .evaluation import run_loso,save_run,paired_difference
from .uncertainty import run_uncertainty
from . import plotting
from .provenance import make_lock, verify_lock

SOURCE_CITATIONS=[
 'Schmidt P, Reiss A, Duerichen R, Marberger C, Van Laerhoven K. Introducing WESAD. ICMI 2018, 400–408. doi:10.1145/3242969.3242985',
 'Schmidt P, Reiss A. WESAD dataset. UCI Machine Learning Repository. doi:10.24432/C57K5T']

def root(c):return Path(c['output']['root'])
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def load_records(c):
    return [load_subject(p,c['dataset']['trusted_pickle']) for p in discover(c['dataset']['raw_dir']).values()]

def audit_stage(records,c,synthetic=False):
    setup(c);r=root(c)
    table=audit(records,c);table.to_csv(r/'qc/dataset_audit.csv',index=False)
    alignment_table=pd.concat([alignment(x,c['qc']['max_duration_error_seconds']) for x in records])
    alignment_table.to_csv(r/'qc/alignment.csv',index=False)
    questionnaires={}
    for record in records:
        p=Path(record.get('source','synthetic'))
        if p.exists():
            for q in p.parent.glob(f'S{record["subject_id"]}_quest.*'):
                questionnaires[str(record['subject_id'])]=read_questionnaire(q)
    write_json(r/'qc/questionnaires.json',questionnaires)
    actual=set(x['subject_id'] for x in records)
    write_json(r/'qc/audit_status.json',{'synthetic':synthetic,'observed_subjects':sorted(actual),
               'missing_expected_subjects':sorted(set(c['dataset']['expected_subjects'])-actual),
               'label_mapping_status':'configured from supplied specification; original archive README must be checked manually',
               'questionnaire_timing_status':'raw rows retained, units not automatically assumed',
               'alignment_status':'duration consistency only; supplied synchronization not independently reconstructed'})
    return table

def feature_stage(records,c):
    lock=root(c)/'protocol_lock.json'
    if lock.exists(): verify_lock(lock,c)
    if lock.exists() and json.loads(lock.read_text())['config_sha256']!=fingerprint(c):
        raise ValueError('Configuration differs from frozen protocol; use a new output directory')
    tables,qc=extract(records,c)
    qc.to_csv(root(c)/'qc/signal_qc.csv',index=False)
    summary=[]
    for site,t in tables.items():
        for (sid,label),g in t.groupby(['subject_id','label']):
            summary.append({'site':site,'subject_id':int(sid),'label':int(label),
                            'candidate_windows':len(g),'valid_windows':int(g.valid.sum()),
                            'excluded_fraction':float(1-g.valid.mean())})
    pd.DataFrame(summary).to_csv(root(c)/'qc/window_coverage.csv',index=False)
    plotting.qc_plot(qc,root(c)/'figures/qc_heatmap.png')
    plotting.signal_example(records[0],root(c)/'figures/signal_psd_example.png')
    save_tables(tables,c)
    # Freeze the configuration before fitting; never silently overwrite a different protocol.
    lock=root(c)/'protocol_lock.json';sig=fingerprint(c)
    if lock.exists() and json.loads(lock.read_text())['config_sha256']!=sig:
        raise ValueError('Configuration differs from frozen protocol; use a new output directory')
    write_json(lock,make_lock(c))
    return tables,qc

def load_tables(c):
    verify_lock(root(c)/'protocol_lock.json',c)
    fmt=c['output']['feature_format'];tables={}
    for site in ['chest','wrist']:
        p=root(c)/'features'/f'{site}_features.{fmt}'
        t=pd.read_parquet(p) if fmt=='parquet' else pd.read_csv(p)
        if t.valid.dtype!=bool:t['valid']=t.valid.astype(str).str.lower().eq('true')
        tables[site]=t
    return tables

def confirmatory_stage(tables,c):
    lock=verify_lock(root(c)/'protocol_lock.json',c)
    if lock['config_sha256']!=fingerprint(c):raise ValueError('Protocol changed after feature extraction')
    results=[]
    for site,frame in comparison_tables(tables,paired_only=True).items():
        if set(frame.label)!={1,2,3}:raise ValueError('Confirmatory analysis requires exactly labels 1,2,3')
        for model in ['majority','logistic_regression','random_forest']:
            result,pred=run_loso(frame,c,model,site)
            tag=f'confirmatory_{site}_{model}';save_run(result,pred,c,tag);results.append(result)
            plotting.confusion(result['confusion_matrix_sum'],result['classes'],root(c)/'figures'/f'{tag}_confusion.png')
    contrasts=[]
    for model in ['logistic_regression','random_forest']:
        a=next(x for x in results if x['site']=='early_fusion' and x['model']==model)
        b=next(x for x in results if x['site']=='wrist' and x['model']==model)
        contrasts.append({'contrast':'early_fusion_minus_wrist','model':model,**paired_difference(a,b,c)})
    write_json(root(c)/'results/confirmatory_contrasts.json',contrasts)
    relevance=[]
    for meta in (root(c)/'models').glob('three_class_*/*_metadata.json'):
        info=json.loads(meta.read_text())
        for feature,value in info['training_only_relevance'].items():
            relevance.append({'model_group':meta.parent.name,'test_subject':info['test_subject'],
                              'feature':feature,'training_relevance':value})
    if relevance:
        rel=pd.DataFrame(relevance)
        rel.to_csv(root(c)/'results/training_feature_relevance_by_fold.csv',index=False)
        rel.groupby(['model_group','feature']).training_relevance.agg(['mean','std','count']).to_csv(
            root(c)/'results/training_feature_relevance_summary.csv')
    return results

def secondary_stage(tables,c):
    results=[]
    # Paired chest/wrist comparison belongs to the primary common-window analysis;
    # these are task changes and modality-specific availability analyses.
    for task in ['stress_vs_nonstress','baseline_vs_stress','availability_three_class']:
        data=comparison_tables(tables,paired_only=task!='availability_three_class')
        for site,frame in data.items():
            if task=='availability_three_class' and site=='early_fusion':continue
            frame=frame.copy()
            if task=='stress_vs_nonstress':frame['label']=(frame.label==2).astype(int);classes=(0,1)
            elif task=='baseline_vs_stress':frame=frame[frame.label.isin([1,2])];classes=(1,2)
            else:classes=(1,2,3)
            result,pred=run_loso(frame,c,'logistic_regression',site,task,classes)
            result['analysis_role']='secondary'
            save_run(result,pred,c,f'secondary_{task}_{site}');results.append(result)
    return results

def uncertainty_stage(tables,c):
    frame=comparison_tables(tables,True)['early_fusion']
    pred,logs,diagrams,curves=run_uncertainty(frame,c)
    p=root(c)/'results'
    pred.to_csv(p/'uncertainty_predictions.csv',index=False)
    diagrams.to_csv(p/'reliability_by_subject_class.csv',index=False)
    curves.to_csv(p/'risk_acceptance.csv',index=False)
    write_json(p/'uncertainty_splits.json',logs)
    coverage=[]
    for (sid,label),g in pred.groupby(['subject_id','label']):
        a=g[g.accepted]
        entry={'subject_id':int(sid),'label':int(label),'acceptance':float(g.accepted.mean()),
               'accepted_accuracy':float((a.prediction_raw==a.label).mean()) if len(a) else None,
               'n_windows':len(g)}
        if 'set_covers' in g:
            entry.update(empirical_set_coverage=float(g.set_covers.mean()),mean_set_size=float(g.set_size.mean()))
        coverage.append(entry)
    write_json(p/'coverage_by_subject_class.json',coverage)
    plotting.reliability_plot(diagrams,root(c)/'figures/reliability.png')
    return pred

def report_stage(c,synthetic=False):
    verify_lock(root(c)/'protocol_lock.json',c)
    p=root(c);items=[]
    for f in sorted((p/'results').glob('*.json')):
        if f.name.startswith(('confirmatory_','secondary_')) and 'contrasts' not in f.name:
            obj=json.loads(f.read_text())
            if 'primary' in obj:items.append(obj)
    audit_status=json.loads((p/'qc/audit_status.json').read_text()) if (p/'qc/audit_status.json').exists() else {}
    exploratory_file=p/'exploratory/results/status.json'
    exploratory_status=json.loads(exploratory_file.read_text()) if exploratory_file.exists() else {'status':'not_run'}
    result={'exploratory_status':exploratory_status,'data_kind':'synthetic_execution_test' if synthetic else 'real_WESAD',
            'config_sha256':fingerprint(c),'environment':environment(),'citations':SOURCE_CITATIONS,
            'audit':audit_status,'analyses':items,
            'limitations':['15 original subjects; laboratory protocol; no external validation',
                           'Experimental condition labels are not continuous personal stress',
                           'Conditional subject bootstrap; overlapping LOSO training sets remain dependent',
                           'QC and peak detection are engineering heuristics, not validated physiological annotations',
                           'No clinically validated or safety-related use',
                           'No guaranteed individual coverage; small calibration sample'],
            'permutation_status':'Permutation test not applicable without a validated exchangeability scheme.'}
    write_json(p/'results/results.json',result)
    lines=['# WESAD experimental-condition classification','',
           '**SYNTHETIC EXECUTION TEST — no WESAD performance claims.**' if synthetic else 'Research classification of laboratory experimental conditions.',
           '', '## Research question','Subject-independent comparison of wrist, chest and combined biosignals.',
           '', '## Protocol','60-second windows, 50% overlap; no condition-crossing windows. Native sampling rates. LOSO with nested subject-grouped tuning. Primary comparisons use common valid windows. Primary metric: equal-weight subject mean balanced accuracy.',
           '', '## Results','', '| Role | Task | Site | Model | Subject mean BA | Conditional 95% CI | PR AUC trapezoidal | Average precision |', '|---|---|---|---|---:|---|---:|---:|']
    for x in items:
        m=x['primary'];lo,hi=m['ci95'];role='secondary' if x.get('analysis_role')=='secondary' else 'confirmatory'
        lines.append(f"| {role} | {x['task']} | {x['site']} | {x['model']} | {m["mean"]:.3f} | [{lo:.3f}, {hi:.3f}] | {x['macro_pr_auc_trapezoidal_ovr']['mean']:.3f} | {x['macro_average_precision_ovr']['mean']:.3f} |")
    lines+=['','## Availability and audit','See qc/window_coverage.csv, qc/audit_status.json and qc/questionnaires.json. Questionnaire timing units and code mapping must be reviewed against the original archive README.',
            '', '## Interpretation','A higher laboratory score does not establish natural stress detection, clinical validity or wrist noninferiority. No wrist sufficiency claim is made without a prespecified margin.',
            '', '## Limitations']+['- '+s for s in result['limitations']]
    lines+=['', '## Feature relevance','Training-only relevance by LOSO fold and descriptive mean/standard deviation are saved in results/training_feature_relevance_by_fold.csv and results/training_feature_relevance_summary.csv. Impurity importance and absolute coefficients are not causal evidence and must not be used for retrospective feature selection.',
            '', '## Statistical testing',result['permutation_status'],
            '', '## Exploratory analyses','Implemented in a separate gated workflow. Status: '+json.dumps(exploratory_status)+'. See exploratory/reports/exploratory_report.md. No exploratory result is inserted in the main results table.',
            '', '## References']+['- '+s for s in SOURCE_CITATIONS]
    (p/'reports/report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
