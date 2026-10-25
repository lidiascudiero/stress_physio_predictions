"""Real-data audit before synchronized classification is possible."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
from .config import load,setup
from .paths import project_root
from .raw_io import audit_raw_subject,load_e4,E4_RATES

def run(subject_dir,c,subject='S2'):
    result=audit_raw_subject(subject_dir,c,subject)
    out=Path(c['output']['root']);(out/'qc').mkdir(parents=True,exist_ok=True);(out/'reports').mkdir(parents=True,exist_ok=True)
    (out/'qc'/f'{subject}_raw_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    pd.DataFrame(result['schedule']).to_csv(out/'qc'/f'{subject}_schedule.csv',index=False)
    rows=[]
    for name,stats in result['chest']['channels'].items():
        rows.append({'site':'chest','channel':name,'sampling_hz':700,'duration_seconds':result['chest']['duration_seconds'],**stats})
    for name,stats in result['wrist'].items():rows.append({'site':'wrist',**stats})
    pd.DataFrame(rows).to_csv(out/'qc'/f'{subject}_raw_channels.csv',index=False)
    lines=[f'# Audit reale {subject}','','Dati grezzi caricati e letti. Nessun risultato di classificazione.',
           '',f"RespiBAN: {result['chest']['n_samples']} campioni a 700 Hz; durata {result['chest']['duration_seconds']:.3f} s.",
           '', '| E4 | Hz | Campioni | Durata s | Min | Max | Fuori range |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,a in result['wrist'].items():
        lines.append(f"| {name} | {a['sampling_hz']:g} | {a['n_rows']} | {a['duration_seconds']:.3f} | {a['minimum']:.3f} | {a['maximum']:.3f} | {a['out_of_range']} |")
    lines+=['','## Protocollo','Tempi minutes.seconds convertiti in secondi dal principio della registrazione RespiBAN; nessun trasferimento delle etichette su E4 non sincronizzato.']
    for ep in result['schedule']:
        lines.append(f"- {ep['condition']}: {ep['start_seconds']}–{ep['end_seconds']} s; label {ep['label']}; ignorata={ep['ignored']}")
    lines+=['','## Qualità nota',result['quality_note'],
            '', '## Stato delle analisi',result['synchronization_status'],result['fusion_status'],result['loso_status'],
            '', '## Evidenza e limiti','Il README originale fornito conferma codici delle etichette, unità, formule, righe iniziali E4 e formato dei tempi. HR.csv e IBI.csv sono derivati esclusi come richiesto dal README. Gli header temporali non dimostrano sincronizzazione dei dispositivi. Nessun offset stimato o applicato.',
            '', '## Licenza dichiarata nei file','Il README del pacchetto ricevuto dichiara CC BY 4.0; questa dichiarazione è registrata senza equipararla automaticamente ai termini della distribuzione originale. Il pacchetto di codice non redistribuisce i segnali.',
            '', '## Fonte','wesad_readme.pdf, sezioni II.1–II.3 e III.1; S2_readme.txt; info.txt; segnali S2 allegati.']
    (out/'reports'/f'{subject}_raw_audit.md').write_text('\n'.join(lines)+'\n')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--subject',default='S2');p.add_argument('--config',default='configs/config.yaml');args=p.parse_args()
    import os
    os.chdir(project_root());c=load(args.config);setup(c)
    # Separate real raw audit artifacts from confirmed-model results.
    c['output']['root']='artifacts/real_raw_audit'
    result=run(Path(c['dataset']['raw_dir'])/'WESAD'/args.subject,c,args.subject)
    print(json.dumps({'subject':args.subject,'chest_duration_seconds':result['chest']['duration_seconds'],
                      'wrist_duration_seconds':{k:v['duration_seconds'] for k,v in result['wrist'].items()},
                      'synchronization':result['synchronization_status'],'fusion':result['fusion_status'],
                      'output':c['output']['root']},indent=2))

if __name__=='__main__':main()
