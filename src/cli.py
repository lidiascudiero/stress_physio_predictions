import argparse
from .config import load, setup, smoke_config
from . import workflow as w

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/config.yaml')
    p.add_argument('--synthetic',action='store_true')
    p.add_argument('--stage',choices=['all','audit','features','confirmatory','secondary','uncertainty','exploratory','report'],default='all')
    args=p.parse_args()
    import os
    from .paths import project_root
    os.chdir(project_root())
    c=load(args.config)
    if args.synthetic:c=smoke_config(c)
    setup(c)
    if args.stage in ['all','audit','features']:
        if args.synthetic:
            from .synthetic import make_records
            records=make_records(seed=c['splits']['seed'])
        else:records=w.load_records(c)
        if args.stage in ['all','audit']:w.audit_stage(records,c,args.synthetic)
        if args.stage in ['all','features']:tables,_=w.feature_stage(records,c)
    else:tables=None
    if args.stage in ['all','confirmatory','secondary','uncertainty','exploratory']:
        if tables is None:tables=w.load_tables(c)
        for stage in ['confirmatory','secondary','uncertainty']:
            if args.stage in ['all',stage]:
                print(f'Running {stage} ({"synthetic" if args.synthetic else "real WESAD"})',flush=True)
                getattr(w,stage+'_stage')(tables,c)
    if args.stage=='exploratory' or (args.stage=='all' and c.get('exploratory',{}).get('enabled',False)):
        from .exploratory import run
        run(tables,c,synthetic=args.synthetic)
    if args.stage in ['all','report']:w.report_stage(c,args.synthetic)
    print('Completed. Outputs: '+c['output']['root'])

if __name__=='__main__':main()
