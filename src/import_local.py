"""Copy local WESAD files to relative project data folders without modifying originals."""
from pathlib import Path
import argparse,hashlib,json,re,shutil
from .paths import project_root

def normalized_name(name):
    return re.sub(r'\(\d+\)(?=\.[^.]+$)','',name)

def _copy(source,dest):
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():
        with source.open('rb') as a,dest.open('rb') as b:
            equal=hashlib.file_digest(a,'sha256').digest()==hashlib.file_digest(b,'sha256').digest()
        if not equal:raise FileExistsError(f'Different file already exists at {dest}; no overwrite')
    else:shutil.copyfile(source,dest)

def import_subject(source,destination,subject):
    source=Path(source).expanduser();destination=Path(destination)
    if not source.is_dir():raise FileNotFoundError(source)
    if not re.fullmatch(r'S\d+',subject):raise ValueError('Use subject identifiers such as S2')
    base=destination/subject;imported=[]
    wanted={f'{subject}.pkl',f'{subject}_quest.csv',f'{subject}_readme.txt',f'{subject}_respiban.txt',f'{subject}_rs2_readme.txt'}
    for p in source.rglob('*'):
        if not p.is_file():continue
        name=normalized_name(p.name)
        if name in wanted:
            target=base/name
        elif name in {'ACC.csv','BVP.csv','EDA.csv','TEMP.csv','HR.csv','IBI.csv','info.txt','tags.csv'}:
            # Requires subject-specific ancestry; do not mix E4 files from different subjects.
            if not any(x in {subject,f'{subject}_E4_Data'} for x in p.parts):continue
            target=base/f'{subject}_E4_Data'/name
        else:continue
        if p.resolve()==target.resolve():continue
        _copy(p,target);imported.append(str(target))
    if not imported:raise FileNotFoundError('No matching subject files; check --source and --subject')
    return imported

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True,type=Path,help='Existing local WESAD directory')
    parser.add_argument('--subject',default='S2')
    args=parser.parse_args();root=project_root()
    files=import_subject(args.source,root/'data/raw/WESAD',args.subject)
    print('Copied files (originals unchanged):')
    for p in files:print(Path(p).relative_to(root))

if __name__=='__main__':main()
