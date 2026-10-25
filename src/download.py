"""Official archive acquisition; does not use mirrors or assume ucimlrepo has arrays."""
from pathlib import Path
import hashlib, urllib.request, urllib.parse, zipfile

OFFICIAL_HOSTS={'uni-siegen.sciebo.de','www.eti.uni-siegen.de',
                'ubicomp.eti.uni-siegen.de','ubi29.informatik.uni-siegen.de',
                'archive.ics.uci.edu'}

def download_archive(url, destination, expected_sha256=None):
    if urllib.parse.urlparse(url).scheme!='https' or urllib.parse.urlparse(url).hostname not in OFFICIAL_HOSTS:
        raise ValueError('Use an HTTPS archive URL linked by the official WESAD site')
    p=Path(destination); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.part')
    try:
        with urllib.request.urlopen(url,timeout=120) as response, tmp.open('wb') as out:
            while chunk:=response.read(1024*1024): out.write(chunk)
        with tmp.open('rb') as archive_stream:
            digest=hashlib.file_digest(archive_stream,'sha256').hexdigest()
        if expected_sha256 and digest.lower()!=expected_sha256.lower():
            raise ValueError('Archive checksum mismatch')
        if not zipfile.is_zipfile(tmp): raise ValueError('Response is not a ZIP archive')
        tmp.replace(p)
        p.with_suffix(p.suffix+'.sha256').write_text(digest+'  '+p.name+'\n')
        return digest
    finally:
        if tmp.exists(): tmp.unlink()

def extract_archive(archive, destination):
    root=Path(destination).resolve();root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            target=(root/member.filename).resolve()
            if not target.is_relative_to(root): raise ValueError('Unsafe ZIP member')
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Symlinks are not accepted')
        z.extractall(root)
