"""Deterministic source provenance; legacy locks are never silently upgraded."""
from pathlib import Path
import hashlib
import json
from .config import fingerprint

def code_manifest():
    root = Path(__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob('*.py'))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return digest, files

def make_lock(c):
    digest, files = code_manifest()
    return dict(config=c, config_sha256=fingerprint(c), code_sha256=digest, source_files_sha256=files)

def verify_lock(path, c=None):
    lock = json.loads(Path(path).read_text(encoding='utf-8'))
    if fingerprint(lock['config']) != lock['config_sha256']:
        raise ValueError('Saved configuration hash mismatch')
    if c is not None and fingerprint(c) != lock['config_sha256']:
        raise ValueError('Current configuration differs from protocol')
    if 'code_sha256' not in lock:
        raise ValueError('Legacy protocol without source fingerprint: rerun extraction in a new output directory')
    digest, files = code_manifest()
    if digest != lock['code_sha256'] or files != lock['source_files_sha256']:
        raise ValueError('Source fingerprint mismatch: use the frozen source or a new output directory')
    return lock
