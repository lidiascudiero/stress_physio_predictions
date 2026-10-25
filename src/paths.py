"""Portable project discovery: no user-specific or machine-specific path literals."""
from pathlib import Path

def project_root(start=None):
    if start is not None:
        here=Path(start).resolve()
        for p in (here,*here.parents):
            if (p/'configs/config.yaml').is_file() and (p/'src').is_dir():return p
    # The module anchors the root even when launched from an unrelated working directory.
    return Path(__file__).resolve().parents[1]

def project_path(relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts:raise ValueError('Use a path relative to the project root')
    return project_root()/p
