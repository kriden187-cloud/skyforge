"""Размер кеша и его чистка. data/work/ — промежуточное, удаляется свободно;
data/raw/ — только по явной команде с подтверждением."""

from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path

from skyforge.paths import Paths


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def listing(paths: Paths) -> list[tuple[str, int]]:
    rows = []
    if paths.raw.exists():
        for mission in sorted(d for d in paths.raw.iterdir() if d.is_dir() and d.name != "manifests"):
            for obs in sorted(d for d in mission.iterdir() if d.is_dir()):
                rows.append((paths.rel(obs), dir_size(obs)))
    if paths.work_root.exists():
        for t in sorted(d for d in paths.work_root.iterdir() if d.is_dir()):
            for b in sorted(d for d in t.iterdir() if d.is_dir()):
                rows.append((paths.rel(b), dir_size(b)))
    return rows


def _force_remove(func, path, _exc):  # сырые файлы помечены «только чтение»
    os.chmod(path, stat.S_IWRITE)
    func(path)


def remove_tree(p: Path) -> int:
    size = dir_size(p)
    if p.exists():
        shutil.rmtree(p, onexc=_force_remove)
    return size
