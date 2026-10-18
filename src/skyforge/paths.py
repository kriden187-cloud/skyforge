"""Где что лежит. Корень — SKYFORGE_HOME или ближайшая папка вверх с configs/ и pyproject.toml."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("SKYFORGE_HOME")
    if env:
        return Path(env).resolve()
    here = (start or Path.cwd()).resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").exists() and (p / "configs").is_dir():
            return p
    return here


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def inbox(self) -> Path:
        return self.root / "data" / "inbox"

    @property
    def work_root(self) -> Path:
        return self.root / "data" / "work"

    @property
    def out_root(self) -> Path:
        return self.root / "out"

    def manifest(self, slug: str, mission: str, instrument: str) -> Path:
        inst = instrument.replace("/", "_")
        return self.raw / "manifests" / f"{slug}_{mission}_{inst}.json"

    def work(self, slug: str, bin_: int) -> Path:
        return self.work_root / slug / f"bin{bin_}"

    def current(self, slug: str) -> Path:
        """Папка для отдельных команд: превью последнего запуска каждого этапа."""
        return self.out_root / slug / "current"

    def new_run(self, slug: str) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        return self.out_root / slug / stamp

    def rel(self, p: Path) -> str:
        try:
            return Path(p).resolve().relative_to(self.root).as_posix()
        except ValueError:
            return Path(p).resolve().as_posix()

    def abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p


class StageLog:
    """run.json в папке вывода: что делал каждый этап, с какими параметрами и сколько длилось."""

    def __init__(self, out_dir: Path):
        self.path = out_dir / "run.json"
        out_dir.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"stages": {}}

    def update(self, stage: str, info: dict[str, Any]) -> None:
        data = self.load()
        data.setdefault("stages", {})[stage] = info
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    def set_meta(self, **kw: Any) -> None:
        data = self.load()
        data.update(kw)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
