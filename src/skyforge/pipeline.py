"""Запуск этапов с записью в run.json: по одному (out/<slug>/current/) или целиком (out/<slug>/<дата-время>/)."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from skyforge.align import align
from skyforge.compose import compose
from skyforge.config import Config
from skyforge.fetch import fetch
from skyforge.finish import finish
from skyforge.inspect import inspect
from skyforge.log import console, timed
from skyforge.paths import Paths, StageLog
from skyforge.report import build_report
from skyforge.stretch import stretch

# Что смотреть после каждого этапа — печатается в консоль.
LOOK = {
    "inspect": ("inspect_sheet.png", "на каждом превью узнаётся объект"),
    "align": ("align_check.png, align_zoom.png", "звёзды на границах клеток не смещены, нет цветных ореолов"),
    "stretch": ("stretch_hist.png, stretch_sheet.png", "пик фона у нуля, детали туманности не выжжены"),
    "compose": ("compose.png", "фон нейтральный, структура как в пресс-релизе"),
    "finish": ("finish_zoom.png", "нет «пластилина» от денойза и ореолов от резкости"),
    "sweep": ("sweep.png", "выбери лучшую клетку и перенеси её числа в YAML"),
}


def fetch_stage(cfg: Config, paths: Paths, offline: bool = False, max_gb: float | None = None) -> dict[str, Any]:
    res = fetch(cfg, paths, offline=offline, max_gb=max_gb)
    return {"products": {f: asdict(p) for f, p in res.products.items()}, "missing": res.missing}


def run_stage(name: str, fn: Callable[[], dict[str, Any]], out_dir: Path) -> dict[str, Any]:
    console.rule(f"[bold]{name}")
    info: dict[str, Any] = {}
    with timed(info):
        info.update(fn())
    StageLog(out_dir).update(name, info)
    if name in LOOK:
        files, what = LOOK[name]
        console.print(f"[cyan]Посмотри:[/] {out_dir / files.split(',')[0]} — {what}")
    return info


def prepare_out(cfg: Config, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    StageLog(out_dir).set_meta(target=cfg.target, mission=cfg.mission, instrument=cfg.instrument, bin=cfg.bin,
                               started=datetime.now().isoformat(timespec="seconds"))


def run_all(cfg: Config, paths: Paths, *, offline: bool = False, force_align: bool = False,
            out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.new_run(cfg.slug)
    prepare_out(cfg, out_dir)
    run_stage("fetch", lambda: fetch_stage(cfg, paths, offline=offline), out_dir)
    run_stage("inspect", lambda: inspect(cfg, paths, out_dir), out_dir)
    run_stage("align", lambda: align(cfg, paths, out_dir, force=force_align), out_dir)
    run_stage("stretch", lambda: stretch(cfg, paths, out_dir), out_dir)
    run_stage("compose", lambda: compose(cfg, paths, out_dir), out_dir)
    run_stage("finish", lambda: finish(cfg, paths, out_dir), out_dir)
    report = build_report(out_dir)
    console.rule()
    console.print(f"[bold green]Готово.[/] Отчёт: {report}")
    return out_dir
