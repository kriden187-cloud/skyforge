"""Запуск этапов с записью в run.json: по одному (out/<slug>/current/) или целиком (out/<slug>/<дата-время>/)."""

from __future__ import annotations

import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from skyforge.align import align
from skyforge.clean import clean
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
    "build": ("l2_destripe.png, l2_overlaps.png, l2_stars.png", "полосы ушли в кадре детектора, уровни в перекрытиях сведены, звёзды как в i2d"),
    "align": ("align_check.png, align_zoom.png", "звёзды на границах клеток не смещены, нет цветных ореолов"),
    "clean": ("background_model.png, star_cores.png", "стыки плиток ушли в модель фона, ядро звезды белое"),
    "stretch": ("stretch_hist.png, stretch_sheet.png", "пик фона у нуля, детали туманности не выжжены"),
    "compose": ("compose.png", "фон нейтральный, структура как в пресс-релизе"),
    "finish": ("finish_zoom.png", "нет «пластилина» от денойза и ореолов от резкости"),
    "sweep": ("sweep.png", "выбери лучшую клетку и перенеси её числа в YAML"),
    "baseline": ("baseline_diff.png, crops_compare.png", "разность с принятым — только там, где её ждали; фактура та же"),
}


def fetch_stage(cfg: Config, paths: Paths, offline: bool = False, max_gb: float | None = None) -> dict[str, Any]:
    res = fetch(cfg, paths, offline=offline, max_gb=max_gb)
    info: dict[str, Any] = {"products": {f: asdict(p) for f, p in res.products.items()}, "missing": res.missing}
    if cfg.source == "level2" and not offline:
        from skyforge.fetch import fetch_level2

        info["level2"] = fetch_level2(cfg, paths, max_gb=max_gb)
    return info


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
                               style=cfg.style, started=datetime.now().isoformat(timespec="seconds"))


def run_all(cfg: Config, paths: Paths, *, offline: bool = False, force_align: bool = False,
            out_dir: Path | None = None) -> Path:
    out_dir = out_dir or paths.new_run(cfg.slug, cfg.style)
    prepare_out(cfg, out_dir)
    copy_reference(cfg, paths, out_dir)
    t0 = time.perf_counter()
    run_stage("fetch", lambda: fetch_stage(cfg, paths, offline=offline), out_dir)
    if cfg.source == "level2":
        from skyforge.level2 import build

        run_stage("build", lambda: build(cfg, paths, out_dir), out_dir)
    run_stage("inspect", lambda: inspect(cfg, paths, out_dir), out_dir)
    run_stage("align", lambda: align(cfg, paths, out_dir, force=force_align), out_dir)
    run_stage("clean", lambda: clean(cfg, paths, out_dir), out_dir)
    run_stage("stretch", lambda: stretch(cfg, paths, out_dir), out_dir)
    run_stage("compose", lambda: compose(cfg, paths, out_dir), out_dir)
    fin = run_stage("finish", lambda: finish(cfg, paths, out_dir), out_dir)
    from skyforge.baseline import compare

    run_stage("baseline", lambda: compare(cfg, paths, out_dir, out_dir / fin["files"]["png"]), out_dir)
    StageLog(out_dir).set_meta(total_seconds=round(time.perf_counter() - t0, 1))
    report = build_report(out_dir)
    console.rule()
    console.print(f"[bold green]Готово.[/] Отчёт: {report}")
    return out_dir


def copy_reference(cfg: Config, paths: Paths, out_dir: Path) -> str | None:
    """Референс (если он есть у объекта) — обязательная клетка отчёта рядом с итогом."""
    if not cfg.reference_image:
        return None
    src = paths.abs(cfg.reference_image)
    if not src.exists():
        console.print(f"[yellow]Референс {cfg.reference_image} не найден — отчёт будет без него[/]")
        return None
    name = "reference" + src.suffix.lower()
    shutil.copy2(src, out_dir / name)
    return name


def _stack_images(parts: list[Path], dst: Path) -> None:
    """Превью стилей одно под другим (например, фон по каналам у measured и showcase)."""
    from PIL import Image

    ims = [Image.open(p).convert("RGB") for p in parts if p.exists()]
    if not ims:
        return
    w = max(i.width for i in ims)
    sheet = Image.new("RGB", (w, sum(i.height for i in ims)), (17, 17, 17))
    y = 0
    for im in ims:
        sheet.paste(im, (0, y))
        y += im.height
    sheet.save(dst)


def run_styles(cfgs: list[Config], paths: Paths, diffs: dict[str, list[str]], *, offline: bool = False,
               force_align: bool = False) -> Path:
    """Все стили одного объекта подряд в одну папку: общие этапы считаются один раз и берутся из кеша."""
    from skyforge.report import build_comparison

    top = paths.new_run(cfgs[0].slug, "styles")
    top.mkdir(parents=True, exist_ok=True)
    for cfg in cfgs:
        console.rule(f"[bold magenta]стиль {cfg.style}")
        run_all(cfg, paths, offline=offline, force_align=force_align, out_dir=top / cfg.style)
    ref_name = copy_reference(cfgs[0], paths, top)
    _stack_images([top / c.style / "background_check.png" for c in cfgs], top / "background_check.png")
    report = build_comparison(top, [c.style for c in cfgs], diffs, ref_name)
    console.rule()
    console.print(f"[bold green]Готово.[/] Сравнение стилей: {report}")
    return top
