"""Таблица фильтров и чёрно-белое превью каждого — первая проверка, что скачалось то, что нужно."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from rich.table import Table

from skyforge.config import Config
from skyforge.fetch import require_products
from skyforge.filters import wavelength_um
from skyforge.fitsio import load_sci, read_info
from skyforge.log import console
from skyforge.paths import Paths
from skyforge.preview import (PREVIEW_A, PREVIEW_BLACK, PREVIEW_VERSION, PREVIEW_WHITE, cache_previews, contact_sheet,
                              quick_norm, save_png, shrink)


def inspect(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    """Таблица и превью сырых кадров. Кешируется: при тех же файлах превью копируются из data/work/."""
    prods = require_products(cfg, paths)
    cache = paths.work_root / cfg.slug / "inspect"
    key = {"files": {f: [p.path, p.size] for f, p in prods.items()}, "max_size": cfg.preview.max_size,
           "version": PREVIEW_VERSION}
    meta_path = cache / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key:
            cache_previews(meta["previews"], cache, out_dir)
            console.print(f"inspect: кеш актуален ({paths.rel(cache)})")
            _print_table(cfg, meta["info"]["filters"])
            return {**meta["info"], "cached": True}
    info = _compute(cfg, paths, prods, out_dir)
    previews = [r["preview"] for r in info["filters"]] + [info["sheet"]]
    cache_previews(previews, out_dir, cache)
    meta_path.write_text(json.dumps({"key": key, "info": info, "previews": previews}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return {**info, "cached": False}


def _compute(cfg: Config, paths: Paths, prods: dict, out_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    images, labels = [], []
    for f, prod in prods.items():
        path = paths.abs(prod.path)
        info = read_info(path)
        data, _ = load_sci(path)
        lam = wavelength_um(f, cfg.mission, cfg.instrument)
        img = quick_norm(shrink(data, cfg.preview.max_size), PREVIEW_BLACK, PREVIEW_WHITE, PREVIEW_A)
        save_png(img, out_dir / f"inspect_{f}.png")
        images.append(img)
        labels.append(f"{f} · {lam:.2f} мкм")
        rows.append({
            "filter": f,
            "wavelength_um": round(lam, 3),
            "shape": f"{info.shape[1]}×{info.shape[0]}",
            "pixel_scale_arcsec": round(info.pixel_scale_arcsec, 4),
            "exptime_s": round(info.exptime, 1) if info.exptime else None,
            "date_obs": info.date_obs,
            "release": prod.release,
            "bunit": info.bunit,
            "nan_percent": round(float(np.isnan(data).mean()) * 100, 1),
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "source": prod.source,
            "file": prod.path,
            "preview": f"inspect_{f}.png",
        })
        del data

    out_dir.mkdir(parents=True, exist_ok=True)
    contact_sheet(images, labels, out_dir / "inspect_sheet.png",
                  title=f"{cfg.target} · {cfg.mission.upper()} {cfg.instrument.upper()} · сырые кадры")
    _print_table(cfg, rows)
    return {"filters": rows, "sheet": "inspect_sheet.png"}


def _print_table(cfg: Config, rows: list[dict[str, Any]]) -> None:
    t = Table(title=f"{cfg.target}: что на диске")
    cols = [("фильтр", "filter"), ("λ, мкм", "wavelength_um"), ("размер", "shape"),
            ('пиксель, "', "pixel_scale_arcsec"), ("экспозиция, с", "exptime_s"), ("снят", "date_obs"),
            ("открыт с", "release"), ("NaN, %", "nan_percent"), ("МБ", "size_mb"), ("откуда", "source")]
    for title, _ in cols:
        t.add_column(title)
    for r in rows:
        t.add_row(*[str(r[k]) if r[k] is not None else "—" for _, k in cols])
    console.print(t)
