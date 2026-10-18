"""Таблица фильтров и чёрно-белое превью каждого — первая проверка, что скачалось то, что нужно."""

from __future__ import annotations

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
from skyforge.preview import contact_sheet, quick_norm, save_png, shrink


def inspect(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    prods = require_products(cfg, paths)
    rows: list[dict[str, Any]] = []
    images, labels = [], []
    for f, prod in prods.items():
        path = paths.abs(prod.path)
        info = read_info(path)
        data, _ = load_sci(path)
        lam = wavelength_um(f, cfg.mission, cfg.instrument)
        s = cfg.stretch.for_filter(f)
        img = quick_norm(shrink(data, cfg.preview.max_size), s.black, s.white, s.a)
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

    contact_sheet(images, labels, out_dir / "inspect_sheet.png",
                  title=f"{cfg.target} · {cfg.mission.upper()} {cfg.instrument.upper()} · сырые кадры")

    t = Table(title=f"{cfg.target}: что на диске")
    cols = [("фильтр", "filter"), ("λ, мкм", "wavelength_um"), ("размер", "shape"),
            ('пиксель, "', "pixel_scale_arcsec"), ("экспозиция, с", "exptime_s"), ("снят", "date_obs"),
            ("открыт с", "release"), ("NaN, %", "nan_percent"), ("МБ", "size_mb"), ("откуда", "source")]
    for title, _ in cols:
        t.add_column(title)
    for r in rows:
        t.add_row(*[str(r[k]) if r[k] is not None else "—" for _, k in cols])
    console.print(t)
    return {"filters": rows, "sheet": "inspect_sheet.png"}
