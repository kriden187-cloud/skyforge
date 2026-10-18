"""Сетка вариантов обработки на одном листе: два параметра × по три значения.

Статистика (чёрная и белая точки) считается по полному кадру, а кривая и цвет — по
уменьшенной копии, поэтому лист строится за секунды и совпадает с полным прогоном по тону.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from skyforge.clean import load_input, load_star_weight
from skyforge.compose import compose_layers
from skyforge.config import Config, SweepAxis
from skyforge.log import console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink
from skyforge.stretch import stretch_array


def _fmt(v: float) -> str:
    return f"{v:g}"


def sweep(cfg: Config, paths: Paths, out_dir: Path, x: SweepAxis | None = None,
          y: SweepAxis | None = None, crop: tuple[int, int, int] | None = None) -> dict[str, Any]:
    """crop — (x, y, сторона) кропа 100% на сетке текущего bin вместо уменьшенного кадра."""
    # Только --x из командной строки — одна строка вариантов (например, вес одного фильтра).
    one_row = x is not None and y is None
    x = x or cfg.sweep.x
    y = None if one_row else (y or cfg.sweep.y)
    arrays, _ = load_input(cfg, paths)
    tile = max(200, cfg.preview.max_size // max(len(x.values), 1))
    needs_stars = any(c.color.star_saturation < 1 for c in _variants(cfg, x, y))
    weight = load_star_weight(cfg.with_param("color.star_saturation", 0.0), paths) if needs_stars else None
    if crop:
        cx, cy, side = crop
        sl = (slice(max(cy - side // 2, 0), cy + side // 2), slice(max(cx - side // 2, 0), cx + side // 2))
        small = {f: np.asarray(a)[sl] for f, a in arrays.items()}
        weight = weight.crop(sl) if weight is not None else None
    else:
        small = {f: shrink(np.asarray(a), tile) for f, a in arrays.items()}
        factor = math.ceil(max(next(iter(arrays.values())).shape) / tile)
        weight = weight.shrink(factor) if weight is not None else None

    images, labels = [], []
    for yv in (y.values if y else [None]):
        for xv in x.values:
            c = cfg.with_param(x.param, xv)
            label = f"{x.param}={_fmt(xv)}"
            if y is not None:
                c = c.with_param(y.param, yv)
                label += f" · {y.param}={_fmt(yv)}"
            layers = {}
            for f, a in arrays.items():
                layers[f], _ = stretch_array(small[f], c.stretch.for_filter(f), stats_from=a)
            rgb, _ = compose_layers(c, layers, weight)
            images.append(rgb)
            labels.append(label)
            console.print(f"  {label}")
    title = f"{cfg.target}: столбцы — {x.param}" + (f", строки — {y.param}" if y else "")
    if crop:
        title += f" · кроп 100% вокруг ({crop[0]}, {crop[1]})"
    contact_sheet(images, labels, out_dir / "sweep.png", cols=len(x.values), title=title,
                  interpolation="nearest" if crop else "lanczos")
    return {"x": x.model_dump(), "y": y.model_dump() if y else None, "previews": ["sweep.png"]}


def _variants(cfg: Config, x: SweepAxis, y: SweepAxis | None) -> list[Config]:
    out = []
    for yv in (y.values if y else [None]):
        for xv in x.values:
            c = cfg.with_param(x.param, xv)
            out.append(c.with_param(y.param, yv) if y is not None else c)
    return out
