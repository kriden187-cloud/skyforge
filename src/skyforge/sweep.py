"""Сетка вариантов обработки на одном листе: два параметра × по три значения.

Статистика (чёрная и белая точки) считается по полному кадру, а кривая и цвет — по
уменьшенной копии, поэтому лист строится за секунды и совпадает с полным прогоном по тону.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from skyforge.align import load_aligned
from skyforge.compose import compose_layers
from skyforge.config import Config, SweepAxis
from skyforge.log import console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink
from skyforge.stretch import stretch_array


def _fmt(v: float) -> str:
    return f"{v:g}"


def sweep(cfg: Config, paths: Paths, out_dir: Path, x: SweepAxis | None = None,
          y: SweepAxis | None = None) -> dict[str, Any]:
    x = x or cfg.sweep.x
    y = y or cfg.sweep.y
    arrays, _, _ = load_aligned(cfg, paths)
    tile = max(200, cfg.preview.max_size // max(len(x.values), 1))
    small = {f: shrink(np.asarray(a), tile) for f, a in arrays.items()}

    images, labels = [], []
    for yv in y.values:
        for xv in x.values:
            c = cfg.with_param(x.param, xv).with_param(y.param, yv)
            layers = {}
            for f, a in arrays.items():
                layers[f], _ = stretch_array(small[f], c.stretch.for_filter(f), stats_from=a)
            rgb, _ = compose_layers(c, layers)
            images.append(rgb)
            labels.append(f"{x.param}={_fmt(xv)} · {y.param}={_fmt(yv)}")
            console.print(f"  {labels[-1]}")
    contact_sheet(images, labels, out_dir / "sweep.png", cols=len(x.values),
                  title=f"{cfg.target}: столбцы — {x.param}, строки — {y.param}")
    return {"x": x.model_dump(), "y": y.model_dump(), "previews": ["sweep.png"]}
