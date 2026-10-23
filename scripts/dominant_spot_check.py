"""Фаза 8: сработало бы правило stretch.exclude_dominant_spot на объекте — доля крупнейшего связного пятна среди пикселей
выборки туманности выше белой точки, по фильтрам (нужны кеши align/clean прогона с этим bin).

Запуск: uv run python scripts/dominant_spot_check.py <конфиг> <стиль> <bin> [порог доли]
"""

from __future__ import annotations

import pathlib
import sys

from skyforge.align import read_aligned_meta, stats_field
from skyforge.clean import load_input
from skyforge.config import Config, DominantSpotConfig
from skyforge.paths import Paths
from skyforge.stretch import dominant_spot, filter_masks, load_masks


def main() -> None:
    cfg_path, style, bin_ = sys.argv[1], sys.argv[2], int(sys.argv[3])
    share = float(sys.argv[4]) if len(sys.argv) > 4 else 0.9
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load(cfg_path, style).model_copy(update={"bin": bin_})
    arrays, _ = load_input(c, P)
    masks = load_masks(c, P)
    if masks is None:
        print(f"{cfg_path} {style}: масок нет (anchor не masks) — правило не применяется")
        return
    field = stats_field(c, P, list(arrays.values()))
    scale = read_aligned_meta(c, P).get("pixel_scale_arcsec")
    base = c.model_copy(update={"stretch": c.stretch.model_copy(update={"exclude_dominant_spot": None})})
    for f, a in arrays.items():
        mf, _ = filter_masks(base, a, masks, scale, field)   # выборка как в принятом прогоне
        m, info = dominant_spot(a, mf[1], c.stretch.for_filter(f).white, DominantSpotConfig(share=share), scale, field)
        print(f"{c.target} {style} bin{bin_} {f}: доля крупнейшего пятна {info.get('share')}, областей {info.get('regions')}"
              + (f" — СРАБОТАЛО БЫ: центр {info['center']}, отступ {info['radius_arcsec']}″" if m is not None else ""))


if __name__ == "__main__":
    main()
