"""3j: сетка для выбора владельцем — яркость звёзд поля (finish.stars.scale) × обесцвечивание (color.star_saturation).
Кропы: три звезды поля 80 px ×3 и sky3 100%, NASA первым. Без выводов на листе.

Запуск: uv run python scripts/field_stars_grid_3j.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import load_nebula_sd, load_saturated_stars, load_sky_mask, load_star_weight
from skyforge.compose import base_color, load_stretched, tail_color
from skyforge.config import Config
from skyforge.finish import finish_masks, finish_region
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402

FIELD = [(1966, 1179), (276, 1412), (845, 1758)]      # яркие звёзды поля на пустом небе (stars_sheet_3j.py)
SCALES = (0.5, 0.7, 1.0)


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    sat_now = cfg.color.star_saturation
    SATS = (sat_now, round((sat_now + 1) / 2, 3), 1.0)
    layers, _ = load_stretched(cfg, P)
    field = stats_field(cfg, P, list(layers.values()))
    sd, sky, sm = load_nebula_sd(cfg, P), load_sky_mask(cfg, P), load_star_weight(cfg, P)
    sat_stars = load_saturated_stars(cfg, P)
    b, _ = base_color(cfg, layers, sky, sd, field=field)
    sw_, sd_, origin = finish_masks(cfg.model_copy(update={"finish": cfg.finish.model_copy(
        update={"stars": cfg.finish.stars.model_copy(update={"scale": 0.99})})}), P)
    H = sd.shape[0]
    c3 = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}["sky3"]
    regions = [(f"поле ({x}, {y})", (slice(H - 1 - y - 40, H - 1 - y + 40), slice(x - 40, x + 40)), 3) for x, y in FIELD]
    regions.append(("sky3", display_crop(sd.shape, c3.x, c3.y, c3.size, 1), 1))
    comp0 = tail_color(cfg, b, layers, sm, sd, sat_stars)
    nasa = nasa_on_grid(np.clip(comp0[::-1], 0, 1), str(P.abs(cfg.reference_image)))
    tiles, labels = [], []
    for rname, sl, k in regions:
        tiles.append(np.kron(nasa[sl], np.ones((k, k, 1))))
        labels.append(f"{rname[:14]} NASA")
        for sat in SATS:
            c = cfg.with_param("color.star_saturation", sat)
            comp = tail_color(c, b, layers, sm, sd, sat_stars)
            for sc in SCALES:
                f = c.finish.model_copy(update={"stars": c.finish.stars.model_copy(update={"scale": sc, "shrink": 0.0})})
                im = finish_region(comp, f, sw_, sd_, origin, sl)
                tiles.append(np.kron(np.clip(im, 0, 1), np.ones((k, k, 1))))
                labels.append(f"sat {sat:g} · scale {sc:g}")
    contact_sheet(tiles, labels, out / "18_звёзды_поля_для_выбора.png", cols=1 + len(SATS) * len(SCALES), tile_inches=2.4,
                  interpolation="nearest",
                  title="Звёзды поля: finish.stars.scale (0,5 / 0,7 / 1) × color.star_saturation (как сейчас / вдвое слабее / выкл), shrink 0")


if __name__ == "__main__":
    main()
