"""3k, п.2(б): фиксированная маска «где наклон экспозиций не подгоняется» для сборки уровня 2 — туманность clean
с кольцом ореола RING_PX и круг RADIUS_PX вокруг центральной звезды, на нашей сетке (bin 1), с WCS.
Сборка идёт раньше clean, поэтому маска — файл, а не результат clean: от прогона к прогону она одна и та же.
Печатает долю свободного неба сборки, которая останется для подгонки, по фильтрам.

Запуск: uv run python scripts/make_slope_mask_3k.py data/reference/ngc3132_slope_exclusion.fits
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales

from skyforge.align import load_aligned
from skyforge.clean import load_nebula_sd
from skyforge.config import Config
from skyforge.fetch import require_products
from skyforge.level2 import _wcs, reproject_mask, source_mask_grid
from skyforge.paths import Paths

RING_PX, RADIUS_PX = 300, 900
CENTER_IMG = (1134, 877)  # центральная звезда на итоговой картинке (x вправо, y вниз)


def main() -> None:
    dest = pathlib.Path(sys.argv[1])
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    _, wcs, _ = load_aligned(cfg, P)
    sd = load_nebula_sd(cfg, P)
    H, W = sd.shape
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(xx - CENTER_IMG[0], yy - (H - 1 - CENTER_IMG[1]))
    excl = (sd > -RING_PX) | (r < RADIUS_PX)
    hdr = wcs.to_header()
    hdr["COMMENT"] = f"3k: slope exclusion = nebula mask + {RING_PX} px ring, disk {RADIUS_PX} px around central star"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(excl.astype(np.uint8), header=hdr).writeto(dest, overwrite=True)
    print(f"{dest}: исключено {excl.mean() * 100:.1f}% нашей сетки")
    prods = require_products(cfg, P, level3=True)
    for f, p in prods.items():
        with fits.open(P.abs(p.path), memmap=True) as h:
            i2d = np.array(h["SCI"].data, dtype=np.float32)
            if "WHT" in h:
                i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
            gw = _wcs(h["SCI"].header)
        scale = float(np.mean(proj_plane_pixel_scales(gw))) * 3600
        free = np.isfinite(i2d) & ~source_mask_grid(i2d, scale, cfg)
        ex = reproject_mask(dest, gw, i2d.shape)
        print(f"  {f}: свободного неба для подгонки наклона {(free & ~ex).sum() / free.sum() * 100:.1f}% свободного")


if __name__ == "__main__":
    main()
