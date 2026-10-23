"""Фаза 8, шаг 3: сколько крупной структуры туманности снимает destripe в отдельной экспозиции.

Для экспозиций визитов без поправки наведения (у OMC3-N — 001 и 003, там i2d верен): destripe_frame с той же маской,
что в сборке (по i2d, спроецированному в кадр детектора); снятое = до − после. Обе карты сглаживаются гауссом σ 32 px
по пикселям, где есть i2d; подгонка снятое = k·(i2d − медиана) + c. k > 0 — destripe уносит долю перепадов туманности
(k = 0,1 — десятую часть), k ≈ 0 — снимает только полосы.

Запуск: uv run python scripts/destripe_effect.py <фильтр> [число экспозиций]
"""

from __future__ import annotations

import glob
import pathlib
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy import ndimage

from skyforge.config import Config
from skyforge.level2 import destripe_frame, load_exposure, mask_to_frame, source_mask_grid

warnings.filterwarnings("ignore")


def main() -> None:
    f = sys.argv[1].upper()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    cfg = Config.load("configs/omc3n/showcase.yaml", "showcase")
    c = cfg.level2
    with fits.open(glob.glob(f"data/raw/jwst/jw07433-o001_t001_nircam_clear-{f.lower()}/*i2d.fits")[0]) as h:
        i2d = np.array(h["SCI"].data, dtype=np.float32)
        i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
        gw = WCS(h["SCI"].header)
    scale = float(np.mean(np.abs(np.linalg.eigvals(gw.pixel_scale_matrix)))) * 3600
    src = source_mask_grid(i2d, scale, cfg)
    files = [p for p in sorted(glob.glob("data/raw/omc3n/l2/*_cal.fits"))
             if fits.getheader(p)["FILTER"] == f and pathlib.Path(p).name[10:13] in ("001", "003")]
    rows = []
    for p in files[:: max(1, len(files) // n)][:n]:
        sci, _, fw, _ = load_exposure(pathlib.Path(p), c.dq_bad)
        mask = mask_to_frame(src, gw, fw, sci.shape, c.mask_grow_px)
        clean, st = destripe_frame(sci, mask, c)
        ref, _ = reproject_interp((i2d, gw), fw, shape_out=sci.shape, order="bilinear")
        ok = np.isfinite(sci) & np.isfinite(clean) & np.isfinite(ref)
        if ok.mean() < 0.3:
            continue

        def sm(a: np.ndarray) -> np.ndarray:
            num = ndimage.gaussian_filter(np.where(ok, a, 0), 32)
            den = ndimage.gaussian_filter(ok.astype(np.float64), 32)
            return np.where(den > 0.5, num / den, np.nan)

        rem, neb = sm(sci - clean), sm(ref)
        sel = (np.isfinite(rem) & np.isfinite(neb))[::16, ::16]           # точки через 16 px
        y, x = rem[::16, ::16][sel], neb[::16, ::16][sel]
        x = x - np.median(x)
        k, b = np.polyfit(x, y, 1)
        frac_mask = float(mask[ok].mean())
        rows.append((pathlib.Path(p).name, k, float(np.std(y)), float(np.std(x)), frac_mask))
        print(f"{pathlib.Path(p).name}: снятое = {k:+.4f}·(i2d − медиана) {b:+.4f}; разброс снятого {np.std(y):.4f}, "
              f"перепадов i2d {np.std(x):.4f}; под маской {frac_mask * 100:.0f}% кадра")
    if rows:
        print(f"{f}: медиана k {np.median([r[1] for r in rows]):+.4f} по {len(rows)} экспозициям")


if __name__ == "__main__":
    main()
