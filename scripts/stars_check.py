"""Фаза 8: FWHM и поток звёзд своей сборки уровня 2 против i2d — 20+ звёзд на фильтр, только там, где i2d верен.

Зона — пиксели, куда не попала ни одна экспозиция с поправкой наведения больше I2D_OK_ARCSEC (у OMC3-N — визиты
001/003). Звёзды: локальные максимумы своей сборки ярче PEAK_SIGMA шумов, изолированные (нет пика ярче четверти ближе
15 px), окно 31×31 без пустых пикселей в обеих картах; на туманности — с местным фоном по кольцу (level2._fwhm_flux).
FWHM — гаусс, поток — апертура 2·FWHM i2d.

Запуск: uv run python scripts/stars_check.py [число звёзд]
"""

from __future__ import annotations

import glob
import json
import math
import pathlib
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from matplotlib.path import Path as MplPath
from scipy import ndimage

from skyforge.astrometry import I2D_OK_ARCSEC
from skyforge.level2 import _fwhm_flux

warnings.filterwarnings("ignore")
PEAK_SIGMA = 50


def zone_ok(grid: WCS, shape: tuple[int, int], names: list[str], pointing: dict) -> np.ndarray:
    """Пиксели сетки, куда не попала ни одна экспозиция с поправкой больше I2D_OK_ARCSEC (через 4 px)."""
    H, W = shape
    yy, xx = np.mgrid[0:H:4, 0:W:4]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    moved = np.zeros(yy.shape, dtype=bool)
    for name in names:
        if math.hypot(*pointing["files"][name]["shift"]) <= I2D_OK_ARCSEC:
            continue
        h = fits.getheader(f"data/raw/omc3n/l2/{name}", "SCI")
        w, n = WCS(h).celestial, h["NAXIS1"]
        t = np.linspace(0, n - 1, 20)
        fx = np.concatenate([t, np.full(20, n - 1), t[::-1], np.zeros(20)])
        fy = np.concatenate([np.zeros(20), t, np.full(20, n - 1), t[::-1]])
        gx, gy = grid.all_world2pix(*w.all_pix2world(fx, fy, 0), 0)
        moved |= MplPath(np.column_stack([gx, gy])).contains_points(pts).reshape(yy.shape)
    ok = ~ndimage.binary_dilation(moved, iterations=8)          # отступ 32 px от зоны со сдвигом
    return np.repeat(np.repeat(ok, 4, axis=0), 4, axis=1)[:H, :W]


def singles(f: str, own: np.ndarray, ref: np.ndarray, grid: WCS, sm: np.ndarray, bg: np.ndarray, loc: np.ndarray,
            noise: float, meta: dict, pointing: dict, nmax: int) -> np.ndarray:
    """Звёзды по всему кадру против отдельных экспозиций _cal (WCS с поправкой наведения): медиана FWHM и потока по
    экспозициям, где звезда целиком в кадре детектора. Строки: FWHM своя/экспоз., поток своя/экспоз., то же у i2d
    (NaN, где i2d в этом месте не годится — сдвиг визита)."""
    from skyforge.astrometry import apply_pointing

    names = list(meta["filters"][f]["pointing"])
    frames = []
    for name in names:
        with fits.open(f"data/raw/omc3n/l2/{name}") as h:
            s = np.array(h["SCI"].data, dtype=np.float32)
            s[(np.asarray(h["DQ"].data) & 1) != 0] = np.nan
            frames.append((s, apply_pointing(WCS(h["SCI"].header).celestial, pointing["files"][name])))
    pk = np.argwhere((sm == ndimage.maximum_filter(sm, 31)) & (sm - bg > PEAK_SIGMA * noise))
    pk = pk[np.argsort(-(sm - bg)[pk[:, 0], pk[:, 1]])]
    out = []
    for y, x in pk:
        if not (20 <= y < own.shape[0] - 20 and 20 <= x < own.shape[1] - 20):
            continue
        win = (slice(y - 15, y + 16), slice(x - 15, x + 16))
        if not np.isfinite(own[win]).all():
            continue
        peak = sm[y, x] - bg[y, x]
        if (loc[win] & (sm[win] - bg[win] > 0.25 * peak)).sum() > 1:
            continue
        fo, _ = _fwhm_flux(own, y, x, 6)
        if not np.isfinite(fo):
            continue
        ra, dec = grid.all_pix2world(x, y, 0)
        vals = []
        for s, w in frames:
            lx, ly = w.wcs_world2pix(ra, dec, 0)
            if not (20 <= lx < s.shape[1] - 20 and 20 <= ly < s.shape[0] - 20):
                continue
            px, py = (int(round(float(v))) for v in w.all_world2pix(ra, dec, 0))
            if not (20 <= px < s.shape[1] - 20 and 20 <= py < s.shape[0] - 20):
                continue
            cut = s[py - 20:py + 21, px - 20:px + 21].astype(np.float64)
            bad = ~np.isfinite(cut)
            if bad.mean() > 0.05 or bad[17:24, 17:24].any():
                continue   # насыщенное ядро или много плохих пикселей
            if bad.any():   # одиночные плохие пиксели (DQ) — средним непустых соседей 3×3
                num = ndimage.uniform_filter(np.where(bad, 0, cut), 3)
                den = ndimage.uniform_filter((~bad).astype(float), 3)
                cut[bad] = (num / np.maximum(den, 1e-9))[bad]
            fs, fls = _fwhm_flux(cut, 20, 20, max(6, 2 * fo))
            if np.isfinite(fs) and fls > 0:
                vals.append((fs, fls))
        if len(vals) < 2:
            continue
        v = np.median(np.array(vals), axis=0)
        fl_own = _fwhm_flux(own, y, x, max(6, 2 * fo))[1]
        fi, fli = (_fwhm_flux(ref, y, x, 6)[0], _fwhm_flux(ref, y, x, max(6, 2 * fo))[1]) \
            if np.isfinite(ref[win]).all() else (np.nan, np.nan)
        out.append((fo / v[0], fl_own / v[1], fi / v[0], fli / v[1]))
        if len(out) >= nmax:
            break
    a = np.array(out).reshape(-1, 4)
    return a


def main() -> None:
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    d = pathlib.Path("data/work/omc3n/level2")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pointing = json.loads((d / "astrometry.json").read_text(encoding="utf-8"))
    lines = ["фильтр\tзвёзд\tFWHM своя/i2d: медиана (5–95%)\tпоток своя/i2d: медиана (5–95%)\tв допуске FWHM"]
    for f in ("F150W", "F200W", "F356W", "F480M"):
        with fits.open(d / meta["filters"][f]["file"]) as h:
            own = np.array(h["SCI"].data, dtype=np.float32)
            grid = WCS(h["SCI"].header)
        with fits.open(glob.glob(f"data/raw/jwst/jw07433-o001_t001_nircam_clear-{f.lower()}/*i2d.fits")[0]) as h:
            ref = np.array(h["SCI"].data, dtype=np.float32)
            ref[np.asarray(h["WHT"].data) <= 0] = np.nan
        ok = zone_ok(grid, own.shape, list(meta["filters"][f]["pointing"]), pointing)
        a = np.nan_to_num(own)
        sm = ndimage.gaussian_filter(a, 1.0)
        d1 = np.diff(own, axis=1)
        d1 = d1[np.isfinite(d1)]
        noise = 1.4826 * float(np.median(np.abs(d1 - np.median(d1)))) / math.sqrt(2)
        bg = ndimage.median_filter(sm[::8, ::8], 9)
        bg = np.repeat(np.repeat(bg, 8, axis=0), 8, axis=1)[:a.shape[0], :a.shape[1]]
        pk = np.argwhere((sm == ndimage.maximum_filter(sm, 31)) & (sm - bg > PEAK_SIGMA * noise) & ok)
        pk = pk[np.argsort(-(sm - bg)[pk[:, 0], pk[:, 1]])]
        loc = sm == ndimage.maximum_filter(sm, 5)
        rows = []
        for y, x in pk:
            if not (20 <= y < a.shape[0] - 20 and 20 <= x < a.shape[1] - 20):
                continue
            win = (slice(y - 15, y + 16), slice(x - 15, x + 16))
            if not (np.isfinite(own[win]).all() and np.isfinite(ref[win]).all()):
                continue   # насыщенное ядро или край
            peak = sm[y, x] - bg[y, x]
            if (loc[win] & (sm[win] - bg[win] > 0.25 * peak)).sum() > 1:
                continue
            f1, _ = _fwhm_flux(ref, y, x, 6)
            if not np.isfinite(f1):
                continue
            f2, fl2 = _fwhm_flux(own, y, x, max(6, 2 * f1))
            fl1 = _fwhm_flux(ref, y, x, max(6, 2 * f1))[1]
            if np.isfinite(f2) and fl1 > 0 and fl2 > 0:
                rows.append((f2 / f1, fl2 / fl1, f1))
            if len(rows) >= nmax:
                break
        r = np.array(rows)
        single = singles(f, own, ref, grid, sm, bg, loc, noise, meta, pointing, nmax)
        tol = 8.0 if np.median(r[:, 2]) < 2.5 else 5.0
        inside = int((np.abs(r[:, 0] - 1) * 100 <= tol).sum())
        q = lambda v: (f"{np.nanmedian(v):.3f} ({np.nanpercentile(v, 5):.3f}–{np.nanpercentile(v, 95):.3f}), "  # noqa: E731
                       f"n {int(np.isfinite(v).sum())}")
        lines.append(f"{f}\t{len(r)}\t{q(r[:, 0])}\t{q(r[:, 1])}\t{inside}/{len(r)} (±{tol:g}%)")
        print(lines[-1])
        if len(single):
            lines.append(f"{f} к отдельным экспозициям, весь кадр\t{len(single)}\tсвоя {q(single[:, 0])}, i2d "
                         f"{q(single[:, 2])}\tсвоя {q(single[:, 1])}, i2d {q(single[:, 3])}\t—")
            print(lines[-1])
    out = pathlib.Path("out/omc3n/проверка_9/уровень2/звёзды_к_i2d.tsv")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
