"""Фаза 8, шаг 3: крупный масштаб своей сборки уровня 2 против i2d — только там, где i2d верен.

Зоны: «i2d верен» — куда не попала ни одна экспозиция с поправкой наведения больше I2D_OK_ARCSEC
(у OMC3-N — визиты 001 и 003), с отступом 128 px от границы зоны и края данных. Обе карты сглаживаются гауссом
σ 64 px с нормировкой по пустым; подгонка своя = a·i2d + b по точкам через 32 px (с отбрасыванием 3σ). a < 1 —
сборка съедает перепады туманности (как 0,87 у F150W в гибриде).

Запуск: uv run python scripts/large_scale_check.py <фильтр> [<папка листа>]
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

warnings.filterwarnings("ignore")
SIGMA = 64


def smooth(a: np.ndarray, ok: np.ndarray) -> np.ndarray:
    num = ndimage.gaussian_filter(np.where(ok, a, 0).astype(np.float32), SIGMA, mode="nearest")
    den = ndimage.gaussian_filter(ok.astype(np.float32), SIGMA, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0.5, num / den, np.nan)


def main() -> None:
    f = sys.argv[1].upper()
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "out/omc3n/проверка_9/уровень2")
    out.mkdir(parents=True, exist_ok=True)
    d = pathlib.Path("data/work/omc3n/level2")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    pointing = json.loads((d / "astrometry.json").read_text(encoding="utf-8"))
    with fits.open(d / meta["filters"][f]["file"]) as h:
        own = np.array(h["SCI"].data, dtype=np.float32)
        grid = WCS(h["SCI"].header)
    with fits.open(glob.glob(f"data/raw/jwst/jw07433-o001_t001_nircam_clear-{f.lower()}/*i2d.fits")[0]) as h:
        ref = np.array(h["SCI"].data, dtype=np.float32)
        ref[np.asarray(h["WHT"].data) <= 0] = np.nan
    H, W = own.shape
    # зона считается на сетке через 8 px: полный кадр коротких — 92 млн px
    yy, xx = np.mgrid[0:H:8, 0:W:8]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    cover = {True: np.zeros(yy.shape, dtype=bool), False: np.zeros(yy.shape, dtype=bool)}   # с поправкой / без
    shifts = []
    for name in meta["filters"][f]["pointing"]:
        corr = pointing["files"][name]
        is_moved = math.hypot(*corr["shift"]) > I2D_OK_ARCSEC
        if is_moved:
            shifts.append(corr["shift"])
        w = WCS(fits.getheader(f"data/raw/omc3n/l2/{name}", "SCI")).celestial
        n = fits.getheader(f"data/raw/omc3n/l2/{name}", "SCI")["NAXIS1"]
        t = np.linspace(0, n - 1, 20)
        fx = np.concatenate([t, np.full(20, n - 1), t[::-1], np.zeros(20)])
        fy = np.concatenate([np.zeros(20), t, np.full(20, n - 1), t[::-1]])
        gx, gy = grid.all_world2pix(*w.all_pix2world(fx, fy, 0), 0)
        cover[is_moved] |= MplPath(np.column_stack([gx, gy])).contains_points(pts).reshape(yy.shape)
    ok = np.isfinite(own) & np.isfinite(ref)
    so, sr = smooth(own, ok), smooth(ref, ok)
    # зона Б: только экспозиции с поправкой — i2d там стоит по исходному WCS; сдвигаем его на медианную поправку
    sr_b = sr
    if shifts:
        s_med = np.median(np.asarray(shifts), axis=0)
        cy, cx = H / 2, W / 2
        ra, dec = grid.all_pix2world(cx, cy, 0)
        px, py = grid.all_world2pix(ra - s_med[0] / 3600 / math.cos(math.radians(float(dec))), dec - s_med[1] / 3600, 0)
        sr_b = ndimage.shift(sr, (-(float(py) - cy), -(float(px) - cx)), order=1, mode="constant", cval=np.nan)
    lines, fits_ = [], {}
    for zname, zmask, ref_s in (("«i2d верен» (без поправки)", cover[False] & ~cover[True], sr),
                                ("«только с поправкой», i2d сдвинут", cover[True] & ~cover[False], sr_b)):
        zone = ndimage.binary_erosion(ok[::8, ::8] & zmask, iterations=16)     # отступ 128 px
        samp = np.zeros_like(zone)
        samp[::4, ::4] = zone[::4, ::4]                                         # точки через 32 px
        samp &= np.isfinite(so[::8, ::8]) & np.isfinite(ref_s[::8, ::8])
        x, y = ref_s[::8, ::8][samp].astype(np.float64), so[::8, ::8][samp].astype(np.float64)
        if len(x) < 50:
            lines.append(f"{f}: зона {zname} — мало точек ({len(x)})")
            continue
        keep = np.ones(len(x), dtype=bool)
        for _ in range(5):
            a, b = np.polyfit(x[keep], y[keep], 1)
            r = y - (a * x + b)
            sd = 1.4826 * np.median(np.abs(r[keep]))
            new = np.abs(r) < 3 * sd
            if np.array_equal(new, keep):
                break
            keep = new
        lvl = float(np.median(x))
        lines.append(f"{f}: зона {zname} {zone.mean() * 100:.1f}% кадра, точек {len(x)} (в подгонке {keep.sum()}); "
                     f"своя = {a:.4f}·i2d {b:+.4f}; разброс остатков {sd / lvl * 100:.2f}% медианной яркости; медиана "
                     f"i2d {lvl:.3f}, размах (5–95%) {np.percentile(x, 5):.3f}–{np.percentile(x, 95):.3f}")
        fits_[zname] = (x, y, keep, a, b, zone)
    line = "\n".join(lines)
    print(line)
    (out / f"крупный_масштаб_{f}.txt").write_text(line + "\n", encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(22, 6), dpi=100, gridspec_kw={"width_ratios": [1, 1, 2.2]})
    for k, (zname, (x, y, keep, a, b, zone)) in enumerate(fits_.items()):
        ax[k].scatter(x[keep], y[keep], s=3, alpha=0.5)
        ax[k].scatter(x[~keep], y[~keep], s=3, c="r", alpha=0.5)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax[k].plot(lim, lim, "k--", lw=0.8, label="своя = i2d")
        ax[k].plot(lim, [a * v + b for v in lim], "g-", lw=1, label=f"{a:.3f}·i2d {b:+.3f}")
        ax[k].set_xlabel("i2d, сглажено σ 64 px")
        ax[k].set_ylabel("своя сборка, сглажено σ 64 px")
        ax[k].set_title(zname, fontsize=9)
        ax[k].legend()
        ax[2].contour(zone, levels=[0.5], colors="k" if k == 0 else "m", linewidths=0.6)
    ratio = np.where(ok, so / sr_b, np.nan)[::8, ::8]
    im = ax[2].imshow(ratio, origin="lower", cmap="RdBu_r", vmin=0.8, vmax=1.2)
    fig.colorbar(im, ax=ax[2], shrink=0.8, label="своя / i2d (i2d сдвинут на поправку визитов 002/004)")
    ax[2].set_title("отношение крупного масштаба; чёрный контур — зона без поправки, лиловый — только с поправкой",
                    fontsize=9)
    fig.suptitle(f"OMC3-N {f}: крупный масштаб своей сборки уровня 2 против i2d\n{line}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / f"45_крупный_масштаб_{f}.png")


if __name__ == "__main__":
    main()
