"""Фаза 8: что находит детектор точечных источников (stretch.exclude_point_sources) — звёзды, выбросы или узлы туманности.

По слою после clean (вход растяжки) повторяет отбор кандидатов stretch.point_sources и делит находки на три класса:
- выброс — одиночный пиксель: среднее четырёх соседей меньше половины того, что даёт PSF (доля соседей у ярких находок);
- звезда — FWHM по моментам второго порядка (круг 4 px) ≤ 1,5·PSF и вытянутость ≤ 1,3;
- узел — остальное (шире или вытянутое).
PSF — медиана по верхним 5% находок по амплитуде. Для фильтров своей сборки уровня 2 у каждой находки проверяется, есть
ли она в i2d того же фильтра (на исходной сетке сборки: превышение над местной медианой у i2d не меньше половины гибрида).
Лист — N случайных находок, кропы 21 px ×4.

Запуск: uv run python scripts/point_sources_check.py <фильтр> <папка> [N] [конфиг]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import clean_dir
from skyforge.config import Config
from skyforge.field import block_median_map
from skyforge.paths import Paths
from skyforge.preview import contact_sheet


def candidates(a: np.ndarray, ps, scale: float) -> tuple[np.ndarray, np.ndarray, float]:
    ok = np.isfinite(a)
    bg = block_median_map(a, max(16, int(round(ps.window_arcsec / scale))), fill_empty=True)
    r = np.where(ok, a - bg, 0).astype(np.float32)
    d = np.diff(np.where(ok, a, np.nan), axis=1)
    d = d[np.isfinite(d)]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))
    peaks = np.argwhere((r == ndimage.maximum_filter(r, 5)) & (r > ps.sigma * noise)
                        & (r > ps.sharpness * np.maximum(ndimage.uniform_filter(r, 5), 1e-12)))
    return peaks, r, noise


def shape(r: np.ndarray, y: int, x: int, rad: int = 4) -> tuple[float, float, float, float]:
    """(FWHM px, вытянутость, амплитуда, доля соседей) по положительному остатку в круге rad вокруг пика."""
    sub = r[y - rad:y + rad + 1, x - rad:x + rad + 1].astype(np.float64)
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    w = np.where((yy ** 2 + xx ** 2 <= rad * rad) & (sub > 0), sub, 0.0)
    s = w.sum()
    peak = float(r[y, x])
    near = float((r[y - 1, x] + r[y + 1, x] + r[y, x - 1] + r[y, x + 1]) / 4 / peak) if peak > 0 else float("nan")
    if s <= 0:
        return float("nan"), float("nan"), peak, near
    my, mx = (w * yy).sum() / s, (w * xx).sum() / s
    cyy, cxx = (w * (yy - my) ** 2).sum() / s, (w * (xx - mx) ** 2).sum() / s
    cxy = (w * (yy - my) * (xx - mx)).sum() / s
    ev = np.clip(np.linalg.eigvalsh(np.array([[cyy, cxy], [cxy, cxx]])), 1e-6, None)
    fwhm = 2.3548 * float(np.sqrt(np.sqrt(ev[0] * ev[1])))
    return fwhm, float(np.sqrt(ev[1] / ev[0])), peak, near


def in_i2d(peaks: np.ndarray, grid: WCS, blend: pathlib.Path, i2d: pathlib.Path) -> np.ndarray:
    """Есть ли находка в i2d: на исходной сетке сборки превышение i2d над медианой 15×15 (максимум в 5×5) не меньше
    половины такого же у гибрида."""
    with fits.open(blend, memmap=True) as hb, fits.open(i2d, memmap=True) as hi:
        b, a, w = hb["SCI"].data, hi["SCI"].data, WCS(hb["SCI"].header)
        ra, dec = grid.pixel_to_world_values(peaks[:, 1].astype(float), peaks[:, 0].astype(float))
        px, py = w.world_to_pixel_values(ra, dec)
        out = np.zeros(len(peaks), dtype=bool)
        H, W = b.shape
        for k, (x, y) in enumerate(zip(np.rint(px).astype(int), np.rint(py).astype(int))):
            if not (8 <= y < H - 8 and 8 <= x < W - 8):
                continue
            ex = []
            for img in (b, a):
                cut = np.asarray(img[y - 7:y + 8, x - 7:x + 8], dtype=np.float64)
                ex.append(np.nanmax(cut[5:10, 5:10]) - np.nanmedian(cut))
            out[k] = ex[1] >= 0.5 * ex[0]
    return out


def main() -> None:
    f, out = sys.argv[1], pathlib.Path(sys.argv[2])
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    cfg_path = sys.argv[4] if len(sys.argv) > 4 else "configs/omc3n/showcase.yaml"
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load(cfg_path, "showcase").model_copy(update={"bin": 1, "filters": [f]})
    _, grid, _ = load_aligned(c, P)
    a = np.load(clean_dir(c, P) / f"{f}.npy")
    peaks, r, noise = candidates(a, c.stretch.exclude_point_sources, 0.0629)
    H, W = a.shape
    peaks = peaks[(peaks[:, 0] >= 8) & (peaks[:, 0] < H - 8) & (peaks[:, 1] >= 8) & (peaks[:, 1] < W - 8)]
    meas = np.array([shape(r, y, x) for y, x in peaks])
    top = meas[:, 2] >= np.percentile(meas[:, 2], 95)
    psf, psf_near = float(np.nanmedian(meas[top, 0])), float(np.nanmedian(meas[top, 3]))
    spike = meas[:, 3] < 0.5 * psf_near
    star = ~spike & (meas[:, 0] <= 1.5 * psf) & (meas[:, 1] <= 1.3)
    cls = np.where(spike, "выброс", np.where(star, "звезда", "узел"))
    both = None
    if c.uses_level2(f):
        from skyforge.fetch import require_products
        from skyforge.level2 import built_products

        both = in_i2d(peaks, grid, built_products(c, P)[f], pathlib.Path(require_products(c, P, level3=True)[f].path))
    rng = np.random.default_rng(0)
    pick = rng.choice(len(peaks), size=min(n, len(peaks)), replace=False)

    def shares(sel: np.ndarray) -> str:
        m = max(1, int(sel.sum()))
        return ", ".join(f"{k} {int((cls[sel] == k).sum())} ({(cls[sel] == k).sum() / m:.0%})"
                         for k in ("звезда", "выброс", "узел"))

    sel = np.zeros(len(peaks), dtype=bool)
    sel[pick] = True
    line = (f"{f}: находок {len(peaks)}; PSF {psf:.2f} px, соседи у PSF {psf_near:.2f} пика; из {len(pick)} случайных: "
            f"{shares(sel)}; по всем: {shares(np.ones(len(peaks), dtype=bool))}")
    if both is not None:
        line += (f"; есть в i2d {int(both.sum())} ({both.mean():.0%}), нет в i2d {int((~both).sum())}: "
                 f"{shares(~both)}")
    print(line)
    tiles, labels = [], []
    for i in pick:
        y, x = peaks[i]
        cut = r[y - 10:y + 11, x - 10:x + 11]
        lo, hi = np.percentile(cut, [1, 99.5])
        tiles.append(np.clip((cut - lo) / max(hi - lo, 1e-9), 0, 1)[::-1])
        labels.append(f"{cls[i]} FWHM {meas[i, 0]:.1f} сос. {meas[i, 3]:.2f}"
                      + ("" if both is None else (" · в i2d" if both[i] else " · нет в i2d")))
    contact_sheet(tiles, labels, out / f"41_точечные_{f}.png", cols=6, tile_inches=2.0, interpolation="nearest",
                  title=f"OMC3-N {f}: {len(pick)} случайных находок детектора, кропы 21 px ×4 — " + line)
    (out / f"41_точечные_{f}.txt").write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
