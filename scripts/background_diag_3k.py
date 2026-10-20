"""3k, шаг 1: диагноз неровного фона showcase — без правок конвейера.

1. Карты числа экспозиций (NEXP своей сборки) по фильтрам рядом с картой цвета неба (G − (R+B)/2 и яркость,
   сглаженные 64 px по пустому небу) на итоге showcase; границы NEXP поверх карты цвета.
2. Медиана и σ пустого неба по областям NEXP, по фильтрам, на очищенных кадрах (до растяжки).
3. Доля пикселей неба, обрезанных в 0: по фильтрам после растяжки и по каналам итога, по областям NEXP.
4. Полоса F356W: профиль по столбцам на нашей сборке и на i2d STScI в том же месте.

Запуск: uv run python scripts/background_diag_3k.py <папка> <итоговый png showcase>
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import load_input, load_sky_mask
from skyforge.compose import load_stretched
from skyforge.config import Config
from skyforge.fetch import require_products
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def nexp_maps(P: Paths, wcs: WCS, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    d = P.work_root / "ngc3132" / "level2"
    m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    out = {}
    for f, v in m["filters"].items():
        with fits.open(d / v["file"]) as h:
            ne = h["NEXP"].data.astype(np.float32)
            ww = WCS(h["SCI"].header)
        g, _ = reproject_interp((ne, ww), wcs, shape_out=shape, order="nearest-neighbor")
        out[f] = np.nan_to_num(g).astype(np.int16)
    return out


def smooth_on(mask: np.ndarray, a: np.ndarray, s: float) -> np.ndarray:
    num = ndimage.gaussian_filter(np.where(mask, a, 0.0), s)
    den = ndimage.gaussian_filter(mask.astype(float), s)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)


def main() -> None:
    out, final = pathlib.Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    raw, wcs, meta = load_aligned(cfg, P)
    clean, _ = load_input(cfg, P)
    layers, _ = load_stretched(cfg, P)
    sky = load_sky_mask(cfg, P)                                   # сетка: строка 0 — низ
    shape = sky.shape
    ne = nexp_maps(P, wcs, shape)
    rgb = np.asarray(Image.open(final).convert("RGB")).astype(np.float32)[::-1]   # в ориентацию сетки
    g_ex = rgb[..., 1] - (rgb[..., 0] + rgb[..., 2]) / 2
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
    g_s, l_s = smooth_on(sky, g_ex, 64), smooth_on(sky, lum, 64)

    # 1. карты
    fig, ax = plt.subplots(2, 4, figsize=(22, 10))
    for a_, (f, n) in zip(ax.ravel()[:6], ne.items()):
        im = a_.imshow(n[::-1], cmap="viridis", vmin=0, vmax=max(8, n.max()))
        a_.set_title(f"{f}: число экспозиций"); a_.axis("off"); fig.colorbar(im, ax=a_, fraction=0.04)
    im = ax[1, 2].imshow(g_s[::-1], cmap="PiYG_r", vmin=-1.5, vmax=1.5)
    ax[1, 2].set_title("небо: G − (R+B)/2, уровни, сглажено 64 px"); fig.colorbar(im, ax=ax[1, 2], fraction=0.04)
    im = ax[1, 3].imshow(l_s[::-1], cmap="magma", vmin=np.nanpercentile(l_s, 2), vmax=np.nanpercentile(l_s, 98))
    ax[1, 3].set_title("небо: яркость, уровни, сглажено 64 px"); fig.colorbar(im, ax=ax[1, 3], fraction=0.04)
    for a_ in ax[1, 2:]:
        for f, col in (("F090W", "k"), ("F356W", "b"), ("F470N", "r")):
            a_.contour(ne[f][::-1], levels=np.arange(0.5, 12, 1), colors=col, linewidths=0.4)
        a_.axis("off")
    fig.suptitle("NEXP по фильтрам и цвет/яркость неба showcase; контуры NEXP: чёрные F090W, синие F356W, красные F470N")
    fig.tight_layout(); fig.savefig(out / "01_NEXP_и_цвет_неба.png", dpi=60); plt.close(fig)

    # 2. медиана и σ неба по NEXP на очищенных кадрах
    rows = ["фильтр\tNEXP\tмедиана\tσ (MAD)\tпикселей"]
    for f, a in clean.items():
        for k in np.unique(ne[f][sky]):
            m = sky & (ne[f] == k) & np.isfinite(a)
            if m.sum() < 2000:
                continue
            v = a[m]
            med = float(np.median(v)); sig = float(1.4826 * np.median(np.abs(v - med)))
            rows.append(f"{f}\t{k}\t{med:.5g}\t{sig:.5g}\t{int(m.sum())}")
    (out / "02_небо_по_NEXP_до_растяжки.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # 3. обрезка в 0: слои после растяжки по NEXP своего фильтра; каналы итога по областям (F090W, F356W, F470N)
    rows = ["фильтр\tNEXP\tдоля = 0 после растяжки, %\tпикселей"]
    for f, L in layers.items():
        for k in np.unique(ne[f][sky]):
            m = sky & (ne[f] == k) & np.isfinite(L)
            if m.sum() < 2000:
                continue
            rows.append(f"{f}\t{k}\t{(L[m] <= 0).mean() * 100:.1f}\t{int(m.sum())}")
    combo = ne["F090W"] * 100 + ne["F356W"] * 10 + np.minimum(ne["F470N"], 9)
    rows.append("")
    rows.append("область (NEXP F090W/F356W/F470N)\tR = 0, %\tG = 0, %\tB = 0, %\tмедиана R\tмедиана G\tмедиана B\tпикселей")
    for k in np.unique(combo[sky]):
        m = sky & (combo == k)
        if m.sum() < 5000:
            continue
        v = rgb[m]
        rows.append(f"{k // 100}/{(k // 10) % 10}/{k % 10}\t" + "\t".join(f"{(v[:, c] <= 0).mean() * 100:.1f}" for c in range(3))
                    + "\t" + "\t".join(f"{np.median(v[:, c]):.2f}" for c in range(3)) + f"\t{int(m.sum())}")
    (out / "03_обрезка_в_0_по_NEXP.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    # 4. полоса F356W: профиль по столбцам неба — наша сборка (очищенная) и i2d STScI на той же сетке
    prods = require_products(cfg.model_copy(update={"source": "level3"}), P)
    with fits.open(P.abs(prods["F356W"].path)) as h:
        i2d = np.array(h["SCI"].data, dtype=np.float32)
        wi = WCS(h["SCI"].header)
    i2d_g, _ = reproject_interp((i2d, wi), wcs, shape_out=shape, order="bilinear")
    rows_band = slice(int(shape[0] * 0.55), int(shape[0] * 0.95))                 # верх кадра (строки сетки)
    def colprof(a):
        v = np.where(sky[rows_band] & np.isfinite(a[rows_band]), a[rows_band], np.nan)
        return np.nanmedian(v, axis=0)
    ours_p, i2d_p = colprof(clean["F356W"]), colprof(i2d_g)
    fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    for a_, p, lab in ((ax[0], ours_p, "наша сборка F356W (очищенная)"), (ax[1], i2d_p, "i2d STScI F356W")):
        base = ndimage.median_filter(np.nan_to_num(p), 151)
        a_.plot(p - base, lw=0.8); a_.set_title(f"{lab}: медиана неба по столбцу минус гладкий ход (строки верхней части кадра)")
        a_.grid(alpha=0.3); a_.set_ylim(np.nanpercentile(p - base, 1) * 1.5, np.nanpercentile(p - base, 99) * 1.5)
    ax[1].set_xlabel("столбец кадра 2261×2022")
    fig.tight_layout(); fig.savefig(out / "04_полоса_F356W_наша_и_i2d.png", dpi=70); plt.close(fig)
    ex_o = ours_p - ndimage.median_filter(np.nan_to_num(ours_p), 151)
    ex_i = i2d_p - ndimage.median_filter(np.nan_to_num(i2d_p), 151)
    for x0 in range(1150, 1320, 20):
        print(f"столбцы {x0}–{x0 + 20}: наша {np.nanmean(ex_o[x0:x0 + 20]):+.4f}, i2d {np.nanmean(ex_i[x0:x0 + 20]):+.4f}")
    np.save(out / "nexp_F356W.npy", ne["F356W"])


if __name__ == "__main__":
    main()
