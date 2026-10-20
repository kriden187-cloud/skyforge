"""3k, п.2, проверка: перепад фона наш (сведение уровней в стеке) или в данных.

1. Карта остатков неба, как лист 12 (небо − медиана, сглажено 24 px по пустому небу, в σ), — наша сборка
   и i2d STScI того же фильтра на нашей сетке, и их разность.
2. Наклоны экспозиций относительно соседей (таблица scripts/exposure_slopes_3k.py), по детекторам.
Всё на одном листе 14; числа — в таблицу 14_наш_и_i2d.tsv.

Запуск: uv run python scripts/sky_vs_i2d_3k.py <папка>   (в папке — 14_наклоны_экспозиций.tsv)
"""

from __future__ import annotations

import csv
import pathlib
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

from skyforge.align import load_aligned
from skyforge.clean import load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.fetch import require_products
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import smooth_on  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMOOTH_PX = 24
DET_COLORS = {"NRCB1": "tab:blue", "NRCB2": "tab:orange", "NRCB3": "tab:green", "NRCB4": "tab:red",
              "NRCBLONG": "tab:purple"}


def residual(a: np.ndarray, ok: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = a[ok]
    med = float(np.median(v))
    sig = float(1.4826 * np.median(np.abs(v - med)))
    return smooth_on(ok, np.where(ok, a, 0) - med, SMOOTH_PX), med, sig


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    raw, wcs, _ = load_aligned(cfg, P)
    sky, sd = load_sky_mask(cfg, P), load_nebula_sd(cfg, P)
    prods = require_products(cfg.model_copy(update={"source": "level3"}), P)
    slopes = []
    tsv = out / "14_наклоны_экспозиций.tsv"
    if tsv.exists():
        with tsv.open(encoding="utf-8") as fh:
            slopes = [r for r in csv.DictReader(fh, delimiter="\t") if r["наклон x, σ/1000 px"] not in ("—", "")]
    filters = list(raw)
    fig = plt.figure(figsize=(26, 6.2 * len(filters) / 2 + 7))
    gs = fig.add_gridspec(len(filters) // 2 + 1, 7, width_ratios=[1, 1, 1, 0.08, 1, 1, 1], hspace=0.25, wspace=0.08)
    near = (sd < 0) & (sd > -300)
    far = sd < -600
    rows = ["фильтр\tσ неба: наша / i2d\tразброс сглаженного неба 5–95%, σ: наша / i2d / разность"
            "\tсвязь карт (коэфф. корреляции)\tу туманности (0–300 px) минус дальше 600 px, σ: наша / i2d"]
    for k, f in enumerate(filters):
        a = np.asarray(raw[f], dtype=np.float32)
        with fits.open(P.abs(prods[f].path), memmap=True) as h:
            i2 = np.array(h["SCI"].data, dtype=np.float32)
            if "WHT" in h:
                i2[np.asarray(h["WHT"].data) <= 0] = np.nan
            wi = WCS(h["SCI"].header)
        i2g, _ = reproject_interp((i2, wi), wcs, shape_out=a.shape, order="bilinear")
        ok = sky & np.isfinite(a) & np.isfinite(i2g)
        mo, _, so = residual(a, ok)
        mi, _, si = residual(i2g.astype(np.float32), ok)
        mo, mi = mo / so, mi / si
        diff = mo - mi
        sp = lambda m: float(np.nanpercentile(m, 95) - np.nanpercentile(m, 5))  # noqa: E731
        both = np.isfinite(mo) & np.isfinite(mi)
        corr = float(np.corrcoef(mo[both][::7], mi[both][::7])[0, 1])
        ring = lambda m: float(np.nanmedian(m[near & ok]) - np.nanmedian(m[far & ok]))  # noqa: E731
        rows.append(f"{f}\t{so:.4g} / {si:.4g}\t{sp(mo):.2f} / {sp(mi):.2f} / {sp(diff):.2f}\t{corr:.2f}"
                    f"\t{ring(mo):+.2f} / {ring(mi):+.2f}")
        r, c0 = k // 2, (k % 2) * 4
        for j, (m, lab, cm) in enumerate(((mo, "наша сборка", "RdBu_r"), (mi, "i2d STScI", "RdBu_r"),
                                          (diff, "наша − i2d", "PuOr_r"))):
            ax = fig.add_subplot(gs[r, c0 + j])
            ax.imshow(m[::-1], cmap=cm, vmin=-0.8, vmax=0.8)
            ax.set_title(f"{f}: {lab}" + (f" (σ {so:.3g})" if j == 0 else f" (σ {si:.3g})" if j == 1 else ""), fontsize=10)
            ax.axis("off")
    # наклоны экспозиций по детекторам
    for j, f in enumerate(filters):
        ax = fig.add_subplot(gs[len(filters) // 2, j if j < 3 else j + 1])
        pts = [r for r in slopes if r["фильтр"] == f]
        for r in pts:
            ax.scatter(float(r["наклон x, σ/1000 px"]), float(r["наклон y, σ/1000 px"]),
                       c=DET_COLORS.get(r["детектор"], "k"), s=40)
            ax.annotate(r["позиция дизеринга"], (float(r["наклон x, σ/1000 px"]), float(r["наклон y, σ/1000 px"])),
                        fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.grid(alpha=0.3)
        ax.set_title(f"{f}: наклон экспозиций, σ/1000 px (x, y)", fontsize=10)
        if j == 0:
            for d, col in DET_COLORS.items():
                ax.scatter([], [], c=col, label=d)
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("Перепад фона: наша сборка против i2d STScI (небо − медиана, сглажено 24 px, σ; шкала ±0,8σ) и наклоны "
                 "экспозиций относительно соседей (цифра — позиция дизеринга)", fontsize=13)
    fig.savefig(out / "14_перепад_наш_или_в_данных.png", dpi=45, bbox_inches="tight"); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / "14_наш_и_i2d.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
