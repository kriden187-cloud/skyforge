"""3k, п.2 вариант 2: полосатость мозаики по строкам сетки (ось y детектора) — медиана свободного неба по полосам
BIN строк, в σ неба phase-3j. Сравнивает i2d STScI, сборку phase-3j и проверяемую сборку на сетке i2d фильтра.
Кривые по y не должны вносить своих полос (размах профиля не больше, чем у phase-3j).

Запуск: uv run python scripts/row_profile_3k.py <папка> <фильтр> <проверяемая сборка .fits> [метка]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales

from skyforge.config import Config
from skyforge.fetch import require_products
from skyforge.level2 import _wcs, source_mask_grid
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BIN = 64


def main() -> None:
    out, f, test = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else "проверяемая"
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase")
    l3 = require_products(cfg, P, level3=True)
    with fits.open(P.abs(l3[f].path)) as h:
        i2d = np.array(h["SCI"].data, np.float32)
        if "WHT" in h:
            i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
        gw = _wcs(h["SCI"].header)
    scale = float(np.mean(proj_plane_pixel_scales(gw))) * 3600
    free = np.isfinite(i2d) & ~source_mask_grid(i2d, scale, cfg)
    S = {"i2d STScI": i2d}
    for lab, p in (("phase-3j", P.work_root / "ngc3132" / "compare_3k" / f"{f}_phase3j.fits"), (label, test)):
        if not p.exists() and lab == "phase-3j":
            p = P.work_root / "ngc3132" / "compare_3k" / f"{f}_per_amp_phase3j.fits"
        with fits.open(p) as h:
            S[lab] = np.array(h["SCI"].data, np.float32)
    ok = free.copy()
    for a in S.values():
        ok &= np.isfinite(a)
    sig = float(1.4826 * np.median(np.abs(S["phase-3j"][ok] - np.median(S["phase-3j"][ok]))))
    H = ok.shape[0]
    fig, ax = plt.subplots(figsize=(12, 5))
    rows = ["сборка\tразмах профиля по строкам 5–95%, σ\tразброс профиля, σ"]
    for lab, a in S.items():
        v = a - np.median(a[ok])
        ys, prof = [], []
        for y0 in range(0, H, BIN):
            m = ok[y0:y0 + BIN]
            if m.sum() > 3000:
                prof.append(np.median(v[y0:y0 + BIN][m]) / sig)
                ys.append(y0 + BIN / 2)
        prof = np.array(prof)
        rows.append(f"{lab}\t{np.percentile(prof, 95) - np.percentile(prof, 5):.2f}\t{prof.std():.2f}")
        ax.plot(ys, prof, label=lab)
    ax.axhline(0, color="k", lw=0.5); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlabel("строка сетки i2d"); ax.set_ylabel("небо − медиана, σ неба phase-3j")
    ax.set_title(f"{f}: медиана свободного неба по полосам {BIN} строк")
    fig.tight_layout(); fig.savefig(out / f"19_{f}_профиль_по_строкам.png", dpi=70); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / f"19_{f}_профиль_по_строкам.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
