"""3k, п.2, проверка 1: волны фона по y — общие для усилителей детектора (шум считывания 1/f) или нет,
и уходят ли они при более широком окне destripe (диагноз, не лечение).

Для выбранных экспозиций: destripe с окнами WINDOWS, проекция на сетку i2d, остаток относительно стека phase-3j
r = (e − S)·N/(N − 1) на свободном небе, константа на усилитель вычтена (медиана), медиана по полосам BIN строк
детектора — отдельно по усилителям. Лист 17: строки — экспозиции, столбцы — окна, линии — усилители.
Таблица: корреляция профилей усилителей между собой и размах волн (5–95%) по окнам.

Запуск: uv run python scripts/amp_waves_3k.py <папка> [фильтр] [экспозиции через запятую]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from skyforge.config import Config
from skyforge.fetch import level2_files, require_products
from skyforge.level2 import (_bbox_on_grid, _rotated, _wcs, destripe_frame, load_exposure, mask_to_frame, project,
                             source_mask_grid)
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BIN = 64
WINDOWS = (201, 601, 1001)
DEFAULT = "jw02733001001_02103_00002_nrcb2,jw02733001001_02103_00002_nrcb4,jw02733001001_02103_00002_nrcb1"


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    f = sys.argv[2] if len(sys.argv) > 2 else "F187N"
    names = (sys.argv[3] if len(sys.argv) > 3 else DEFAULT).split(",")
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase")
    c = cfg.level2
    l3, l2 = require_products(cfg, P, level3=True), level2_files(cfg, P)
    with fits.open(P.abs(l3[f].path), memmap=True) as h:
        i2d = np.array(h["SCI"].data, dtype=np.float32)
        if "WHT" in h:
            i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
        gw = _wcs(h["SCI"].header)
    shape = i2d.shape
    scale = float(np.mean(proj_plane_pixel_scales(gw))) * 3600
    src = source_mask_grid(i2d, scale, cfg)
    free = np.isfinite(i2d) & ~src
    cmp_dir = P.work_root / "ngc3132" / "compare_3k"
    with fits.open(cmp_dir / f"{f}_per_amp_phase3j.fits", memmap=True) as h:
        S = np.array(h["SCI"].data, dtype=np.float32)
        N = np.array(h["NEXP"].data, dtype=np.float32)
    ok_s = free & np.isfinite(S)
    sig = float(1.4826 * np.median(np.abs(S[ok_s] - np.median(S[ok_s]))))
    meta = json.loads((cmp_dir / "meta_phase3j.json").read_text(encoding="utf-8"))
    astro = {r["экспозиция"]: r for r in meta["filters"][f].get("astrometry", [])}
    files = {p.name.replace("_cal.fits", ""): p for p in l2[f]}
    fig, ax = plt.subplots(len(names), len(WINDOWS), figsize=(6 * len(WINDOWS), 4.2 * len(names)), squeeze=False, sharex=True)
    rows = ["экспозиция\tокно destripe\tразмах волн 5–95% по усилителям, σ стека\tкорреляция профилей усилителей (средняя по парам)"]
    for i, name in enumerate(names):
        a = astro.get(name, {})
        sci, _, fw, prim = load_exposure(files[name], c.dq_bad, c.weights)
        if a.get("применено_dx") is not None:
            fw = _rotated(fw, np.array([a["применено_dx"], a["применено_dy"]]), math.radians(a.get("применён_поворот_град") or 0.0))
        mask = mask_to_frame(src, gw, fw, sci.shape, c.mask_grow_px)
        box = _bbox_on_grid(fw, sci.shape, gw, shape)
        y0, y1, x0, x1 = box
        sub = gw.slice((slice(y0, y1), slice(x0, x1)))
        aw = sci.shape[1] // max(1, c.amplifiers)
        amp_img = np.minimum(np.arange(sci.shape[1]) // aw, c.amplifiers - 1)[None, :].repeat(sci.shape[0], 0)
        amp, _ = reproject_interp((amp_img.astype(np.float32), fw), sub, shape_out=(y1 - y0, x1 - x0), order="nearest-neighbor")
        rows_img = np.arange(sci.shape[0], dtype=np.float32)[:, None].repeat(sci.shape[1], 1)
        yd, _ = reproject_interp((rows_img, fw), sub, shape_out=(y1 - y0, x1 - x0), order="bilinear")
        n = N[y0:y1, x0:x1]
        for j, w in enumerate(WINDOWS):
            clean, _ = destripe_frame(sci, mask, c.model_copy(update={"window": w}))
            e = project(clean, fw, gw, box)
            r = (e - S[y0:y1, x0:x1]) * n / np.maximum(n - 1, 1) / sig
            base = np.isfinite(r) & free[y0:y1, x0:x1] & (n >= 2) & np.isfinite(yd) & np.isfinite(amp)
            profs = {}
            for k in range(c.amplifiers):
                m = base & (amp == k)
                if m.sum() < 5000:
                    continue
                rv = r[m] - np.median(r[m])
                yb = (yd[m] // BIN).astype(int)
                prof = np.full(int(math.ceil(sci.shape[0] / BIN)), np.nan)
                for b in np.unique(yb):
                    v = rv[yb == b]
                    if v.size >= 500 and 0 <= b < prof.size:
                        prof[b] = float(np.median(v))
                profs[k] = prof
                ax[i, j].plot((np.arange(prof.size) + 0.5) * BIN, prof, "-", lw=1.2, label=f"усилитель {k + 1}")
            spread = [float(np.nanpercentile(p, 95) - np.nanpercentile(p, 5)) for p in profs.values()]
            ks = list(profs)
            cors = []
            for u in range(len(ks)):
                for v in range(u + 1, len(ks)):
                    pa, pb = profs[ks[u]], profs[ks[v]]
                    both = np.isfinite(pa) & np.isfinite(pb)
                    if both.sum() >= 8:
                        cors.append(float(np.corrcoef(pa[both], pb[both])[0, 1]))
            cr = float(np.mean(cors)) if cors else float("nan")
            rows.append(f"{name}\t{w}\t" + ", ".join(f"{s:.2f}" for s in spread) + f"\t{cr:.2f}")
            ax[i, j].set_title(f"{prim.get('DETECTOR')} поз. {prim.get('PATT_NUM')}, окно {w}: корр. {cr:.2f}", fontsize=10)
            ax[i, j].axhline(0, color="k", lw=0.5); ax[i, j].set_ylim(-1.5, 1.5); ax[i, j].grid(alpha=0.3)
            print(rows[-1], flush=True)
    for a_ in ax[-1]:
        a_.set_xlabel("строка детектора")
    for a_ in ax[:, 0]:
        a_.set_ylabel("экспозиция − стек, σ стека")
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{f}: волны фона вдоль y по усилителям (медиана по {BIN} строк), destripe с разными окнами")
    fig.tight_layout(); fig.savefig(out / f"17_{f}_волны_по_усилителям.png", dpi=60); plt.close(fig)
    (out / f"17_{f}_волны_по_усилителям.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
