"""3k, п.2(б): форма фона экспозиции вдоль оси y детектора — прямая или нет.

Путь сборки phase-3j до сведения уровней (константа на экспозицию и усилитель), стек — копия сборки phase-3j
(data/work/ngc3132/compare_3k/<фильтр>_per_amp_phase3j.fits). Остаток экспозиции относительно соседей
r = (e − S)·N/(N − 1) на свободном небе, медиана по полосам 64 строки детектора — отдельно вне маски исключения
наклонов (внешнее небо) и внутри (у туманности). Лист 16: по детектору на панель, линии — позиции дизеринга.

Запуск: uv run python scripts/exposure_profiles_3k.py <папка> [фильтр]
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
from skyforge.level2 import (_bbox_on_grid, _rotated, _wcs, destripe_frame, load_exposure, mask_to_frame,
                             match_levels_per_amp, project, reproject_mask, source_mask_grid)
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BIN = 64


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    f = sys.argv[2] if len(sys.argv) > 2 else "F187N"
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
    excl = reproject_mask(P.abs(cfg.stretch.sky_poly_exclude or "data/reference/ngc3132_slope_exclusion.fits"), gw, shape)
    cmp_dir = P.work_root / "ngc3132" / "compare_3k"
    with fits.open(cmp_dir / f"{f}_per_amp_phase3j.fits", memmap=True) as h:
        S = np.array(h["SCI"].data, dtype=np.float32)
        N = np.array(h["NEXP"].data, dtype=np.float32)
    ok_s = free & np.isfinite(S)
    sig = float(1.4826 * np.median(np.abs(S[ok_s] - np.median(S[ok_s]))))
    meta = json.loads((cmp_dir / "meta_phase3j.json").read_text(encoding="utf-8"))
    astro = {r["экспозиция"]: r for r in meta["filters"][f].get("astrometry", [])}
    exps, info = [], []
    for p in l2[f]:
        name = p.name.replace("_cal.fits", "")
        a = astro.get(name, {})
        if a.get("исключена"):
            continue
        sci, _, fw, prim = load_exposure(p, c.dq_bad, c.weights)
        if a.get("применено_dx") is not None:
            fw = _rotated(fw, np.array([a["применено_dx"], a["применено_dy"]]), math.radians(a.get("применён_поворот_град") or 0.0))
        clean, _ = destripe_frame(sci, mask_to_frame(src, gw, fw, sci.shape, c.mask_grow_px), c)
        box = _bbox_on_grid(fw, sci.shape, gw, shape)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        sub = gw.slice((slice(y0, y1), slice(x0, x1)))
        aw = sci.shape[1] // max(1, c.amplifiers)
        amp_img = np.minimum(np.arange(sci.shape[1]) // aw, c.amplifiers - 1)[None, :].repeat(sci.shape[0], 0)
        ai, _ = reproject_interp((amp_img.astype(np.float32), fw), sub, shape_out=(y1 - y0, x1 - x0), order="nearest-neighbor")
        rows = np.arange(sci.shape[0], dtype=np.float32)[:, None].repeat(sci.shape[1], 1)
        yd, _ = reproject_interp((rows, fw), sub, shape_out=(y1 - y0, x1 - x0), order="bilinear")
        exps.append({"box": box, "data": project(clean, fw, gw, box), "amp": np.where(np.isfinite(ai), ai, -1).astype(np.int8),
                     "yd": yd.astype(np.float32)})
        info.append((prim.get("DETECTOR"), prim.get("PATT_NUM")))
        print(f"  {name}", flush=True)
    match_levels_per_amp(exps, free)
    dets = sorted({d for d, _ in info})
    fig, ax = plt.subplots(2, len(dets), figsize=(5.5 * len(dets), 9), sharex=True, squeeze=False)
    for e, (det, pos) in zip(exps, info):
        y0, y1, x0, x1 = e["box"]
        n = N[y0:y1, x0:x1]
        r = (e["data"] - S[y0:y1, x0:x1]) * n / np.maximum(n - 1, 1) / sig
        base = np.isfinite(r) & free[y0:y1, x0:x1] & (n >= 2) & np.isfinite(e["yd"])
        col = dets.index(det)
        for row, (m, lab) in enumerate(((base & ~excl[y0:y1, x0:x1], "внешнее небо"), (base & excl[y0:y1, x0:x1], "у туманности"))):
            yb = (e["yd"][m] // BIN).astype(int)
            rv = r[m]
            xs, ys = [], []
            for b in np.unique(yb):
                v = rv[yb == b]
                if v.size >= 2000:
                    xs.append((b + 0.5) * BIN)
                    ys.append(float(np.median(v)))
            ax[row, col].plot(xs, ys, "-", lw=1, label=f"поз. {pos}")
            ax[row, col].set_title(f"{f} {det}: {lab}")
    for a_ in ax.ravel():
        a_.axhline(0, color="k", lw=0.5); a_.set_ylim(-1.5, 1.5); a_.grid(alpha=0.3)
        a_.set_xlabel("строка детектора"); a_.set_ylabel("экспозиция − соседи, σ неба стека")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle(f"{f}: фон экспозиции вдоль оси y детектора относительно соседей (медиана по {BIN} строк), сборка phase-3j")
    fig.tight_layout(); fig.savefig(out / f"16_{f}_профиль_фона_по_y.png", dpi=60); plt.close(fig)


if __name__ == "__main__":
    main()
