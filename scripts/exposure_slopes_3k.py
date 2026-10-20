"""3k, п.2, проверка 1: наклон каждой экспозиции относительно соседей по перекрытиям, без пересборки.

Путь сборки повторяется до сведения уровней включительно (та же астрометрия из meta.json, destripe, проекция,
константа на экспозицию и усилитель); стек — готовая сборка. Остаток экспозиции относительно соседей:
r = (e − S)·N/(N − 1) (S — стек, N — число экспозиций в пикселе; e входит в S с весом ≈ 1/N). На свободном
небе (маска сборки) и где N ≥ 2 к r подгоняется робастная плоскость a + b·x + c·y (fit_plane, как в 3e).
Наклон — в σ неба стека на 1000 px сетки. Пишет таблицу по экспозициям и сводку по детекторам и позициям.

Запуск: uv run python scripts/exposure_slopes_3k.py <папка> [фильтры через запятую]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp

from skyforge.background import fit_plane
from skyforge.config import Config
from skyforge.fetch import level2_files, require_products
from skyforge.level2 import (_bbox_on_grid, _rotated, _wcs, destripe_frame, load_exposure, mask_to_frame,
                             match_levels, match_levels_per_amp, project, source_mask_grid)
from skyforge.paths import Paths

MAX_POINTS = 200_000


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    c = cfg.level2
    l3 = require_products(cfg, P, level3=True)
    l2 = level2_files(cfg, P)
    bd = P.work_root / "ngc3132" / "level2"
    meta = json.loads((bd / "meta.json").read_text(encoding="utf-8"))
    want = sys.argv[2].split(",") if len(sys.argv) > 2 else list(l2)
    rows = ["фильтр\tэкспозиция\tдетектор\tпозиция дизеринга\tнаклон x, σ/1000 px\tнаклон y, σ/1000 px"
            "\tсредний уровень, σ\tпикселей"]
    for f in want:
        with fits.open(P.abs(l3[f].path), memmap=True) as h:
            i2d = np.array(h["SCI"].data, dtype=np.float32)
            if "WHT" in h:
                i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
            grid_wcs = _wcs(h["SCI"].header)
        shape = i2d.shape
        scale = float(np.mean(proj_plane_pixel_scales(grid_wcs))) * 3600
        src_grid = source_mask_grid(i2d, scale, cfg)
        free_grid = np.isfinite(i2d) & ~src_grid
        fm = meta["filters"][f]
        with fits.open(bd / fm["file"], memmap=True) as h:
            S = np.array(h["SCI"].data, dtype=np.float32)
            N = np.array(h["NEXP"].data, dtype=np.float32)
        okS = free_grid & np.isfinite(S)
        sig = float(1.4826 * np.median(np.abs(S[okS] - np.median(S[okS]))))
        astro = {r["экспозиция"]: r for r in fm.get("astrometry", [])}
        # 3k п.2(б): прямая по y заменена кривой (level2.match_curve_y) и выключена — сборка без наклонов
        slopes_b = None
        exps, info = [], []
        for p in l2[f]:
            name = p.name.replace("_cal.fits", "")
            a = astro.get(name, {})
            if a.get("исключена"):
                continue
            sci, _, fw, prim = load_exposure(p, c.dq_bad, c.weights)
            if a.get("применено_dx") is not None:
                fw = _rotated(fw, np.array([a["применено_dx"], a["применено_dy"]]),
                              math.radians(a.get("применён_поворот_град") or 0.0))
            mask = mask_to_frame(src_grid, grid_wcs, fw, sci.shape, c.mask_grow_px)
            clean, _ = destripe_frame(sci, mask, c)
            box = _bbox_on_grid(fw, sci.shape, grid_wcs, shape)
            if box is None:
                continue
            e = {"name": name, "box": box, "data": project(clean, fw, grid_wcs, box)}
            if c.match_per_amp:
                aw = sci.shape[1] // max(1, c.amplifiers)
                amp_img = np.minimum(np.arange(sci.shape[1]) // aw, c.amplifiers - 1)[None, :].repeat(sci.shape[0], 0)
                y0, y1, x0, x1 = box
                ai, _ = reproject_interp((amp_img.astype(np.float32), fw), grid_wcs.slice((slice(y0, y1), slice(x0, x1))),
                                         shape_out=(y1 - y0, x1 - x0), order="nearest-neighbor")
                e["amp"] = np.where(np.isfinite(ai), ai, -1).astype(np.int8)
            if slopes_b is not None:  # сборка 3k (б): наклон по y детектора вычитается до констант, как в сборке
                yd = ((np.arange(sci.shape[0], dtype=np.float32) - sci.shape[0] / 2) / 1000)[:, None].repeat(sci.shape[1], 1)
                y0, y1, x0, x1 = box
                ydet, _ = reproject_interp((yd, fw), grid_wcs.slice((slice(y0, y1), slice(x0, x1))),
                                           shape_out=(y1 - y0, x1 - x0), order="bilinear")
                e["data"] = (e["data"] - np.float32(slopes_b[len(exps)]) * ydet.astype(np.float32)).astype(np.float32)
            exps.append(e)
            info.append((prim.get("DETECTOR"), prim.get("PATT_NUM")))
            print(f"  {f}: {name} загружена", flush=True)
        if c.match_levels and len(exps) > 1:
            if c.match_per_amp:
                match_levels_per_amp(exps, free_grid)
            else:
                lv, _ = match_levels(exps, free_grid)
                for e, v in zip(exps, lv):
                    e["data"] -= np.float32(v)
        for e, (det, pos) in zip(exps, info):
            y0, y1, x0, x1 = e["box"]
            n = N[y0:y1, x0:x1]
            r = (e["data"] - S[y0:y1, x0:x1]) * n / np.maximum(n - 1, 1)
            ok = free_grid[y0:y1, x0:x1] & np.isfinite(r) & (n >= 2)
            yy, xx = np.nonzero(ok)
            if yy.size < 5000:
                rows.append(f"{f}\t{e['name']}\t{det}\t{pos}\t—\t—\t—\t{yy.size}")
                continue
            step = max(1, yy.size // MAX_POINTS)
            yy, xx = yy[::step], xx[::step]
            v = r[yy, xx]
            coef = fit_plane((xx + x0).astype(np.float64), (yy + y0).astype(np.float64), v.astype(np.float64))
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            mid = coef[0] + coef[1] * cx + coef[2] * cy
            rows.append(f"{f}\t{e['name']}\t{det}\t{pos}\t{coef[1] * 1000 / sig:+.3f}\t{coef[2] * 1000 / sig:+.3f}"
                        f"\t{mid / sig:+.3f}\t{int(ok.sum())}")
        del exps
        (out / "14_наклоны_экспозиций.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        print("\n".join(r for r in rows if r.startswith(f)), flush=True)


if __name__ == "__main__":
    main()
