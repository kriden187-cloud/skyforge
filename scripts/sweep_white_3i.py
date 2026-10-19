"""3i: перебор белой точки F090W и F187N в showcase (выборка без звёзд), NASA рядом, кропы inner и core.

Запуск: uv run python scripts/sweep_white_3i.py out/ngc3132/проверка_3i [99.6,99.8,99.9,99.95]
Пишет лист 11_white_F090W_F187N.png и таблицу долей обрезанных пикселей по каналам.
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import SimilarityTransform, warp

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask, load_star_weight
from skyforge.compose import base_color, tail_color
from skyforge.config import Config
from skyforge.finish import finish_masks, finish_region
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, shrink
from skyforge.stretch import load_masks, stretch_array


def nasa_on_grid(ours: np.ndarray, ref_path: str, ours_pts: list, ref_guess: list) -> np.ndarray:
    """Референс NASA на нашей сетке: подобие по двум звёздам (уточнение по максимуму рядом с подсказкой)."""
    ref = np.asarray(Image.open(ref_path).convert("RGB")).astype(np.float32) / 255

    def peak(img, x, y, r):
        g = ndimage.gaussian_filter(img.mean(-1), 2)
        y0, x0 = max(0, y - r), max(0, x - r)
        win = g[y0:y + r, x0:x + r]
        j, i = np.unravel_index(np.argmax(win), win.shape)
        return x0 + i, y0 + j

    o = [peak(ours, x, y, 30) for x, y in ours_pts]
    n = [peak(ref, x, y, 80) for x, y in ref_guess]
    t = SimilarityTransform()
    t.estimate(np.array(o, float), np.array(n, float))
    return warp(ref, t, output_shape=ours.shape[:2], order=1)[::-1]   # сетка: строка 0 — низ


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    values = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "99.6,99.8,99.9,99.95").split(",")]
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(cfg, P)
    masks = load_masks(cfg, P)
    field = stats_field(cfg, P, list(arrays.values()))
    sd, sky, sm = load_nebula_sd(cfg, P), load_sky_mask(cfg, P), load_star_weight(cfg, P)
    sw_, sd_, origin = finish_masks(cfg, P)
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    final = sorted(P.out_root.glob("ngc3132/*_styles"))[-1] / "showcase" / "ngc3132_showcase.png"
    ours = np.asarray(Image.open(final).convert("RGB")).astype(np.float32) / 255
    nasa = nasa_on_grid(ours, str(P.abs(cfg.reference_image)), [(1133, 876), (1901, 1701)], [(1491, 1313), (2500, 2388)])

    tile = 600
    thumbs, tlabels = [shrink(nasa, tile)], ["NASA"]
    crop_imgs = {"inner": [], "core": []}
    rows = []
    for v in values:
        c = cfg.with_param("stretch.per_filter.F090W.white", v).with_param("stretch.per_filter.F187N.white", v)
        layers = {f: stretch_array(a, c.stretch.for_filter(f), masks=masks, field=field)[0] for f, a in arrays.items()}
        b, _ = base_color(c, layers, sky, sd, field=field)
        comp = tail_color(c, b, layers, sm, sd)
        thumbs.append(shrink(comp, tile))
        tlabels.append(f"F090W, F187N white {v:g}")
        row = [v]
        for cn in ("inner", "core"):
            cr = crops[cn]
            sl = display_crop(comp.shape, cr.x, cr.y, cr.size, 1)
            fin = finish_region(comp, c.finish, sw_, sd_, origin, sl)
            frac = [(fin[..., k] >= 0.99).mean() * 100 for k in range(3)]
            row += frac
            crop_imgs[cn].append((fin, f"{cn} · white {v:g} · R {frac[0]:.1f}% G {frac[1]:.1f}% B {frac[2]:.1f}% ≥0,99"))
        rows.append(row)
        print(f"white {v:g}: inner R/G/B ≥ 0,99: {row[1]:.1f} / {row[2]:.1f} / {row[3]:.1f}%; "
              f"core: {row[4]:.1f} / {row[5]:.1f} / {row[6]:.1f}%")
    contact_sheet(thumbs, tlabels, out / "11_white_F090W_F187N_целиком.png", cols=len(thumbs), tile_inches=3.6,
                  title="showcase: белая точка F090W и F187N по туманности без звёзд, NASA первым")
    imgs, labels = [], []
    for cn in ("inner", "core"):
        cr = crops[cn]
        sl = display_crop(ours.shape, cr.x, cr.y, cr.size, 1)
        imgs.append(nasa[sl])
        labels.append(f"{cn} · NASA")
        for im, lab in crop_imgs[cn]:
            imgs.append(im)
            labels.append(lab)
    contact_sheet(imgs, labels, out / "12_white_F090W_F187N_inner_core.png", cols=len(values) + 1, tile_inches=3.8,
                  interpolation="nearest", title="inner и core 100%: доля пикселей с каналом ≥ 0,99 после finish")
    # ×3 вокруг ядра центральной звезды (60 px)
    z_imgs, z_lab = [], []
    cr = crops["core"]
    sl = display_crop(ours.shape, cr.x, cr.y, cr.size, 1)
    h = (sl[0].stop - sl[0].start) // 2
    zz = (slice(h - 30, h + 30), slice(h - 30, h + 30))
    z_imgs.append(np.kron(nasa[sl][zz], np.ones((3, 3, 1))))
    z_lab.append("NASA, 60 px ×3")
    for im, lab in crop_imgs["core"]:
        z_imgs.append(np.kron(np.clip(im[zz], 0, 1), np.ones((3, 3, 1))))
        z_lab.append(lab.split(" · R")[0] + ", ×3")
    contact_sheet(z_imgs, z_lab, out / "13_white_core_x3.png", cols=len(values) + 1, tile_inches=3.2,
                  interpolation="nearest", title="Ядро центральной звезды, 60 px ×3")
    (out / "white_F090W_F187N.tsv").write_text(
        "white\tinner_R\tinner_G\tinner_B\tcore_R\tcore_G\tcore_B\n"
        + "\n".join("\t".join(f"{x:.2f}" for x in r) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
