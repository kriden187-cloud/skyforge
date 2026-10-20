"""3k, п.1: цвет неба по областям NEXP — вклад шума отдельно от ступенек уровня.

Итог собирается дважды тем же кодом (base_color + tail_color, до finish): как есть и «без шума» — данные неба
каждого фильтра сглажены гауссом SMOOTH_PX до растяжки (точки растяжки — от исходных). Уровни и ход фона
(ступеньки п.2, ореол) в обеих сборках одни и те же, поэтому разность цвета — вклад пиксельного шума
(обрезка, носок, подъём теней, насыщенность от шума). Цвет пикселя — канал минус среднее трёх каналов, уровни из 255.

Состояния: phase-3j (процентиль black, без носка, подъём от нуля, нейтрализация в ноль) и текущий showcase.

Запуск: uv run python scripts/sky_color_split_3k.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned, stats_field
from skyforge.clean import load_input, load_nebula_sd, load_saturated_stars, load_sky_mask, load_star_weight
from skyforge.compose import base_color, tail_color
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.stretch import load_masks, stretch_array

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import nexp_maps, smooth_on  # noqa: E402

SMOOTH_PX = 6


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    c3j = (cfg.with_param("stretch.sky_level", None).with_param("stretch.black_mode", "percentile")
           .with_param("color.shadows.above_sky", False).with_param("color.background_keep_level", False))
    arrays, _ = load_input(cfg, P)
    masks = load_masks(cfg, P)
    field = stats_field(cfg, P, list(arrays.values()))
    _, wcs, _ = load_aligned(cfg, P)
    sky, sd = load_sky_mask(cfg, P), load_nebula_sd(cfg, P)
    stars, sat = load_star_weight(cfg, P), load_saturated_stars(cfg, P)
    ne = {f: ndimage.median_filter(n, 15) for f, n in nexp_maps(P, wcs, sky.shape).items()}
    combo = ne["F090W"] * 100 + ne["F356W"] * 10 + np.minimum(ne["F470N"], 9)
    regions = [k for k in np.unique(combo[sky]) if (sky & (combo == k)).sum() >= 5000]
    name = lambda k: f"{k // 100}/{(k // 10) % 10}/{k % 10}"

    def chroma_by_region(rgb):
        c = (rgb - rgb.mean(axis=-1, keepdims=True)) * 255
        return {k: np.median(c[sky & (combo == k)], axis=0) for k in regions}

    rows = []
    for label, c in (("phase-3j", c3j), ("носок", cfg)):
        layers = {f: stretch_array(a, c.stretch.for_filter(f), masks=masks, field=field)[0] for f, a in arrays.items()}
        flat = {}
        for f, a in arrays.items():  # без шума: данные неба сглажены до растяжки, точки растяжки — от исходных
            ok = sky & np.isfinite(a)
            sm = smooth_on(ok, np.nan_to_num(a), SMOOTH_PX)
            a0 = np.where(ok & np.isfinite(sm), sm, a).astype(np.float32)
            flat[f] = stretch_array(a0, c.stretch.for_filter(f), stats_from=a, masks=masks, field=field)[0]
        rgb = tail_color(c, base_color(c, layers, sky, sd, field=field)[0], layers, stars, sd, sat)
        rgb0 = tail_color(c, base_color(c, flat, sky, sd, field=field)[0], flat, stars, sd, sat)
        ch, ch0 = chroma_by_region(rgb), chroma_by_region(rgb0)
        ref = max(regions, key=lambda k: (sky & (combo == k)).sum())
        noise = {k: ch[k] - ch0[k] for k in regions}
        lvl = {k: np.median(rgb[sky & (combo == k)].mean(-1)) * 255 for k in regions}
        rows.append(f"## {label}: опорная область {name(ref)} (самая большая)")
        rows.append("область\tуровень неба\tцвет как есть R, G, B\tцвет без шума (только уровни)\tвклад шума\t"
                    "вклад шума минус у опорной")
        worst = np.zeros(3)
        for k in regions:
            d = noise[k] - noise[ref]
            worst = np.maximum(worst, np.abs(d))
            f3 = lambda v: ", ".join(f"{x:+.2f}" for x in v)
            rows.append(f"{name(k)}\t{lvl[k]:.2f}\t{f3(ch[k])}\t{f3(ch0[k])}\t{f3(noise[k])}\t{f3(d)}")
        n1 = [k for k in regions if k // 100 == 1]
        for k in n1:
            rows.append(f"NEXP1 {name(k)} против {name(ref)}: вклад шума в цвет "
                        + ", ".join(f"{x:+.2f}" for x in noise[k] - noise[ref])
                        + "; цвет как есть " + ", ".join(f"{x:+.2f}" for x in ch[k] - ch[ref]))
        rows.append(f"наибольшая |разница вклада шума| с опорной по всем областям: {', '.join(f'{x:.2f}' for x in worst)}")
        rows.append("")
        print("\n".join(rows[-len(regions) - 6:]))
    (out / "11_цвет_неба_шум_и_уровни.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
