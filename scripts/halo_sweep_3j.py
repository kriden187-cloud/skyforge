"""3j: перебор одного параметра для внешнего ореола showcase — целиком с финишем, кроп edge, профиль к NASA.

Запуск: uv run python scripts/halo_sweep_3j.py <папка> <имя> <вариант> [<вариант> ...]
Вариант — как в showcase_variants_3i.py ("color.weights.F212N=0.2"); "" — showcase как в YAML.
Пишет <имя>.png (целиком + edge, NASA первым), <имя>_профиль.png и PNG вариантов в <папка>/<имя>/.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from PIL import Image

from skyforge.align import read_aligned_meta, stats_field, with_resolved_crops
from skyforge.clean import (load_input, load_nebula_sd, load_saturated_stars, load_sky_mask, load_star_weight,
                            load_star_weight_map)
from skyforge.compose import base_color, tail_color
from skyforge.config import Config
from skyforge.finish import finish_array, finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink, to_display, to_uint8
from skyforge.stretch import load_masks, stretch_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from halo_profile_3j import profile  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_white_3i import nasa_on_grid  # noqa: E402


def crop(img: np.ndarray, x: int, y: int, s: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(np.clip(x - s // 2, 0, w - s)), int(np.clip(y - s // 2, 0, h - s))
    return img[y0:y0 + s, x0:x0 + s]


def main() -> None:
    out, name, specs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    (out / name).mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(base, P)
    from skyforge.skymodel import apply_sky_poly

    arrays, _, _ = apply_sky_poly(base, P, arrays)   # 3k п.3: полином неба, как в этапе растяжки (если включён)
    sd, sky, sm = load_nebula_sd(base, P), load_sky_mask(base, P), load_star_weight(base, P)
    sw_, sd_, origin = finish_masks(base, P)
    field = stats_field(base, P, list(arrays.values()))
    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}
    results = []
    for spec in specs:
        c = apply(base, spec)
        masks = load_masks(c, P)
        layers = {f: stretch_array(a, c.stretch.for_filter(f), masks=masks, field=field)[0] for f, a in arrays.items()}
        b, _ = base_color(c, layers, sky, sd, field=field)
        star_w = load_star_weight_map(c, P) if sm is None else None
        done = finish_array(tail_color(c, b, layers, sm, sd, load_saturated_stars(c, P), star_w), c.finish, sw_, sd_, origin)
        img = to_display(to_uint8(done)).astype(np.float32) / 255           # картинка: строка 0 — верх
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / name / f"{(spec or 'yaml').replace(';', '_')}.png")
        results.append((spec or "как в YAML", img))
        print(f"  готово: {spec or 'как в YAML'}")
    nasa = nasa_on_grid(results[0][1], str(P.abs(base.reference_image)), [(1133, 876), (1901, 1701)],
                        [(1491, 1313), (2500, 2388)])[::-1]
    allimgs = [("NASA", nasa)] + results
    tiles = [shrink(im, 700)[::-1] for _, im in allimgs]
    e = crops["edge"]
    tiles += [crop(im, e.x, e.y, e.size)[::-1] for _, im in allimgs]
    labels = [f"{n}" for n, _ in allimgs] + [f"edge · {n}" for n, _ in allimgs]
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(allimgs), tile_inches=3.6, interpolation="nearest",
                  title=f"Внешний ореол: {name} — NASA первым, целиком и кроп edge")
    fig, ax = plt.subplots(figsize=(9, 5))
    rn, pn = profile(nasa)
    for lab, im in results:
        r, pr = profile(im)
        n = min(len(pn), len(pr))
        ax.plot(r[:n], pr[:n] / np.maximum(pn[:n], 1e-4), label=lab)
        vals = {rad: pr[int(rad // 10)] / max(pn[int(rad // 10)], 1e-4) for rad in (600, 800, 1000, 1200)}
        print(f"{lab}: наш/NASA " + ", ".join(f"{k}: {v:.2f}" for k, v in vals.items()))
    ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log"); ax.set_ylim(0.1, 5)
    ax.set_xlabel("радиус от центральной звезды, px"); ax.set_title("Яркость между лучами: наш / NASA")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out / f"{name}_профиль.png", dpi=90)


if __name__ == "__main__":
    main()
