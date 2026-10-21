"""3б, шаг 3: перебор растяжки и цвета showcase Hubble — Heritage 1998 в первой клетке, варианты целиком,
кропы center и edge при 100%. Варианты — как в showcase_variants_3i.py ("stretch.white=99.9"); "" — как в YAML.

Запуск: uv run python scripts/sweep_3b.py <папка> <имя> <вариант> [<вариант> ...]
Пишет <имя>.png, PNG вариантов в <папка>/<имя>/ и таблицу: уровень неба (медиана яркости пустого неба — маска неба), доля
выжженных пикселей туманности (канал ≥ 254).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import (load_input, load_nebula_sd, load_saturated_stars, load_sky_mask, load_star_weight,
                            load_star_weight_map)
from skyforge.compose import base_color, tail_color
from skyforge.config import Config
from skyforge.cosmics import fill_gaps_linear
from skyforge.finish import finish_array, finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink, to_display, to_uint8
from skyforge.skymodel import apply_sky_poly
from skyforge.stretch import load_masks, stretch_array

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from clean_report_3b import reference_on_grid  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def render(c: Config, P: Paths, arrays: dict, ctx: dict, gains: dict | None = None) -> np.ndarray:
    """gains {фильтр: w} — только превью вкуса (лист «вес F658N»): слой над уровнем неба × w, небо на месте.
    color.weights при смешивании average сокращается, если в канале один фильтр (Heritage); у фильтра, который
    входит в канал вместе с другим (жёлтый F360M у Sh2-305 — в R и G), вес меняет канал."""
    a = {f: fill_gaps_linear(v, c.clean.fill_gaps_px)[0] for f, v in arrays.items()} if c.clean.fill_gaps_px else dict(arrays)
    a, _, _ = apply_sky_poly(c, P, a)
    masks = load_masks(c, P)
    layers = {f: stretch_array(v, c.stretch.for_filter(f), masks=masks, field=ctx["field"])[0] for f, v in a.items()}
    lvl = (c.stretch.sky_level or 0) / 255
    for f, g in (gains or {}).items():
        L = layers[f]
        layers[f] = np.where(L > lvl, np.clip(lvl + g * (L - lvl), 0, 1), L).astype(np.float32)
    b, _ = base_color(c, layers, ctx["sky"], ctx["sd"], field=ctx["field"])
    sm = load_star_weight(c, P)
    sw = load_star_weight_map(c, P) if sm is None and c.color.shadows.outside > 0 and c.color.shadows.halo_k > 0 else None
    comp = tail_color(c, b, layers, sm, ctx["sd"], load_saturated_stars(c, P), sw)
    done = finish_array(comp, c.finish, *ctx["fin"])
    return to_display(to_uint8(done)).astype(np.float32) / 255           # картинка: строка 0 — верх


def main() -> None:
    out, name, specs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    crop_names = ("center", "edge")
    if specs and specs[0].startswith("--crops="):              # --crops=edge — только эти кропы
        crop_names, specs = tuple(specs[0].split("=", 1)[1].split(",")), specs[1:]
    (out / name).mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/ngc3132hst/showcase.yaml", "showcase")
    arrays, _ = load_input(base, P)
    _, wcs, _ = load_aligned(base, P)
    ctx = {"field": stats_field(base, P, list(arrays.values())), "sky": load_sky_mask(base, P),
           "sd": load_nebula_sd(base, P), "fin": finish_masks(base, P)}
    shape = next(iter(arrays.values())).shape
    ref, _ = reference_on_grid(str(P.abs(base.reference_image)), wcs, shape)
    ref = ref[::-1]
    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}
    H, W = shape
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    sky_img = (load_sky_mask(base, P) & full)[::-1]                        # без пустых углов вне поля
    neb_img = (load_nebula_sd(base, P) > 0)[::-1]

    def cut(img, c):
        s = c.size // base.bin
        cx, cy = int(round(c.x / base.bin)), int(round(c.y / base.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    results = [("Heritage 1998", ref)]
    rows = ["вариант\tнебо (медиана яркости пустого неба), уровни\tвыжжено в туманности (канал ≥ 254), %"
            "\tнебо R / G / B, уровни\tв нуле на небе R / G / B, %"]
    for spec in specs:
        gains = {}
        if spec.startswith("gain."):                           # gain.F658N=1.3 — превью вкуса, не параметр конфига
            k, v = spec[5:].split("=", 1)
            gains, c = {k.upper(): float(v)}, base
        else:
            c = apply(base, spec)
        img = render(c, P, arrays, ctx, gains)
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / name / f"{(spec or 'yaml').replace(';', '_')}.png")
        lab = spec or "как в YAML"
        results.append((lab, img))
        sk = img[sky_img]
        rows.append(f"{lab}\t{np.median((img @ LUMA)[sky_img]) * 255:.1f}\t{(img[neb_img].max(-1) >= 254 / 255).mean() * 100:.2f}"
                    f"\t{' / '.join(f'{np.median(sk[:, k]) * 255:.1f}' for k in range(3))}"
                    f"\t{' / '.join(f'{(sk[:, k] <= 0).mean() * 100:.2f}' for k in range(3))}")
        print(f"  готово: {lab}", flush=True)
    tiles, labels = [], []
    for lab, im in results:
        tiles.append(shrink(im, 700)[::-1]); labels.append(f"{lab} · целиком")
    for cn in crop_names:
        for lab, im in results:
            tiles.append(cut(im, crops[cn])[::-1]); labels.append(f"{cn} · {lab} · 100%")
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(results), tile_inches=3.6, interpolation="nearest",
                  title=f"Hubble showcase: {name} — Heritage 1998 первым (на нашей сетке по AVM)")
    (out / f"{name}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
