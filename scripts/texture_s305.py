"""Фаза 4, шаг 5: плотность зелёных волокон Sh2-305 — перебор (bin 4), Тевено первой строкой.

Лист 60 (по листу на каждый local_contrast.amount): строки — варианты color.shadows.value × color.saturation,
столбцы — целиком, filaments, globule, bubble при 100%. Лист 61: вес F335M в смешивании average 0,7 против 1,0.
Меры (таблица и подписи):
- σ волокон — разброс яркости (уровни) в кропе filaments только по маске туманности, % — от варианта как в YAML;
- чистый зелёный — доля пикселей маски туманности, где G > 1,5·max(R, B) (та же доля в кропе filaments);
- выжжено — доля пикселей маски туманности с каналом ≥ 254; нули неба R/G/B; зелень звёзд неба G − max(R, B).

Запуск: uv run python scripts/texture_s305.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.finish import finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from faint_s305 import thevenot_on_grid  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
CROPS = ("filaments", "globule", "bubble")


def green_share(img: np.ndarray, m: np.ndarray) -> float:
    p = img[m]
    return float((p[:, 1] > 1.5 * np.maximum(p[:, 0], p[:, 2])).mean() * 100)


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    arrays, _ = load_input(base, P)
    sky, sd = load_sky_mask(base, P), load_nebula_sd(base, P)
    field = stats_field(base, P, list(arrays.values()))
    ctx = {"field": field, "sky": sky, "sd": sd, "fin": finish_masks(base, P)}
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    sky_img, neb_img = (sky & full)[::-1], (sd > 0)[::-1]                # картинка: строка 0 — верх
    star_img = ((sw > 0.5) & (sd < 0))[::-1]
    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}

    def cut(img, c):                                                      # x, y кропа — на картинке (y вниз), как faint_s305
        s = max(16, c.size // base.bin)
        cx, cy = int(round(c.x / base.bin)), int(round(c.y / base.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    def measures(img, with_sky=True):
        lum = img @ LUMA * 255
        fm = cut(neb_img, crops["filaments"])
        r = {"sigma": float(cut(lum, crops["filaments"])[fm].std()),
             "green": green_share(img, neb_img),
             "green_fil": green_share(cut(img, crops["filaments"]), fm),
             "burned": float((img[neb_img].max(-1) >= 254 / 255).mean() * 100)}
        if with_sky:
            sk, st = img[sky_img] * 255, img[star_img] * 255
            r["zeros"] = [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)]
            r["sky"] = [float(np.median(sk[:, i])) for i in range(3)]
            r["star_green"] = float(np.median(st[:, 1] - np.maximum(st[:, 0], st[:, 2])))
        return r

    yaml_spec = (f"color.shadows.value={base.color.shadows.value:g};color.saturation={base.color.saturation:g};"
                 f"finish.local_contrast.amount={base.finish.local_contrast.amount:g}")
    grid = {lc: [f"color.shadows.value={sh};color.saturation={sa};finish.local_contrast.amount={lc}"
                 for sh in (2, 3, 4) for sa in (1.0, 1.2)] for lc in (0.3, 0.6)}
    weight = ["", "color.weights.F335M=0.7"]
    cache: dict[str, tuple[np.ndarray, dict]] = {}

    def run(spec):
        key = spec or yaml_spec
        if key == yaml_spec and yaml_spec in cache:
            return cache[yaml_spec]
        if key not in cache:
            img = render(apply(base, spec), P, arrays, ctx)
            Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / "варианты" / f"{key.replace(';', '_')}.png")
            cache[key] = (img, measures(img))
            print(f"  готово: {key}", flush=True)
        return cache[key]

    (out / "варианты").mkdir(exist_ok=True)
    ref_img, ref = run(yaml_spec)
    thev, treg = thevenot_on_grid(np.asarray(Image.fromarray((ref_img * 255).astype(np.uint8))
                                             .resize((W * base.bin, H * base.bin)), dtype=np.float32) / 255 @ LUMA,
                                  base.bin)
    thev = thev[:H, :W]
    tm = measures(thev, with_sky=False)

    def short(spec):
        if not spec:
            return "как в YAML (финал)"
        d = dict(p.split("=") for p in spec.split(";"))
        if "color.weights.F335M" in d:
            return f"вес F335M {d['color.weights.F335M']}"
        s = f"тени {d['color.shadows.value']} · насыщ. {float(d['color.saturation']):.1f}"
        return s + (" = финал" if spec == yaml_spec else "")

    rows = ["вариант\tσ волокон, уровни (% от финала)\tчистый зелёный, % маски\tчистый зелёный в filaments, %"
            "\tвыжжено, %\tнулей неба R/G/B, %\tнебо R/G/B\tзелень звёзд",
            f"Тевено (Commons, {treg})\t{tm['sigma']:.1f} ({tm['sigma'] / ref['sigma'] * 100:.0f}%)\t{tm['green']:.1f}"
            f"\t{tm['green_fil']:.1f}\t{tm['burned']:.2f}\t—\t—\t—"]

    def row(lab, m):
        rows.append(f"{lab}\t{m['sigma']:.1f} ({m['sigma'] / ref['sigma'] * 100:.0f}%)\t{m['green']:.1f}\t{m['green_fil']:.1f}"
                    f"\t{m['burned']:.2f}\t{' / '.join(f'{z:.2f}' for z in m['zeros'])}"
                    f"\t{' / '.join(f'{v:.0f}' for v in m['sky'])}\t{m['star_green']:+.1f}")

    def sheet(path, title, items):
        tiles, labels = [], []
        for lab, img, m in items:
            tiles.append(shrink(img, 700)[::-1])
            labels.append(f"{lab}\nчистый зелёный {m['green']:.0f}% · выжж. {m['burned']:.2f}%")
            for n in CROPS:
                tiles.append(cut(img, crops[n])[::-1])
                if n == "filaments":
                    labels.append(f"filaments 100%\nσ {m['sigma']:.1f} ({m['sigma'] / ref['sigma'] * 100:.0f}%) · "
                                  f"зелёный {m['green_fil']:.0f}%")
                elif n == "globule" and "zeros" in m:
                    labels.append(f"globule 100%\nнули неба {' / '.join(f'{z:.2f}' for z in m['zeros'])}%")
                else:
                    labels.append(f"{n} 100%")
        contact_sheet(tiles, labels, path, cols=4, tile_inches=3.6, interpolation="nearest", title=title)

    for lc, specs in grid.items():
        rows.append(f"— local_contrast.amount {lc}")
        items = [("Тевено", thev, tm)]
        for spec in specs:
            img, m = run(spec)
            row(short(spec), m)
            items.append((f"{short(spec)} · контраст {lc}", img, m))
        sheet(out / f"60_тени_насыщенность_контраст_{lc}.png",
              f"Sh2-305, шаг 5: тени × насыщенность при local_contrast {lc} (bin {base.bin}); Тевено первой строкой", items)
    rows.append("— вес F335M (average)")
    items = [("Тевено", thev, tm)]
    for spec in weight:
        img, m = run(spec)
        row(short(spec), m)
        items.append((short(spec), img, m))
    sheet(out / "61_вес_F335M.png",
          f"Sh2-305, шаг 5: вес F335M в average 0,7 против 1,0 (bin {base.bin}); Тевено первой строкой", items)
    (out / "60_61_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
