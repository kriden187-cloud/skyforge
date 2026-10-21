"""Фаза 4, шаг 6: палитра с тёплым сдвигом и F182M (bin 4), Тевено первой колонкой — лист 64.

Колонки: Тевено, А (финал phase-4), Б (тёплый сдвиг, 4 фильтра), В (тёплый сдвиг + F182M синий; пятифильтровая
проба в отдельной рабочей папке — маски и кеши sh2305 не трогаются). Строки: целиком, filaments, globule, bubble,
star при 100%, sky ×5. Меры — на маске туманности и неба четырёхфильтровой сборки (одни пиксели у всех вариантов):
- разброс оттенка — круговая σ оттенка (градусы) в кропе filaments по маске, пиксели с цветностью > 20 уровней;
- чистый зелёный — доля маски туманности с G > 1,5·max(R, B); выжжено — канал ≥ 254 в маске;
- нули и медиана неба R/G/B; зелень звёзд неба G − max(R, B) — медиана, 75-й и 95-й процентили.

Запуск: uv run python scripts/palette_s305.py <папка> <конфиг пятифильтровой пробы (папка со showcase.yaml)>
"""

from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import load_aligned, stats_field, with_resolved_crops
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
WARM = {"F300M": "#40c0ff", "F335M": "#c0ff40", "F360M": "#ff9020", "F444W": "#ff2020"}
F182 = {"F182M": "#3050ff"}
ROWS = (("целиком", None, 1), ("filaments 100%", "filaments", 1), ("globule 100%", "globule", 1),
        ("bubble 100%", "bubble", 1), ("star 100%", "star", 1), ("sky ×5", "sky", 5))


def hue_deg(hexc: str) -> float:
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def spec(pal: dict[str, str]) -> str:
    return ";".join(f"color.overrides.{f}={c}" for f, c in pal.items())


def workspace(c: Config, P: Paths) -> dict:
    arrays, _ = load_input(c, P)
    _, wcs, _ = load_aligned(c, P)
    return {"arrays": arrays, "wcs": wcs,
            "ctx": {"field": stats_field(c, P, list(arrays.values())), "sky": load_sky_mask(c, P),
                    "sd": load_nebula_sd(c, P), "fin": finish_masks(c, P)}}


def main() -> None:
    out, probe = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base4 = Config.load("configs/sh2305/showcase.yaml", "showcase")
    base5 = Config.load(probe / "showcase.yaml", "showcase")
    # порядок «короче волна — холоднее»: оттенок (0° красный … 240° синий) растёт к коротким волнам
    order = [("F182M", F182["F182M"])] + [(f, WARM[f]) for f in ("F300M", "F335M", "F360M", "F444W")]
    hues = [hue_deg(c) for _, c in order]
    print("оттенки палитры В:", ", ".join(f"{f} {h:.0f}°" for (f, _), h in zip(order, hues)),
          "— порядок сохранён" if all(a > b for a, b in zip(hues, hues[1:])) else "— ПОРЯДОК НАРУШЕН")
    w4, w5 = workspace(base4, P), workspace(base5, P)
    H, W = w4["ctx"]["sd"].shape
    # пятифильтровая сетка → четырёхфильтровая: целый сдвиг по WCS (та же сетка F444W, другой обрез кадра)
    sky0 = w5["wcs"].pixel_to_world(0, 0)
    ox, oy = (float(v) for v in w4["wcs"].world_to_pixel(sky0))
    dx, dy = int(round(ox)), int(round(oy))
    print(f"сетка пробы {w5['ctx']['sd'].shape} → {(H, W)}, сдвиг (x {ox:.3f}, y {oy:.3f})")

    def to4(img5: np.ndarray) -> np.ndarray:                 # картинка пробы (строка 0 — верх) → картинка сетки 4
        g5 = img5[::-1]
        canvas = np.zeros((H, W, 3), np.float32)
        h5, w5_ = g5.shape[:2]
        ys, xs = max(0, -dy), max(0, -dx)
        yd, xd = max(0, dy), max(0, dx)
        h, w = min(H - yd, h5 - ys), min(W - xd, w5_ - xs)
        canvas[yd:yd + h, xd:xd + w] = g5[ys:ys + h, xs:xs + w]
        return canvas[::-1]

    sd, sky = w4["ctx"]["sd"], w4["ctx"]["sky"]
    full = np.logical_and.reduce([np.isfinite(a) for a in w4["arrays"].values()])
    # поле пятифильтровой пробы на сетке 4 (её кадр уже на 23 столбца): меры всех вариантов — только на общем поле
    cover = to4(np.ones((*w5["ctx"]["sd"].shape, 3), np.float32)[::-1])[..., 0] > 0
    print(f"вне поля пробы: {(~cover).mean() * 100:.2f}% кадра — в меры не входит")
    neb_img, sky_img = (sd > 0)[::-1] & cover, (sky & full)[::-1] & cover
    sw = np.load(clean_dir(base4, P) / "star_weight.npy")
    star_img = ((sw > 0.5) & (sd < 0))[::-1]
    crops = {k.name: k for k in with_resolved_crops(base4, P).report.crops}

    def cut(img, c):                                          # x, y кропа — на картинке (y вниз), как faint_s305
        s = max(16, c.size // base4.bin)
        cx, cy = int(round(c.x / base4.bin)), int(round(c.y / base4.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    def measures(img, with_sky=True):
        t, m = cut(img, crops["filaments"]), cut(neb_img, crops["filaments"])
        q = t[m]
        chroma = (q.max(-1) - q.min(-1)) * 255
        hsv = np.array([colorsys.rgb_to_hsv(*v) for v in q[chroma > 20]])
        ang = np.deg2rad(hsv[:, 0] * 360)
        R = np.hypot(np.sin(ang).mean(), np.cos(ang).mean())
        p = img[neb_img]
        r = {"hue_sd": float(np.rad2deg(np.sqrt(-2 * np.log(R)))),
             "hue": float(np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360),
             "chroma": float(np.median(chroma)),
             "green": float((p[:, 1] > 1.5 * np.maximum(p[:, 0], p[:, 2])).mean() * 100),
             "burned": float((p.max(-1) >= 254 / 255).mean() * 100)}
        if with_sky:
            sk, st = img[sky_img] * 255, img[star_img] * 255
            g = st[:, 1] - np.maximum(st[:, 0], st[:, 2])
            r |= {"zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                  "sky": [float(np.median(sk[:, i])) for i in range(3)],
                  "star": [float(np.percentile(g, q_)) for q_ in (50, 75, 95)]}
        return r

    variants = [("А · финал phase-4", base4, w4, ""),
                ("Б · тёплый сдвиг, 4 фильтра", base4, w4, spec(WARM)),
                ("В · тёплый сдвиг + F182M", base5, w5, spec(WARM | F182))]
    (out / "варианты").mkdir(exist_ok=True)
    results = []
    for lab, c, w, sp in variants:
        img = render(apply(c, sp), P, w["arrays"], w["ctx"])
        if w is w5:
            img = to4(img)
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / "варианты" / f"{lab.split(' ')[0]}.png")
        results.append((lab, img, measures(img)))
        print(f"  готово: {lab}", flush=True)
    ref = results[0][1]
    thev, treg = thevenot_on_grid(np.asarray(Image.fromarray((ref * 255).astype(np.uint8))
                                             .resize((W * base4.bin, H * base4.bin)), dtype=np.float32) / 255 @ LUMA,
                                  base4.bin)
    thev = thev[:H, :W]
    cols = [("Тевено", thev, measures(thev, with_sky=False))] + results
    tiles, labels = [], []
    for rl, cn, k in ROWS:
        for lab, img, m in cols:
            t = shrink(img, 700) if cn is None else np.clip(cut(img, crops[cn]) * k, 0, 1)
            tiles.append(t[::-1])
            if cn is None:
                extra = (f"\nнебо {' / '.join(f'{v:.0f}' for v in m['sky'])} · нули {' / '.join(f'{z:.2f}' for z in m['zeros'])}%"
                         if "sky" in m else "")
                labels.append(f"{lab}\nчистый зелёный {m['green']:.1f}% · выжж. {m['burned']:.2f}%{extra}")
            elif cn == "filaments":
                labels.append(f"filaments 100%\nразброс оттенка {m['hue_sd']:.1f}° · оттенок {m['hue']:.0f}° · "
                              f"цветность {m['chroma']:.0f}")
            elif cn == "star" and "star" in m:
                labels.append(f"star 100%\nзелень звёзд неба p50/75/95: {' / '.join(f'{v:+.0f}' for v in m['star'])}")
            else:
                labels.append(rl)
    # без общего заголовка: он ложится на подписи первой строки (contact_sheet не оставляет под него места)
    contact_sheet(tiles, labels, out / "64_палитра.png", cols=len(cols), tile_inches=4.2, interpolation="nearest")
    rows = ["вариант\tразброс оттенка в filaments, °\tоттенок, °\tцветность\tчистый зелёный, % маски\tвыжжено, %"
            "\tнули неба R/G/B, %\tнебо R/G/B\tзелень звёзд p50 / p75 / p95"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t{m['hue_sd']:.1f}\t{m['hue']:.0f}\t{m['chroma']:.0f}\t{m['green']:.1f}\t{m['burned']:.2f}\t"
                    + ("\t".join([' / '.join(f'{z:.2f}' for z in m['zeros']), ' / '.join(f'{v:.0f}' for v in m['sky']),
                                  ' / '.join(f'{v:+.0f}' for v in m['star'])]) if "sky" in m else "—\t—\t—"))
    rows.append(f"Тевено: {treg}")
    (out / "64_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
