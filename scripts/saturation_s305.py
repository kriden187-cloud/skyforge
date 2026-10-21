"""Фаза 4, шаг 7: насыщенность на палитре В (5 фильтров с F182M) — лист 70, bin 4, Тевено первой колонкой.

Колонки: Тевено, color.saturation 1,2 / 1,4 / 1,6. Строки: целиком, filaments, bubble, globule, star при 100%, sky ×5.
Меры: цветность (медиана max − min, уровни) и разброс оттенка (круговая σ, пиксели с цветностью > 20) в кропе
filaments по маске туманности; выжжено (канал ≥ 254 в маске); нули и медиана неба R/G/B; зелень звёзд неба
G − max(R, B) — 50-й, 75-й, 95-й процентили.

Запуск: uv run python scripts/saturation_s305.py <папка>
"""

from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import with_resolved_crops
from skyforge.clean import clean_dir
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from faint_s305 import thevenot_on_grid  # noqa: E402
from palette_s305 import workspace  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SATS = (1.2, 1.4, 1.6)
ROWS = (("целиком", None, 1), ("filaments 100%", "filaments", 1), ("bubble 100%", "bubble", 1),
        ("globule 100%", "globule", 1), ("star 100%", "star", 1), ("sky ×5", "sky", 5))


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    (out / "варианты").mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    neb_img, sky_img = (sd > 0)[::-1], (sky & full)[::-1]
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    star_img = ((sw > 0.5) & (sd < 0))[::-1]
    crops = {k.name: k for k in with_resolved_crops(base, P).report.crops}

    def cut(img, c):                                          # x, y кропа — на картинке (y вниз)
        s = max(16, c.size // base.bin)
        cx, cy = int(round(c.x / base.bin)), int(round(c.y / base.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    def measures(img, with_sky=True):
        q = cut(img, crops["filaments"])[cut(neb_img, crops["filaments"])]
        chroma = (q.max(-1) - q.min(-1)) * 255
        ang = np.deg2rad(np.array([colorsys.rgb_to_hsv(*v)[0] * 360 for v in q[chroma > 20]]))
        R = np.hypot(np.sin(ang).mean(), np.cos(ang).mean())
        p = img[neb_img]
        r = {"chroma": float(np.median(chroma)), "hue_sd": float(np.rad2deg(np.sqrt(-2 * np.log(R)))),
             "burned": float((p.max(-1) >= 254 / 255).mean() * 100)}
        if with_sky:
            sk, st = img[sky_img] * 255, img[star_img] * 255
            g = st[:, 1] - np.maximum(st[:, 0], st[:, 2])
            r |= {"zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                  "sky": [float(np.median(sk[:, i])) for i in range(3)],
                  "star": [float(np.percentile(g, k)) for k in (50, 75, 95)]}
        return r

    cols = []
    for s in SATS:
        img = render(apply(base, f"color.saturation={s}"), P, w["arrays"], w["ctx"])
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / "варианты" / f"насыщенность_{s}.png")
        cols.append((f"насыщенность {s}" + (" (сейчас в YAML)" if s == base.color.saturation else ""), img, measures(img)))
        print(f"  готово: {s}", flush=True)
    ref = cols[0][1]
    thev, treg = thevenot_on_grid(np.asarray(Image.fromarray((ref * 255).astype(np.uint8))
                                             .resize((W * base.bin, H * base.bin)), dtype=np.float32) / 255 @ LUMA,
                                  base.bin)
    cols.insert(0, ("Тевено", thev[:H, :W], measures(thev[:H, :W], with_sky=False)))
    tiles, labels = [], []
    for rl, cn, k in ROWS:
        for lab, img, m in cols:
            t = shrink(img, 700) if cn is None else np.clip(cut(img, crops[cn]) * k, 0, 1)
            tiles.append(t[::-1])
            if cn is None:
                extra = (f"\nнебо {' / '.join(f'{v:.0f}' for v in m['sky'])} · нули {' / '.join(f'{z:.2f}' for z in m['zeros'])}%"
                         if "sky" in m else "")
                labels.append(f"{lab}\nвыжжено {m['burned']:.2f}%{extra}")
            elif cn == "filaments":
                labels.append(f"filaments 100%\nцветность {m['chroma']:.0f} · разброс оттенка {m['hue_sd']:.1f}°")
            elif cn == "star" and "star" in m:
                labels.append(f"star 100%\nзелень звёзд p50/75/95: {' / '.join(f'{v:+.0f}' for v in m['star'])}")
            else:
                labels.append(rl)
    contact_sheet(tiles, labels, out / "70_насыщенность.png", cols=len(cols), tile_inches=4.2, interpolation="nearest")
    rows = ["вариант\tцветность в filaments\tразброс оттенка, °\tвыжжено, %\tнули неба R/G/B, %\tнебо R/G/B"
            "\tзелень звёзд p50 / p75 / p95"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t{m['chroma']:.0f}\t{m['hue_sd']:.1f}\t{m['burned']:.2f}\t" + (
            "\t".join([' / '.join(f'{z:.2f}' for z in m['zeros']), ' / '.join(f'{v:.0f}' for v in m['sky']),
                       ' / '.join(f'{v:+.0f}' for v in m['star'])]) if "sky" in m else "—\t—\t—"))
    rows.append(f"Тевено: {treg}")
    (out / "70_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
