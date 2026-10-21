"""Фаза 4, шаг 4c: палитра «два тёплых соседа у PAH» — лист 76, bin 4, Тевено первой колонкой.

Колонки: Тевено, финал (палитра В), новая палитра (F182M #3050ff, F300M #40c0ff, F335M #ffe040, F360M #ff8020,
F444W #ff2020). Строки: кроп владельца (как листы 72–75), filaments, bubble, globule, star при 100%, sky ×5.
Меры: четверти промежутков кропа владельца (вне маски туманности, без звёзд; четверти по яркости финала — одни пиксели
у всех), цветность и разброс оттенка в filaments по маске, выжжено, нули неба R/G/B, зелень звёзд p50 / p75 / p95.

Запуск: uv run python scripts/palette2_s305.py <папка>
"""

from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import with_resolved_crops
from skyforge.clean import clean_dir
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from faint_s305 import thevenot_on_grid  # noqa: E402
from palette_s305 import hue_deg, workspace  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
NEW = {"F182M": "#3050ff", "F300M": "#40c0ff", "F335M": "#ffe040", "F360M": "#ff8020", "F444W": "#ff2020"}
OWNER = (2254, 862, 1725, 2346)                     # x, y, ширина, высота на картинке bin 1 (скриншот владельца)
ROWS = (("кроп владельца", "owner", 1), ("filaments 100%", "filaments", 1), ("bubble 100%", "bubble", 1),
        ("globule 100%", "globule", 1), ("star 100%", "star", 1), ("sky ×5", "sky", 5))


def hue(v: np.ndarray) -> int:
    return round(colorsys.rgb_to_hsv(*(np.clip(v, 0, 255) / 255))[0] * 360)


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    (out / "варианты_76").mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    order = [(f, NEW[f]) for f in ("F182M", "F300M", "F335M", "F360M", "F444W")]
    hs = [hue_deg(c) for _, c in order]
    print("оттенки новой палитры:", ", ".join(f"{f} {h:.0f}°" for (f, _), h in zip(order, hs)),
          "— порядок сохранён" if all(a > b for a, b in zip(hs, hs[1:])) else "— ПОРЯДОК НАРУШЕН", flush=True)
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    H, W = sd.shape
    b = base.bin
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    neb_img, sky_img = (sd > 0)[::-1], (sky & full)[::-1]
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    star_img = ((sw > 0.5) & (sd < 0))[::-1]
    stars3 = ndimage.binary_dilation(sw > 0.01, iterations=1)[::-1]
    crops = {k.name: k for k in with_resolved_crops(base, P).report.crops}
    ox, oy, ow, oh = (v // b for v in OWNER)
    os_ = (slice(oy, oy + oh), slice(ox, ox + ow))
    gap = (~neb_img & ~stars3)[os_]

    def cut(img, name):
        if name == "owner":
            return img[os_]
        c = crops[name]
        s = max(16, c.size // b)
        cx, cy = int(round(c.x / b)), int(round(c.y / b))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    variants = [("финал (палитра В)", ""),
                ("новая палитра", ";".join(f"color.overrides.{f}={c}" for f, c in NEW.items()))]
    imgs = []
    for lab, spec in variants:
        img = render(apply(base, spec), P, w["arrays"], w["ctx"])
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / "варианты_76" / f"{lab.split(' (')[0]}.png")
        imgs.append((lab, img))
        print(f"  готово: {lab}", flush=True)
    ref = imgs[0][1]
    thev, treg = thevenot_on_grid(np.asarray(Image.fromarray((ref * 255).astype(np.uint8)).resize((W * b, H * b)),
                                             dtype=np.float32) / 255 @ LUMA, b)
    thev = thev[:H, :W]
    lum = (ref[os_][gap] @ LUMA) * 255
    q = np.percentile(lum, [25, 50, 75])
    sets = [lum <= q[0], (lum > q[0]) & (lum <= q[1]), (lum > q[1]) & (lum <= q[2]), lum > q[2]]

    def measures(img, whole=True):
        g = img[os_][gap] * 255
        qq = cut(img, "filaments")[cut(neb_img, "filaments")]
        chroma = (qq.max(-1) - qq.min(-1)) * 255
        ang = np.deg2rad(np.array([colorsys.rgb_to_hsv(*v)[0] * 360 for v in qq[chroma > 20]]))
        R = np.hypot(np.sin(ang).mean(), np.cos(ang).mean())
        p = img[neb_img]
        r = {"q": [np.median(g[s], axis=0) for s in sets], "chroma": float(np.median(chroma)),
             "hue_sd": float(np.rad2deg(np.sqrt(-2 * np.log(R)))), "burned": float((p.max(-1) >= 254 / 255).mean() * 100)}
        if whole:
            sk, st = img[sky_img] * 255, img[star_img] * 255
            gx = st[:, 1] - np.maximum(st[:, 0], st[:, 2])
            r |= {"zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                  "sky": [float(np.median(sk[:, i])) for i in range(3)],
                  "star": [float(np.percentile(gx, k)) for k in (50, 75, 95)]}
        return r

    cols = [("Тевено", thev, measures(thev, whole=False))] + [(lab, img, measures(img)) for lab, img in imgs]
    tiles, labels = [], []
    for rl, cn, k in ROWS:
        for lab, img, m in cols:
            tiles.append(np.clip(cut(img, cn) * k, 0, 1)[::-1])
            if cn == "owner":
                d, l4 = m["q"][0], m["q"][3]
                labels.append(f"{lab} · кроп владельца\nтёмная четверть {d[0]:.0f}/{d[1]:.0f}/{d[2]:.0f}, {hue(d)}°\n"
                              f"светлая {l4[0]:.0f}/{l4[1]:.0f}/{l4[2]:.0f}, {hue(l4)}°")
            elif cn == "filaments":
                labels.append(f"filaments 100%\nцветность {m['chroma']:.0f} · разброс оттенка {m['hue_sd']:.1f}°\n"
                              f"выжжено {m['burned']:.2f}%")
            elif cn == "star" and "star" in m:
                labels.append(f"star 100%\nзелень звёзд p50/75/95: {' / '.join(f'{v:+.0f}' for v in m['star'])}")
            elif cn == "sky" and "sky" in m:
                labels.append(f"sky ×5\nнебо {' / '.join(f'{v:.0f}' for v in m['sky'])} · нули "
                              f"{' / '.join(f'{z:.2f}' for z in m['zeros'])}%")
            else:
                labels.append(rl)
    contact_sheet(tiles, labels, out / "76_палитра_тёплые_соседи.png", cols=len(cols), tile_inches=4.6,
                  interpolation="nearest")
    rows = ["вариант\t" + "\t".join(f"{n} четверть" for n in ("тёмная", "вторая", "третья", "светлая"))
            + "\tцветность\tразброс оттенка, °\tвыжжено, %\tнебо R/G/B\tнули неба R/G/B, %\tзелень звёзд p50/p75/p95"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t" + "\t".join(f"{v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}, {hue(v)}°" for v in m["q"])
                    + f"\t{m['chroma']:.0f}\t{m['hue_sd']:.1f}\t{m['burned']:.2f}\t"
                    + ("\t".join([' / '.join(f'{v:.0f}' for v in m['sky']), ' / '.join(f'{z:.2f}' for z in m['zeros']),
                                  ' / '.join(f'{v:+.0f}' for v in m['star'])]) if "sky" in m else "—\t—\t—"))
    rows.append(f"Тевено: {treg}; кроп владельца при bin {b}: x {ox}, y {oy}, {ow}×{oh}; четверти по яркости финала")
    (out / "76_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
