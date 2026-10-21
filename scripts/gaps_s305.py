"""Фаза 4, шаг 4b: тёмные промежутки между волокнами — лист 75, bin 1, кроп владельца (как лист 72).

Промежутки — пиксели кропа вне маски туманности и без звёзд (маска clean с отступом 3 px). Четверти по яркости
задаются по финалу и одинаковы для всех колонок (те же пиксели у Тевено и у вариантов). Для каждой четверти — медиана
R/G/B и оттенок; по всему кадру — медиана неба и нули неба R/G/B.
Строки листа: кроп владельца (1:5,75, ×2), небо ×5 (кроп sky при 100%, ×5), центр кропа 600 px при 100%.

Запуск: uv run python scripts/gaps_s305.py <папка> <png финала bin 1> <имя листа> <подпись>=<правки> ...
  (правки через «;», «финал» без правок — готовый png)
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
from skyforge.preview import contact_sheet, display_crop

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from palette_s305 import workspace  # noqa: E402
from refgrid_s305 import ref_on_grid  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
X0, Y0, CW, CH, SCALE = 2254, 862, 1725, 2346, 5.75
Q = ("тёмная", "вторая", "третья", "светлая")


def hue(v: np.ndarray) -> int:
    return round(colorsys.rgb_to_hsv(*(np.clip(v, 0, 255) / 255))[0] * 360)


def main() -> None:
    out, final_png, sheet = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    variants = [tuple(a.split("=", 1)) for a in sys.argv[4:]]
    vdir = out / f"варианты_{sheet.split('_')[0]}"
    vdir.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    sky_img = (sky & full)[::-1]
    stars = ndimage.binary_dilation(np.load(clean_dir(base, P) / "star_weight.npy") > 0.01, iterations=3)[::-1]
    H, W = sd.shape
    cs = (slice(Y0, Y0 + CH), slice(X0, X0 + CW))
    cy, cx = Y0 + CH // 2, X0 + CW // 2
    zs = (slice(cy - 300, cy + 300), slice(cx - 300, cx + 300))
    k = {x.name: x for x in with_resolved_crops(base, P).report.crops}["sky"]
    ss = display_crop((H, W), k.x, k.y, k.size, 1)
    ss = (slice(H - ss[0].stop, H - ss[0].start), ss[1])
    gap = ((sd < 0)[::-1] & ~stars)[cs]
    final = np.asarray(Image.open(final_png).convert("RGB"), dtype=np.float32) / 255
    lum = (final[cs][gap] @ LUMA) * 255
    q = np.percentile(lum, [25, 50, 75])
    sets = [lum <= q[0], (lum > q[0]) & (lum <= q[1]), (lum > q[1]) & (lum <= q[2]), lum > q[2]]
    thev, treg = ref_on_grid("data/reference/sh2305/thevenot_lw_commons.jpg", -1, final @ LUMA)
    print("Тевено:", treg, flush=True)

    def measures(img, whole=True):
        g = img[cs][gap] * 255
        r = {"q": [np.median(g[s], axis=0) for s in sets]}
        if whole:
            sk = img[sky_img] * 255
            r |= {"sky": [float(np.median(sk[:, i])) for i in range(3)],
                  "zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)]}
        return r

    cols = [("Тевено", thev, measures(thev, whole=False))]
    for lab, spec in variants:
        use_final = lab == "финал" and not spec
        img = final if use_final else render(apply(base, spec), P, w["arrays"], w["ctx"])
        if not use_final:
            Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(vdir / f"{lab}.png")
        cols.append((lab, img, measures(img)))
        print(f"  готово: {lab}", flush=True)
    tiles, labels = [], []
    for lab, img, m in cols:
        t = np.asarray(Image.fromarray((np.clip(img[cs], 0, 1) * 255).astype(np.uint8))
                       .resize((int(CW / SCALE * 2), int(CH / SCALE * 2)), Image.LANCZOS), dtype=np.float32) / 255
        tiles.append(t[::-1])
        d = m["q"][0]
        labels.append(f"{lab} · кроп владельца\nтёмная четверть {d[0]:.0f}/{d[1]:.0f}/{d[2]:.0f}, {hue(d)}°\n"
                      f"светлая {m['q'][3][0]:.0f}/{m['q'][3][1]:.0f}/{m['q'][3][2]:.0f}")
    for lab, img, m in cols:
        tiles.append(np.clip(img[ss] * 5, 0, 1)[::-1])
        labels.append(f"{lab} · небо ×5" + (f"\nнебо {' / '.join(f'{v:.0f}' for v in m['sky'])} · нули "
                                             f"{' / '.join(f'{z:.2f}' for z in m['zeros'])}%" if "sky" in m else ""))
    for lab, img, _ in cols:
        tiles.append(img[zs][::-1])
        labels.append(f"{lab} · центр 600 px при 100%")
    contact_sheet(tiles, labels, out / f"{sheet}.png", cols=len(cols), tile_inches=4.2, interpolation="lanczos")
    rows = ["вариант\t" + "\t".join(f"{n} четверть R/G/B, оттенок" for n in Q) + "\tнебо R/G/B\tнули неба R/G/B, %"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t" + "\t".join(f"{v[0]:.0f}/{v[1]:.0f}/{v[2]:.0f}, {hue(v)}°" for v in m["q"]) + "\t"
                    + (f"{' / '.join(f'{v:.0f}' for v in m['sky'])}\t{' / '.join(f'{z:.2f}' for z in m['zeros'])}"
                       if "sky" in m else "—\t—"))
    rows.append(f"Тевено: {treg}; кроп x {X0}, y {Y0}, {CW}×{CH} px; четверти по яркости финала, промежутков {gap.sum()} px")
    (out / f"{sheet.split('_')[0]}_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
