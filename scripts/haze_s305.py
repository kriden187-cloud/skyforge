"""Фаза 4, шаг 7: «плотные комки» против «прозрачной дымки» — лист 72, bin 1, кроп по скриншоту владельца.

Кроп найден сопоставлением скриншота с финалом (масштаб 1 : 5,75, совпадение 0,93): x 2254, y 862 на картинке bin 1,
1725 × 2346 px. Колонки: Тевено (та же сетка), финал, finish.local_contrast.amount 0, color.shadows.value 2 и 0,
контраст 0 + тени 0. Строки: вид как у владельца (уменьшение 5,75) и центр кропа 600 px при 100%.
Шумоподавление (finish.denoise) в финале уже 0 — отдельной клетки нет. Меры в кропе (уровни 0–255):
- туманность / промежутки — медиана яркости в маске туманности и вне её (без звёзд);
- жёсткость края — 95-й процентиль модуля градиента яркости, сглаженной гауссом σ 6 px (резкий край сгустка — больше);
- мелкая фактура / яркость — σ остатка после размытия σ 2 px, делённая на среднюю яркость, в маске;
- по всему кадру — выжжено в маске и нули неба R/G/B.

Запуск: uv run python scripts/haze_s305.py <папка шага> <png финала bin 1> [<имя листа> <подпись>=<правки> ...]
  без имени листа — лист 72 с вариантами VARIANTS; иначе свои варианты, правки через «;», пусто — конфиг как есть;
  подпись «финал» без правок берёт готовый png финала (без перерисовки).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.clean import clean_dir
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from palette_s305 import workspace  # noqa: E402
from refgrid_s305 import ref_on_grid  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
X0, Y0, CW, CH, SCALE = 2254, 862, 1725, 2346, 5.75
VARIANTS = (("финал", ""), ("контраст 0", "finish.local_contrast.amount=0"), ("тени 2", "color.shadows.value=2"),
            ("тени 0", "color.shadows.value=0"), ("контраст 0 + тени 0", "finish.local_contrast.amount=0;color.shadows.value=0"))


def main() -> None:
    out, final_png = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    sheet, variants = "72_дымка_или_комки", VARIANTS
    if len(sys.argv) > 3:
        sheet, variants = sys.argv[3], tuple(tuple(a.split("=", 1)) for a in sys.argv[4:])
    vdir = out / f"варианты_{sheet.split('_')[0]}"
    vdir.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    neb, sky_img = (sd > 0)[::-1], (sky & full)[::-1]
    stars = ndimage.binary_dilation(np.load(clean_dir(base, P) / "star_weight.npy") > 0.01, iterations=3)[::-1]
    cs = (slice(Y0, Y0 + CH), slice(X0, X0 + CW))
    cy, cx = Y0 + CH // 2, X0 + CW // 2
    zs = (slice(cy - 300, cy + 300), slice(cx - 300, cx + 300))
    ours_final = np.asarray(Image.open(final_png).convert("RGB"), dtype=np.float32) / 255
    thev, treg = ref_on_grid("data/reference/sh2305/thevenot_lw_commons.jpg", -1, ours_final @ LUMA)
    print("Тевено:", treg, flush=True)

    def measures(img, whole=True):
        lum = img[cs] @ LUMA * 255
        m, bgm = neb[cs] & ~stars[cs], ~neb[cs] & ~stars[cs]
        g = ndimage.gaussian_filter(lum, 6)
        grad = np.hypot(*np.gradient(g))
        hp = lum - ndimage.gaussian_filter(lum, 2)
        r = {"neb": float(np.median(lum[m])), "gap": float(np.median(lum[bgm])), "edge": float(np.percentile(grad, 95)),
             "fine": float(hp[m].std() / lum[m].mean())}
        if whole:
            sk = img[sky_img] * 255
            r |= {"burned": float((img[neb].max(-1) >= 254 / 255).mean() * 100),
                  "zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)]}
        return r

    cols = [("Тевено", thev, measures(thev, whole=False))]
    for lab, spec in variants:
        use_final = lab == "финал" and not spec
        img = ours_final if use_final else render(apply(base, spec), P, w["arrays"], w["ctx"])
        if not use_final:
            Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(vdir / f"{lab}.png")
        cols.append((lab, img, measures(img)))
        print(f"  готово: {lab}", flush=True)
    tiles, labels = [], []
    for lab, img, m in cols:
        t = np.asarray(Image.fromarray((np.clip(img[cs], 0, 1) * 255).astype(np.uint8))
                       .resize((int(CW / SCALE * 2), int(CH / SCALE * 2)), Image.LANCZOS), dtype=np.float32) / 255
        tiles.append(t[::-1])
        z = (f"\nвыжжено {m['burned']:.2f}% · нули неба {' / '.join(f'{v:.2f}' for v in m['zeros'])}%" if "burned" in m else "")
        labels.append(f"{lab} · как у владельца (1:5,75, ×2)\nтуманность {m['neb']:.0f} · промежутки {m['gap']:.0f} · "
                      f"край {m['edge']:.2f}{z}")
    for lab, img, m in cols:
        tiles.append(img[zs][::-1])
        labels.append(f"{lab} · центр 600 px при 100%\nмелкая фактура / яркость {m['fine']:.3f}")
    contact_sheet(tiles, labels, out / f"{sheet}.png", cols=len(cols), tile_inches=4.0, interpolation="lanczos")
    rows = ["вариант\tтуманность, медиана\tпромежутки, медиана\tжёсткость края (p95 |∇|)\tфактура / яркость"
            "\tвыжжено, %\tнули неба R/G/B, %"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t{m['neb']:.0f}\t{m['gap']:.0f}\t{m['edge']:.2f}\t{m['fine']:.3f}\t"
                    + (f"{m['burned']:.2f}\t{' / '.join(f'{v:.2f}' for v in m['zeros'])}" if "burned" in m else "—\t—"))
    rows.append(f"Тевено: {treg}; кроп x {X0}, y {Y0}, {CW}×{CH} px (скриншот владельца, масштаб 1:{SCALE})")
    (out / f"{sheet.split('_')[0]}_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
