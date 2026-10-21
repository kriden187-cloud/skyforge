"""Фаза 4: опорное небо растяжки только по чистому небу — лист 78, bin 1.

Проба: маска «пустого неба» для чёрной точки и носка (stretch.anchor: masks) сужена до её части внутри кропа sky
(чистое небо, левая колонка плиток). Остальное как в финале: маски для цвета и финиша прежние. Подмена — только
в этом скрипте (sweep_3b.load_masks), код конвейера не меняется.
Колонки: Тевено, Маккохрейн, финал, проба. Строки: целиком, левый нижний угол (1800 px, уменьшение 3), кроп владельца
(уменьшение 5,75 ×2), небо ×5 (кроп sky), стык ×5 (кроп seam).
Меры: медиана R/G/B неба в кропе sky (пиксели маски неба), медиана R/G/B краевой полосы 300 px по периметру кадра (поле
всех фильтров, без звёзд), нули неба R/G/B по всей маске неба, выжжено в маске туманности.

Запуск: uv run python scripts/skyref_s305.py <папка> <png финала bin 1>
"""

from __future__ import annotations

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
from skyforge.preview import contact_sheet, display_crop, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import sweep_3b  # noqa: E402
from palette_s305 import workspace  # noqa: E402
from refgrid_s305 import ref_on_grid  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
OWNER = (2254, 862, 1725, 2346)
CORNER = 1800
BAND = 300


def main() -> None:
    out, final_png = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    H, W = sd.shape
    crops = {k.name: k for k in with_resolved_crops(base, P).report.crops}

    def grid_slice(name):
        k = crops[name]
        return display_crop((H, W), k.x, k.y, k.size, 1)

    ref_sky = np.zeros_like(sky)
    gs = grid_slice("sky")
    ref_sky[gs] = sky[gs]
    print(f"опорное небо пробы: {int(ref_sky.sum())} px (вся маска неба — {int(sky.sum())} px, {sky.mean() * 100:.1f}% кадра)")
    orig = sweep_3b.load_masks

    def probe_masks(c, p):
        m = orig(c, p)
        return (ref_sky, m[1])

    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    stars = ndimage.binary_dilation(np.load(clean_dir(base, P) / "star_weight.npy") > 0.01, iterations=3)
    # полоса вдоль края поля всех фильтров: расстояние до края поля ≤ BAND
    band = (ndimage.distance_transform_edt(np.pad(full, 1))[1:-1, 1:-1] <= BAND) & full & ~stars   # рамка: край кадра — тоже край
    sky_img, neb_img, band_img = (sky & full)[::-1], (sd > 0)[::-1], band[::-1]
    ss = grid_slice("sky")
    ss_img = (slice(H - ss[0].stop, H - ss[0].start), ss[1])
    sky_in_crop = sky_img[ss_img]

    def measures(img, whole=True):
        r = {"crop_sky": [float(np.median(img[ss_img][sky_in_crop][:, i] * 255)) for i in range(3)],
             "band": [float(np.median(img[band_img][:, i] * 255)) for i in range(3)]}
        if whole:
            sk = img[sky_img] * 255
            r |= {"zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                  "burned": float((img[neb_img].max(-1) >= 254 / 255).mean() * 100)}
        return r

    final = np.asarray(Image.open(final_png).convert("RGB"), dtype=np.float32) / 255
    sweep_3b.load_masks = probe_masks
    probe = sweep_3b.render(base, P, w["arrays"], w["ctx"])
    sweep_3b.load_masks = orig
    Image.fromarray((probe * 255 + 0.5).astype(np.uint8)).save(out / "78_проба_опорное_небо.png")
    print("  готово: проба", flush=True)
    lum = final @ LUMA
    thev, rt = ref_on_grid("data/reference/sh2305/thevenot_lw_commons.jpg", -1, lum)
    mcc, rm = ref_on_grid("data/reference/sh2305/mccaughrean_commons.jpg", 0, lum)
    cols = [("Тевено", thev, measures(thev, False)), ("Маккохрейн", mcc, measures(mcc, False)),
            ("финал", final, measures(final)), ("проба: небо по кропу sky", probe, measures(probe))]
    ox, oy, ow, oh = OWNER
    corner = (slice(H - CORNER, H), slice(0, CORNER))
    seam = grid_slice("seam")
    seam_img = (slice(H - seam[0].stop, H - seam[0].start), seam[1])
    tiles, labels = [], []

    def small(t, f):
        return np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
                          .resize((int(t.shape[1] / f), int(t.shape[0] / f)), Image.LANCZOS), dtype=np.float32) / 255

    for lab, img, m in cols:
        tiles.append(shrink(img, 900)[::-1])
        extra = (f"\nнули неба {' / '.join(f'{z:.2f}' for z in m['zeros'])}% · выжжено {m['burned']:.2f}%"
                 if "zeros" in m else "")
        labels.append(f"{lab} · целиком{extra}")
    for lab, img, m in cols:
        tiles.append(small(img[corner], 3)[::-1])
        labels.append(f"{lab} · левый нижний угол {CORNER} px (1:3)\nкраевая полоса {BAND} px: "
                      f"{' / '.join(f'{v:.0f}' for v in m['band'])}")
    for lab, img, _ in cols:
        tiles.append(small(img[oy:oy + oh, ox:ox + ow], 5.75 / 2)[::-1])
        labels.append(f"{lab} · кроп владельца (1:5,75, ×2)")
    for lab, img, m in cols:
        tiles.append(np.clip(img[ss_img] * 5, 0, 1)[::-1])
        labels.append(f"{lab} · небо ×5\nмедиана неба в кропе sky {' / '.join(f'{v:.0f}' for v in m['crop_sky'])}")
    for lab, img, _ in cols:
        tiles.append(np.clip(img[seam_img] * 5, 0, 1)[::-1])
        labels.append(f"{lab} · стык ×5")
    contact_sheet(tiles, labels, out / "78_опорное_небо.png", cols=len(cols), tile_inches=4.4, interpolation="lanczos")
    rows = ["колонка\tнебо в кропе sky R/G/B\tкраевая полоса 300 px R/G/B\tнули неба R/G/B, %\tвыжжено, %"]
    for lab, _, m in cols:
        rows.append(f"{lab}\t{' / '.join(f'{v:.0f}' for v in m['crop_sky'])}\t{' / '.join(f'{v:.0f}' for v in m['band'])}\t"
                    + (f"{' / '.join(f'{z:.2f}' for z in m['zeros'])}\t{m['burned']:.2f}" if "zeros" in m else "—\t—"))
    rows.append(f"Тевено: {rt}; Маккохрейн: {rm}")
    (out / "78_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
