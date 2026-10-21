"""Фаза 4: опорное небо растяжки по самым тёмным блокам кадра (+ sky_level 16 / 22) — лист 79, bin 1.

Опора: кадр делится на блоки 512 px; берутся 3 блока с самой низкой медианой F444W (после clean) по полю всех фильтров вне
маски звёзд (не меньше 50% таких пикселей в блоке; при 90% проходят 11 блоков из ~150, все в туманности); маска опорного неба — эти блоки вне звёзд. Подмена маски неба для
чёрной точки и носка — только в этом скрипте (sweep_3b.load_masks), код конвейера не меняется.
Колонки: Тевено, Маккохрейн, финал, опора по тёмным блокам, + sky_level 16, + sky_level 22.
Строки: целиком, левый верхний угол 1800 px (1:3), нижняя полоса 600 px по всей ширине (1:4), небо ×5 (кроп sky),
seam при 100%, gap212 при 100%.
Меры (уровни, R/G/B): медиана в четырёх углах 600 px и в нижней полосе 300 px (поле всех фильтров, без звёзд) против
медианы неба в кропе sky; нули неба по маске неба финала; выжжено в маске туманности; полосы в кропах sky и gap212
(σ медиан столбцов / шум, scripts/stripes_s305.py).

Запуск: uv run python scripts/skyref2_s305.py <папка> <png финала bin 1>
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
from showcase_variants_3i import apply  # noqa: E402
from stripes_s305 import stripes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
BLOCK, NBLOCKS, CORNER, BAND, CSQ = 512, 3, 1800, 300, 600


def main() -> None:
    out, final_png = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    stars = ndimage.binary_dilation(np.load(clean_dir(base, P) / "star_weight.npy") > 0.01, iterations=3)
    ok = full & ~stars
    f444 = w["arrays"]["F444W"]
    blocks = []
    for y in range(0, H - BLOCK + 1, BLOCK):
        for x in range(0, W - BLOCK + 1, BLOCK):
            m = ok[y:y + BLOCK, x:x + BLOCK]
            if m.mean() >= 0.5:                                    # маска звёзд с отступом — ~20% кадра
                blocks.append((float(np.median(f444[y:y + BLOCK, x:x + BLOCK][m])), y, x))
    blocks.sort()
    ref = np.zeros((H, W), bool)
    for _, y, x in blocks[:NBLOCKS]:
        ref[y:y + BLOCK, x:x + BLOCK] = ok[y:y + BLOCK, x:x + BLOCK]
    crops = {k.name: k for k in with_resolved_crops(base, P).report.crops}
    ks = crops["sky"]
    gs = display_crop((H, W), ks.x, ks.y, ks.size, 1)
    sky_crop_med = float(np.median(f444[gs][sky[gs] & ok[gs]]))
    print("самые тёмные блоки (медиана F444W; строка, столбец сетки; на картинке x, y):",
          "; ".join(f"{v:.3f} ({y}, {x}; x {x}, y {H - y - BLOCK})" for v, y, x in blocks[:NBLOCKS]),
          f"| медиана F444W в кропе sky {sky_crop_med:.3f}; блоков в выборке {len(blocks)}", flush=True)
    orig = sweep_3b.load_masks

    def probe_masks(c, p):
        return (ref, orig(c, p)[1])

    # области мер на картинке (строка 0 — верх)
    ok_img = ok[::-1]
    corners = np.zeros((H, W), bool)
    corners[:CSQ, :CSQ] = corners[:CSQ, -CSQ:] = corners[-CSQ:, :CSQ] = corners[-CSQ:, -CSQ:] = True
    edge = ~np.pad(np.ones((H - 2, W - 2), bool), 1)
    dist = ndimage.distance_transform_edt(np.pad(full, 1))[1:-1, 1:-1][::-1]     # до края поля всех фильтров
    rows_valid = np.nonzero(full[::-1].any(1))[0]
    bottom = np.zeros((H, W), bool)
    bottom[rows_valid[-1] - BAND + 1:rows_valid[-1] + 1, :] = True
    corners_m, bottom_m = corners & ok_img & (dist > 0), bottom & ok_img
    del edge
    sky_img, neb_img = (sky & full)[::-1], (sd > 0)[::-1]
    ssl = (slice(H - gs[0].stop, H - gs[0].start), gs[1])
    sky_in = sky_img[ssl] & ok_img[ssl]

    def img_slice(name):
        k = crops[name]
        s = display_crop((H, W), k.x, k.y, k.size, 1)
        return (slice(H - s[0].stop, H - s[0].start), s[1])

    gap_sl, seam_sl = img_slice("gap212"), img_slice("seam")

    def measures(img, whole=True):
        r = {"corners": [float(np.median(img[corners_m][:, i] * 255)) for i in range(3)],
             "bottom": [float(np.median(img[bottom_m][:, i] * 255)) for i in range(3)],
             "sky": [float(np.median(img[ssl][sky_in][:, i] * 255)) for i in range(3)]}
        if whole:
            sk = img[sky_img] * 255
            r |= {"zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                  "burned": float((img[neb_img].max(-1) >= 254 / 255).mean() * 100),
                  "str_sky": stripes(img[ssl] @ LUMA * 255, ~stars[::-1][ssl]),
                  "str_gap": stripes(img[gap_sl] @ LUMA * 255, ~stars[::-1][gap_sl])}
        return r

    final = np.asarray(Image.open(final_png).convert("RGB"), dtype=np.float32) / 255
    lum = final @ LUMA
    thev, rt = ref_on_grid("data/reference/sh2305/thevenot_lw_commons.jpg", -1, lum)
    mcc, rm = ref_on_grid("data/reference/sh2305/mccaughrean_commons.jpg", 0, lum)
    cols = [("Тевено", thev, measures(thev, False)), ("Маккохрейн", mcc, measures(mcc, False)),
            ("финал", final, measures(final))]
    sweep_3b.load_masks = probe_masks
    for lab, spec in (("опора: тёмные блоки", ""), ("+ sky_level 16", "stretch.sky_level=16"),
                      ("+ sky_level 22", "stretch.sky_level=22")):
        img = sweep_3b.render(apply(base, spec), P, w["arrays"], w["ctx"])
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / f"79_вариант_{lab.replace(':', '').replace(' ', '_')}.png")
        cols.append((lab, img, measures(img)))
        print(f"  готово: {lab}", flush=True)
    sweep_3b.load_masks = orig

    def small(t, f):
        return np.asarray(Image.fromarray((np.clip(t, 0, 1) * 255).astype(np.uint8))
                          .resize((max(1, int(t.shape[1] / f)), max(1, int(t.shape[0] / f))), Image.LANCZOS),
                          dtype=np.float32) / 255

    fmt = lambda v: " / ".join(f"{x:.0f}" for x in v)
    tiles, labels = [], []
    for lab, img, m in cols:
        tiles.append(shrink(img, 900)[::-1])
        extra = f"\nнули неба {fmt(m['zeros'])}% · выжжено {m['burned']:.2f}%" if "zeros" in m else ""
        labels.append(f"{lab} · целиком{extra}")
    for lab, img, m in cols:
        tiles.append(small(img[:CORNER, :CORNER], 3)[::-1])
        labels.append(f"{lab} · левый верхний угол (1:3)\nуглы 600 px {fmt(m['corners'])} · небо sky {fmt(m['sky'])}")
    for lab, img, m in cols:
        r0 = rows_valid[-1] - 2 * BAND + 1
        tiles.append(small(img[r0:rows_valid[-1] + 1, :], 4)[::-1])
        labels.append(f"{lab} · нижняя полоса 600 px (1:4)\nнижние 300 px {fmt(m['bottom'])}")
    for lab, img, m in cols:
        tiles.append(np.clip(img[ssl] * 5, 0, 1)[::-1])
        labels.append(f"{lab} · небо ×5" + (f"\nполосы sky верт./гориз. {m['str_sky'][0]:.1f} / {m['str_sky'][1]:.1f}"
                                            if "str_sky" in m else ""))
    for lab, img, _ in cols:
        tiles.append(img[seam_sl][::-1])
        labels.append(f"{lab} · seam 100%")
    for lab, img, m in cols:
        tiles.append(img[gap_sl][::-1])
        labels.append(f"{lab} · gap212 100%" + (f"\nполосы верт./гориз. {m['str_gap'][0]:.1f} / {m['str_gap'][1]:.1f}"
                                               if "str_gap" in m else ""))
    contact_sheet(tiles, labels, out / "79_опора_тёмные_блоки.png", cols=len(cols), tile_inches=3.8, interpolation="lanczos")
    rows = ["колонка\tуглы 600 px R/G/B\tнижние 300 px R/G/B\tнебо sky R/G/B\tнули неба R/G/B, %\tвыжжено, %"
            "\tполосы sky верт./гориз.\tполосы gap212 верт./гориз."]
    for lab, _, m in cols:
        rows.append(f"{lab}\t{fmt(m['corners'])}\t{fmt(m['bottom'])}\t{fmt(m['sky'])}\t" + (
            f"{' / '.join(f'{z:.2f}' for z in m['zeros'])}\t{m['burned']:.2f}\t{m['str_sky'][0]:.1f} / {m['str_sky'][1]:.1f}"
            f"\t{m['str_gap'][0]:.1f} / {m['str_gap'][1]:.1f}" if "zeros" in m else "—\t—\t—\t—"))
    rows.append(f"Тевено: {rt}; Маккохрейн: {rm}; опорные блоки: "
                + "; ".join(f"x {x}, y {H - y - BLOCK} ({BLOCK} px, F444W {v:.3f})" for v, y, x in blocks[:NBLOCKS]))
    (out / "79_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
