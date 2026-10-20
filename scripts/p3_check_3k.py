"""3k, п.3: проверки полинома неба в showcase — углы, туманность, три детали фона.

1. Яркость пустого неба в четырёх углах кадра (окна CORNER_PX, медиана яркости по маске неба, уровни из 255).
2. Изменение внутри туманности и кольца ореола (маска туманности clean + 300 px) против прежнего состояния:
   |Δ| по каналам на шкале 0–1 (16-битные TIFF), среднее и 95%.
3. Детали: полоса у x≈1190 (кроп листа 07: x 1000–1500, y 0–500), ступенька у x≈1225 (тот же кроп), прямоугольник
   в кропе sky — числа и лист видимости при 100%, ×2, ×5 (до и после).

Запуск: uv run python scripts/p3_check_3k.py <папка> <до: прогон showcase> <после: прогон showcase> [метки до,после]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import load_aligned, with_resolved_crops
from skyforge.clean import load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import nexp_maps  # noqa: E402
from final_sheet_3i import crop  # noqa: E402

CORNER_PX, INSET = 300, 30
LUMA = np.array([0.2126, 0.7152, 0.0722])
BAND = (0, 500, 1000, 1500)       # y0, y1, x0, x1 на картинке (y вниз), как лист 07


def load_tif(run: pathlib.Path) -> np.ndarray:
    import tifffile

    a = tifffile.imread(run / "ngc3132_showcase.tif").astype(np.float64)
    return a[..., :3] / (65535.0 if a.max() > 255 else 255.0)


def main() -> None:
    out, before, after = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    lab_b, lab_a = (sys.argv[4].split(",") if len(sys.argv) > 4 else ["до", "после"])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    sky = load_sky_mask(cfg, P)[::-1]            # в ориентацию картинки (строка 0 — верх)
    sd = load_nebula_sd(cfg, P)[::-1]
    _, wcs, _ = load_aligned(cfg, P)
    ne = {f: n[::-1] for f, n in nexp_maps(P, wcs, sky.shape).items()}
    imgs = {lab_b: load_tif(before), lab_a: load_tif(after)}
    H, W = sky.shape
    rows = []
    # 1. углы
    corners = {"верх-лево": (INSET, INSET), "верх-право": (INSET, W - INSET - CORNER_PX),
               "низ-лево": (H - INSET - CORNER_PX, INSET), "низ-право": (H - INSET - CORNER_PX, W - INSET - CORNER_PX)}
    rows.append("углы: яркость пустого неба, уровни из 255\t" + "\t".join(corners) + "\tразмах")
    for lab, im in imgs.items():
        vals = []
        for y0, x0 in corners.values():
            m = sky[y0:y0 + CORNER_PX, x0:x0 + CORNER_PX]
            vals.append(float(np.median((im[y0:y0 + CORNER_PX, x0:x0 + CORNER_PX] @ LUMA)[m])) * 255)
        rows.append(f"{lab}\t" + "\t".join(f"{v:.2f}" for v in vals) + f"\t{max(vals) - min(vals):.2f}")
    # 2. туманность и кольцо
    zone = sd > -300
    d = np.abs(imgs[lab_a] - imgs[lab_b])[zone]
    rows.append("")
    rows.append(f"туманность + кольцо 300 px: |Δ| по каналам (шкала 0–1): среднее {d.mean():.4f}, 95% {np.percentile(d, 95):.4f}, "
                f"макс {d.max():.4f}; только маска туманности: среднее {np.abs(imgs[lab_a] - imgs[lab_b])[sd > 0].mean():.4f}")
    # 3. детали
    y0, y1, x0, x1 = BAND
    rows.append("")
    rows.append("деталь\tсостояние\tвеличина")
    for lab, im in imgs.items():
        win = im[y0:y1, x0:x1] * 255
        m = sky[y0:y1, x0:x1]
        lum = np.where(m, win @ LUMA, np.nan)
        gex = np.where(m, win[..., 1] - (win[..., 0] + win[..., 2]) / 2, np.nan)
        col_l = np.nanmedian(lum, axis=0)
        col_g = np.nanmedian(gex, axis=0)
        xs = np.arange(x0, x1)
        inb = (xs >= 1170) & (xs < 1210)
        side = ((xs >= 1110) & (xs < 1150)) | ((xs >= 1260) & (xs < 1300))
        rows.append(f"полоса x≈1190\t{lab}\tзелёный избыток {np.nanmean(col_g[inb]) - np.nanmean(col_g[side]):+.2f} ур., "
                    f"яркость {np.nanmean(col_l[inb]) - np.nanmean(col_l[side]):+.2f} ур.")
        left, right = (xs >= 1195) & (xs < 1222), (xs >= 1228) & (xs < 1255)
        rows.append(f"ступенька x≈1225\t{lab}\tяркость справа − слева {np.nanmean(col_l[right]) - np.nanmean(col_l[left]):+.2f} ур., "
                    f"зелёный {np.nanmean(col_g[right]) - np.nanmean(col_g[left]):+.2f} ур.")
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    c = crops["sky"]
    cy0, cx0 = int(np.clip(c.y - c.size // 2, 0, H - c.size)), int(np.clip(c.x - c.size // 2, 0, W - c.size))
    sl = (slice(cy0, cy0 + c.size), slice(cx0, cx0 + c.size))
    combo = (ne["F090W"] * 100 + ne["F356W"] * 10 + np.minimum(ne["F470N"], 9))[sl]
    msk = sky[sl]
    for lab, im in imgs.items():
        lum = (im[sl] * 255) @ LUMA
        meds = {int(k): float(np.median(lum[msk & (combo == k)])) for k in np.unique(combo[msk]) if (msk & (combo == k)).sum() > 3000}
        rows.append(f"прямоугольник в sky\t{lab}\tмедиана яркости по областям NEXP (F090W/F356W/F470N): "
                    + ", ".join(f"{k // 100}/{(k // 10) % 10}/{k % 10}: {v:.2f}" for k, v in meds.items())
                    + f"; размах {max(meds.values()) - min(meds.values()):.2f} ур.")
    text = "\n".join(rows) + "\n"
    (out / "22_проверка_п3.tsv").write_text(text, encoding="utf-8")
    print(text)
    # лист видимости
    tiles, labels = [], []
    feats = [("полоса 1190 / ступенька 1225", (y0 + y1) // 2, (x0 + x1) // 2, 500),
             ("прямоугольник sky", c.y, c.x, c.size)]
    for name, cy, cx, size in feats:
        for lab, im in imgs.items():
            for k in (1, 2, 5):
                tiles.append(np.clip(crop(im.astype(np.float32), cx, cy, size) * k, 0, 1))
                labels.append(f"{lab} · {'100%' if k == 1 else f'×{k}'}")
    contact_sheet([t[::-1] for t in tiles], labels, out / "22_видимость_деталей.png", cols=6, tile_inches=3.4,
                  interpolation="nearest", title=f"Детали фона showcase при 100%, ×2, ×5 — сверху полоса у x≈1190 и ступенька у x≈1225 (кроп листа 07), снизу прямоугольник в кропе sky")


if __name__ == "__main__":
    main()
