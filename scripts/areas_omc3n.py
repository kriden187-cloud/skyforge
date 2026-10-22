"""OMC3-N после нового заполнения: лист 37 — ограничение (одиночные дыры F356W у протозвезды и голова выброса) при ×3;
лист 38 — карта областей отбракованных пикселей, заполненных гладко (stars.flagged_areas тем же путём, что clean).

Запуск: uv run python scripts/areas_omc3n.py <png итога> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from skyforge.align import load_aligned, with_resolved_crops
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop
from skyforge.stars import flagged_areas, solid_holes

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/omc3n/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _, meta = load_aligned(c, P)
    sc = meta["pixel_scale_arcsec"]
    solid = {f: solid_holes(a, sc) for f, a in arrays.items()}
    area = {}
    for f, a in arrays.items():
        others = [(arrays[g], s) for g, s in solid.items() if g != f] or None
        area[f] = flagged_areas(a, sc, others)[0]
    union = np.logical_or.reduce(list(area.values()))
    H, W = union.shape
    print("пикселей областей:", {f: int(m.sum()) for f, m in area.items()}, f"объединение {int(union.sum())} "
          f"({union.mean():.2%} кадра)")
    img = np.asarray(Image.open(png).convert("RGB"))                     # строка 0 — верх
    assert img.shape[:2] == (H, W)
    # 38: карта — финал в 1/3, контуры областей по фильтрам
    s = 3
    ov = Image.fromarray(img[::s, ::s].copy())
    d = ImageDraw.Draw(ov)
    colors = {"F150W": (80, 160, 255), "F200W": (0, 255, 255), "F356W": (255, 160, 0), "F480M": (255, 60, 60)}
    font = ImageFont.truetype("arial.ttf", 18)
    for f, m in area.items():
        lab, n = ndimage.label(np.flipud(m))
        for sl in ndimage.find_objects(lab):
            if sl is None:
                continue
            y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
            d.rectangle([x0 / s - 6, y0 / s - 6, x1 / s + 6, y1 / s + 6], outline=colors[f], width=2)
    y = 8
    for f, col in colors.items():
        d.text((8, y), f"{f}: {int(area[f].sum())} px", fill=col, font=font)
        y += 22
    d.text((8, y), f"всего {int(union.sum())} px = {union.mean():.2%} кадра", fill=(255, 255, 255), font=font)
    ov.save(out / "38_карта_областей.png")
    # 37: ограничение при ×3 — нижняя часть кропа cores (одиночные дыры F356W) и голова выброса (кроп jet)
    cr = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    g = img[::-1].astype(np.float32) / 255
    tiles, labels = [], []
    k = cr["cores"]
    sl = display_crop((H, W), k.x, k.y, k.size, 1)
    t = g[sl]
    h, w = t.shape[:2]
    tiles.append(t[h // 3: 2 * h // 3, w // 3: 2 * w // 3][::-1])
    labels.append("cores · ×3: одиночные дыры F356W (точки)")
    k = cr["jet"]
    sl = display_crop((H, W), k.x, k.y, k.size, 1)
    t = g[sl]
    h, w = t.shape[:2]
    tiles.append(t[h // 2: h // 2 + h // 3, w // 3: 2 * w // 3][::-1])
    labels.append("jet · ×3: узлы F200W по ядру, вокруг точки")
    contact_sheet(tiles, labels, out / "37_ограничение_протозвезда.png", cols=2, tile_inches=5.5, interpolation="nearest",
                  title="OMC3-N финал, ×3: что осталось у протозвезды (слева) и у головы выброса (справа)")
    print(out / "37_ограничение_протозвезда.png", out / "38_карта_областей.png")


if __name__ == "__main__":
    main()
