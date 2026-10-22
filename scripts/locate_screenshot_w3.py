"""Фаза 6, шаг 3, W3-2: где на финале скриншот владельца — сопоставление шаблона (NCC по яркости) по сетке масштабов.

Масштаб s — пикселей скриншота на пиксель финала; финал уменьшается в s раз, скриншот — шаблон. Потом уточнение шага
масштаба. Пишет кроп «скриншот | финал тот же участок» и строку с координатами в PNG финала (строка 0 — верх) и тем,
какие щели (кадры_без_щелей.json — полоски и щели между кадрами) попадают в участок.

Запуск: uv run python scripts/locate_screenshot_w3.py <png финала> <скриншот> <папка>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.feature import match_template

Image.MAX_IMAGE_PIXELS = None
SCALE = 0.0629


def gray(im: Image.Image) -> np.ndarray:
    return np.asarray(im.convert("L"), np.float32)


def best_at(final: Image.Image, tpl: np.ndarray, s: float) -> tuple[float, int, int]:
    small = gray(final.resize((round(final.width * s), round(final.height * s)), Image.LANCZOS))
    if small.shape[0] < tpl.shape[0] or small.shape[1] < tpl.shape[1]:
        return -1.0, 0, 0
    ncc = match_template(small, tpl)
    i = int(np.argmax(ncc))
    y, x = divmod(i, ncc.shape[1])
    return float(ncc.flat[i]), x, y


def main() -> None:
    png, shot_p, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    final = Image.open(png).convert("RGB")
    shot = Image.open(shot_p).convert("RGB")
    tpl = gray(shot)
    scores = [(best_at(final, tpl, s), s) for s in np.arange(0.25, 1.51, 0.05)]
    (ncc, _, _), s0 = max(scores)
    print("грубо:", ", ".join(f"{s:.2f}:{r[0]:.2f}" for r, s in scores))
    fine = [(best_at(final, tpl, s), s) for s in np.arange(s0 - 0.04, s0 + 0.041, 0.005)]
    (ncc, x, y), s = max(fine)
    c0, r0 = x / s, y / s
    c1, r1 = c0 + shot.width / s, r0 + shot.height / s
    box = [int(round(v)) for v in (r0, r1, c0, c1)]
    frames = json.load(open(out / "кадры_без_щелей.json", encoding="utf-8"))
    hit = [f["name"] for f in frames
           if f["png_rows"][0] < box[1] and f["png_rows"][1] > box[0]
           and f["png_cols"][0] < box[3] and f["png_cols"][1] > box[2]]
    line = (f"скриншот 709×414 = PNG финала строки {box[0]}–{box[1]}, столбцы {box[2]}–{box[3]} "
            f"({(box[3] - box[2]) * SCALE:.0f}×{(box[1] - box[0]) * SCALE:.0f}″), масштаб скриншота {s:.3f} "
            f"(≈{1 / s:.2f} px финала на px экрана), NCC {ncc:.3f}; пересекает участки: {', '.join(hit)}")
    print(line)
    crop = final.crop((box[2], box[0], box[3], box[1])).resize(shot.size, Image.LANCZOS)
    font = ImageFont.truetype("arial.ttf", 16)
    sheet = Image.new("RGB", (shot.width * 2 + 12, shot.height + 30), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    d.text((6, 6), "скриншот владельца", fill=(255, 255, 0), font=font)
    d.text((shot.width + 18, 6), f"финал 01-40-58, тот же участок (PNG строки {box[0]}–{box[1]}, "
                                 f"столбцы {box[2]}–{box[3]})", fill=(255, 255, 0), font=font)
    sheet.paste(shot, (0, 30))
    sheet.paste(crop, (shot.width + 12, 30))
    sheet.save(out / "скриншот_и_финал.png")
    (out / "скриншот_совмещение.txt").write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
