"""Фаза 6, шаг 3, W3-2: кадры без щелей — прямоугольники между щелями коротких фильтров, кропы финала при 100%.

Щели и пустоты — пиксели внутри областей покрытия (meta.regions), где хоть у одного из восьми выровненных слоёв нет
данных (NaN; одиночные NaN меньше MIN_GAP_PX — ядра звёзд — не щели). Участки между щелями — связные области «все
слои есть» с отступом MARGIN_PX от щели; в каждом — наибольший прямоугольник по осям картинки. Пиксели не
пересчитываются: кропы из принятого PNG как есть.

Запуск: uv run python scripts/gapfree_frames_w3.py <png финала> <папка>
Пишет кадры_без_щелей.png (обзор + кропы при 100%) и кадры_без_щелей.json (координаты в PNG, строка 0 — верх).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.config import Config
from skyforge.paths import Paths

Image.MAX_IMAGE_PIXELS = None
MARGIN_PX, MIN_GAP_PX, MIN_SIDE_PX = 8, 3000, 320          # 320 px ≈ 20″: меньше — полоска, не кадр
SCALE = 0.0629
# что в кадре — по просмотру кропов финала 01-40-58 (2026-10-22); в скобках — кропы конфига внутри кадра
CONTENT = {
    "A1": "сиреневая туманность, у правого края тёмная пыль; звёзды",
    "A2": "яркая звезда (star1), тёмные прожилки, малая глобула вверху; охристая туманность",
    "A3": "сиреневая туманность с тонкими волокнами; много ярких звёзд",
    "A4": "большое тёмное пылевое облако со светлым краем, волокна внизу; звёзды",
    "B1": "яркое облако (cloud): светлая туманность с полосчатой структурой; звёзды",
    "B2": "тёмное небо со звёздами (sky), слабая пыль внизу",
    "B3": "яркое облако, светлая туманность; яркая звезда с лучами у левого края",
    "B4": "тёмное небо со слабой пылью (sky2); звёзды, яркая звезда слева вверху",
}


def largest_rect(m: np.ndarray) -> tuple[int, int, int, int]:
    """Наибольший по площади прямоугольник из True: (r0, r1, c0, c1), концы не включительно (гистограммы по строкам)."""
    h = np.zeros(m.shape[1] + 1, int)
    best, out = 0, (0, 0, 0, 0)
    for r in range(m.shape[0]):
        h[:-1] = np.where(m[r], h[:-1] + 1, 0)
        stack: list[int] = []
        for c in range(len(h)):
            while stack and h[stack[-1]] >= h[c]:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = h[top] * (c - left)
                if area > best:
                    best, out = area, (r - h[top] + 1, r + 1, left, c)
            stack.append(c)
    return out


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config.load("configs/w3_2/showcase.yaml", "showcase")
    cfg = cfg.model_copy(update={"bin": 1})
    arrays, wcs, meta = load_aligned(cfg, Paths(pathlib.Path(".").resolve()))
    H, W = meta["shape"]
    inside = np.zeros((H, W), bool)
    for r0, r1, c0, c1 in meta["regions"]:
        inside[r0:r1, c0:c1] = True
    nan = np.zeros((H, W), bool)
    for a in arrays.values():
        nan |= ~np.isfinite(a)
    lab, k = ndimage.label(nan & inside)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
    gap = np.isin(lab, 1 + np.nonzero(sizes >= MIN_GAP_PX)[0]) | ~inside
    good = ndimage.distance_transform_edt(~gap) > MARGIN_PX
    cells, n = ndimage.label(good)
    img = np.asarray(Image.open(png).convert("RGB"))
    assert img.shape[:2] == (H, W), (img.shape, (H, W))
    crops = {c.name: c for c in cfg.report.crops}

    frames = []
    for i in range(1, n + 1):
        m = cells == i
        rr, cc = np.nonzero(m)
        b0, b1, a0, a1 = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
        r0, r1, c0, c1 = largest_rect(m[b0:b1, a0:a1])
        r0, r1, c0, c1 = r0 + b0, r1 + b0, c0 + a0, c1 + a0
        module = "A" if c1 <= W / 2 else "B"
        # в PNG строка 0 — верх: строки сетки r → H-1-r
        top, bottom = H - r1, H - r0
        inner = [name for name, c in crops.items()
                 if r0 <= float(wcs.world_to_pixel_values(c.ra, c.dec)[1]) < r1
                 and c0 <= float(wcs.world_to_pixel_values(c.ra, c.dec)[0]) < c1]
        frames.append({"module": module, "png_rows": [int(top), int(bottom)], "png_cols": [int(c0), int(c1)],
                       "w_px": int(c1 - c0), "h_px": int(r1 - r0), "w_arcsec": round((c1 - c0) * SCALE, 1),
                       "h_arcsec": round((r1 - r0) * SCALE, 1), "config_crops_inside": inner,
                       "frame": bool((c1 - c0) >= MIN_SIDE_PX and (r1 - r0) >= MIN_SIDE_PX)})
    # порядок: модуль, сверху вниз, слева направо; имена A1… внутри модуля
    frames.sort(key=lambda f: (f["module"], f["png_rows"][0] // 200, f["png_cols"][0]))
    for mod in ("A", "B"):
        k = 0
        for f in frames:
            if f["module"] == mod:
                if f["frame"]:
                    k += 1
                    f["name"] = f"{mod}{k}"
                else:
                    f["name"] = f"{mod}-полоска"
    json.dump(frames, open(out / "кадры_без_щелей.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for f in frames:
        print(f["name"], f["png_rows"], f["png_cols"], f'{f["w_px"]}×{f["h_px"]} px = {f["w_arcsec"]}×{f["h_arcsec"]}″',
              f["config_crops_inside"])

    font = ImageFont.truetype("arial.ttf", 22)
    small = ImageFont.truetype("arial.ttf", 18)
    # обзор: финал в 1/3, щели красным, рамки кадров
    s = 3
    ov = img[::s, ::s].copy()
    g = np.flipud(gap & inside)[::s, ::s]
    ov[g] = (ov[g] * 0.4 + np.array([200, 0, 0]) * 0.6).astype(np.uint8)
    ovi = Image.fromarray(ov)
    d = ImageDraw.Draw(ovi)
    for f in frames:
        (t, b), (l, r) = f["png_rows"], f["png_cols"]
        col = (255, 255, 0) if f["frame"] else (0, 200, 255)
        d.rectangle([l / s, t / s, r / s, b / s], outline=col, width=3 if f["frame"] else 1)
        if f["frame"]:
            d.text((l / s + 6, t / s + 4), f["name"], fill=col, font=font)
    head = Image.new("RGB", (ovi.width, 40), (30, 30, 30))
    ImageDraw.Draw(head).text((8, 8), "W3-2 showcase 01-40-58, 1/3: щели и пустоты (красное), кадры без щелей "
                              f"(жёлтые, отступ {MARGIN_PX} px), полоски меньше 20″ (голубые)",
                              fill=(255, 255, 255), font=small)
    panels = [head, ovi]
    # кропы при 100%: по модулям, в ряд
    for mod in ("A", "B"):
        fr = [f for f in frames if f["module"] == mod and f["frame"]]
        tiles = []
        for f in fr:
            (t, b), (l, r) = f["png_rows"], f["png_cols"]
            tile = Image.fromarray(img[t:b, l:r])
            cap = Image.new("RGB", (tile.width, 92), (30, 30, 30))
            dc = ImageDraw.Draw(cap)
            dc.text((6, 4), f'{f["name"]} — модуль {mod}, {f["w_px"]}×{f["h_px"]} px = {f["w_arcsec"]}×{f["h_arcsec"]}″',
                    fill=(255, 255, 0), font=small)
            dc.text((6, 32), f"PNG: строки {t}–{b}, столбцы {l}–{r}", fill=(200, 200, 200), font=small)
            dc.text((6, 60), CONTENT.get(f["name"], ""), fill=(200, 200, 200), font=small)
            both = Image.new("RGB", (tile.width, tile.height + cap.height), (30, 30, 30))
            both.paste(cap, (0, 0))
            both.paste(tile, (0, cap.height))
            tiles.append(both)
        if tiles:
            row = Image.new("RGB", (sum(t.width + 12 for t in tiles), max(t.height for t in tiles)), (60, 60, 60))
            x = 0
            for t in tiles:
                row.paste(t, (x, 0))
                x += t.width + 12
            panels.append(row)
    sheet = Image.new("RGB", (max(p.width for p in panels), sum(p.height + 12 for p in panels)), (60, 60, 60))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 12
    sheet.save(out / "кадры_без_щелей.png")
    print(out / "кадры_без_щелей.png", sheet.size)


if __name__ == "__main__":
    main()
