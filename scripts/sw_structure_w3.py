"""Фаза 6, W3-2: есть ли рисунок пыли в отложенных F115W, F140M, F162M — лист 42.

Без скачивания i2d (1,6 ГБ на фильтр): quicklook-превью MAST из кеша discover (8-битный JPEG) и WCS из заголовка SCI
этого i2d — первые 300 КБ файла запросом Range. Кроп lanes из configs/w3_2 (центр ra, dec; сторона 50″) в каждом
фильтре, asinh по процентилям кропа; для сравнения тем же способом — F182M (в сборке) и F444W.
Мера рисунка: σ средних по ячейкам 8″ в кропе / σ шума ячейки (шум — по разностям соседних пикселей / размер ячейки):
у ровного поля ≈ 1, у пыли — десятки. Звёзды не маскируются: у всех фильтров одни и те же.

Запуск: uv run python scripts/sw_structure_w3.py <папка>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

from skyforge.config import Config
from skyforge.discover import MAST_DOWNLOAD_URL, UA
from skyforge.preview import contact_sheet

Image.MAX_IMAGE_PIXELS = None
FILTERS = (("F115W", "clear-f115w"), ("F140M", "clear-f140m"), ("F162M", "f150w2-f162m"), ("F182M", "clear-f182m"),
           ("F444W", "clear-f444w"))
SIDE_ARCSEC, CELL_ARCSEC = 50.0, 8.0


def sci_header(product: str) -> fits.Header:
    r = requests.get(MAST_DOWNLOAD_URL, params={"uri": f"mast:JWST/product/{product}"},
                     headers={**UA, "Range": "bytes=0-299999"}, timeout=(30, 120))
    r.raise_for_status()
    raw, pos, hdrs = r.content, 0, []
    while len(hdrs) < 2:
        h, end = None, pos
        while end + 2880 <= len(raw):
            block = raw[end:end + 2880].decode("ascii", "replace")
            end += 2880
            if any(block[i:i + 80].startswith("END") and block[i:i + 80].strip() == "END" for i in range(0, 2880, 80)):
                h = fits.Header.fromstring(raw[pos:end].decode("ascii", "replace"))
                break
        if h is None:
            raise RuntimeError(f"{product}: заголовок не уместился в 300 КБ")
        hdrs.append(h)
        pos = end                                   # у первичного HDU данных нет (NAXIS 0)
    return hdrs[1]


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    c = Config.load("configs/w3_2/showcase.yaml", "showcase")
    lanes = next(k for k in c.report.crops if k.name == "lanes")
    tiles, labels, rows = [], [], ["фильтр\tразмер превью / i2d\tмасштаб превью, ″/px\tрисунок (σ ячеек 8″ / шум)"]
    for f, tag in FILTERS:
        product = f"jw05437-o002_t002_nircam_{tag}_i2d.fits"
        jpg = pathlib.Path("data/cache/discover") / product.replace(".fits", ".jpg")
        h = sci_header(product)
        im = Image.open(jpg).convert("L")
        sx, sy = im.width / h["NAXIS1"], im.height / h["NAXIS2"]
        w = WCS(h)
        col, row = w.world_to_pixel_values(lanes.ra, lanes.dec)
        scale = float(np.sqrt(abs(np.linalg.det(w.pixel_scale_matrix)))) * 3600 / sx       # ″ на пиксель превью
        half = int(SIDE_ARCSEC / scale / 2)
        cx, cy = int(col * sx), int((h["NAXIS2"] - 1 - row) * sy)         # JPEG: строка 0 — верх
        a = np.asarray(im, dtype=np.float32)[max(cy - half, 0):cy + half, max(cx - half, 0):cx + half]
        valid = a[(a > 0) & (a < 255)]
        lo, hi = np.percentile(valid, [1, 99.5]) if valid.size else (0, 255)
        show = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1), 0, None) * 10) / np.arcsinh(10)
        cell = max(2, int(CELL_ARCSEC / scale))
        hh, ww = (a.shape[0] // cell) * cell, (a.shape[1] // cell) * cell
        means = a[:hh, :ww].reshape(hh // cell, cell, ww // cell, cell).mean(axis=(1, 3))
        d = np.diff(a, axis=1).ravel()
        noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)
        score = float(means.std() / max(noise / cell, 1e-3))
        tiles.append(show); labels.append(f"{f} · lanes 50″ · quicklook, asinh\nрисунок {score:.0f}")
        rows.append(f"{f}\t{im.width}×{im.height} / {h['NAXIS1']}×{h['NAXIS2']}\t{scale:.3f}\t{score:.0f}")
        print(rows[-1], flush=True)
    contact_sheet(tiles, labels, out / "42_F115W_F140M_F162M_lanes.png", cols=len(tiles), tile_inches=4.2,
                  interpolation="lanczos",
                  title="W3-2: кроп lanes в отложенных коротких фильтрах (quicklook MAST, 8 бит) рядом с F182M и F444W")
    (out / "42_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
