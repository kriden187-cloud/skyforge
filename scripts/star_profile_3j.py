"""3j: профиль звезды по радиусу — наша картинка(и), NASA на нашей сетке, отдельная экспозиция F090W (линейная,
радиус пересчитан в пиксели нашей сетки). Нормировка на 30 px, чтобы сравнивать форму, а не яркость.

Запуск: uv run python scripts/star_profile_3j.py <папка> <имя> <x> <y> <png> [<png> ...]
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def radial(img2d: np.ndarray, x: float, y: float, rmax: int, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices(img2d.shape)
    r = np.hypot(xx - x, yy - y) * scale
    edges = np.arange(0, rmax, 1.0)
    prof = np.array([np.nanmedian(img2d[(r >= a) & (r < a + 1)]) if ((r >= a) & (r < a + 1)).any() else np.nan for a in edges])
    return edges + 0.5, prof


def main() -> None:
    out, name, x, y, pngs = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5:]
    R = 80
    curves = []
    first = None
    for p in pngs:
        im = np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255
        first = im if first is None else first
        sub = im[y - R:y + R, x - R:x + R] @ LUMA
        curves.append((pathlib.Path(p).parent.name[-22:], *radial(sub, R, R, R)))
    nasa = nasa_on_grid(first, "data/reference/ngc3132_nasa.jpg")[::-1]   # в ориентации картинки
    curves.insert(0, ("NASA", *radial(nasa[y - R:y + R, x - R:x + R] @ LUMA, R, R, R)))
    # отдельная экспозиция F090W: звезда по координатам неба с нашей сетки
    m = json.load(open("data/work/ngc3132/bin1/aligned/meta.json", encoding="utf-8"))
    w = WCS(fits.Header.fromstring(m["wcs"], sep="\n"))
    H = m["shape"][0]
    sky = w.pixel_to_world(x, H - 1 - y)
    cal = None
    for f in sorted(pathlib.Path("data/raw").rglob("*nrcb*_cal.fits")):
        with fits.open(f, memmap=True) as h:
            if h[0].header.get("FILTER") != "F090W":
                continue
            ww = WCS(h["SCI"].header)
            px, py = ww.world_to_pixel(sky)
            if 100 < px < 1948 and 100 < py < 1948:
                cal = (f.name, np.array(h["SCI"].data[int(py) - 2 * R:int(py) + 2 * R, int(px) - 2 * R:int(px) + 2 * R]),
                       float(px) - int(px) + 2 * R, float(py) - int(py) + 2 * R)
                break
    if cal:
        rr, pr = radial(cal[1], cal[2], cal[3], R, scale=0.031 / 0.063)
        curves.append((f"экспозиция {cal[0][:30]} (F090W, линейная)", rr, pr))
    fig, ax = plt.subplots(figsize=(9, 5))
    for lab, r, pr in curves:
        k = np.nanmedian(pr[28:32])
        ax.plot(r, pr / (k if k else 1), label=lab)
    ax.set_yscale("log"); ax.set_xlabel("радиус, px нашей сетки"); ax.set_ylabel("яркость / яркость на 30 px")
    ax.set_title("Профиль звезды (медиана по кольцу): плато и обрыв видны как излом"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=90)
    for lab, r, pr in curves:
        print(lab, " ".join(f"{int(rr)}:{v * (255 if 'экспоз' not in lab else 1):.1f}" for rr, v in zip(r[[0, 3, 6, 9, 12, 15, 20, 30, 45, 60]], pr[[0, 3, 6, 9, 12, 15, 20, 30, 45, 60]])))


if __name__ == "__main__":
    main()
