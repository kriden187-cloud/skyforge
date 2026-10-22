"""OMC3-N: красные точки у протозвёзд — лист 04: точки итога (красный над зелёным и синим, выше окрестности) против
масок заполнения clean по фильтрам (дыры NaN в выровненных слоях: заполнены как ядра или гладко как области).

Запуск: uv run python scripts/red_dots_omc3n.py <png итога> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet
from skyforge.stars import find_core_holes

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/omc3n/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _, meta = load_aligned(c, P)
    img = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)[::-1]          # сетка: строка 0 — низ
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    red = (r - np.maximum(g, b) > 40) & (r - ndimage.median_filter(r, 7) > 25)
    holes = {f: find_core_holes(a, 10 ** 7) for f, a in arrays.items()}
    near = {f: ndimage.binary_dilation(h, iterations=1) for f, h in holes.items()}
    lab, n = ndimage.label(red, structure=np.ones((3, 3)))
    on = {f: int(sum(near[f][lab == i + 1].any() for i in range(n))) for f in holes}
    on_any = int(sum(np.logical_or.reduce([near[f] for f in holes])[lab == i + 1].any() for i in range(n)))
    line = (f"красных точек {n}; на месте заполненных дыр (±1 px) {on_any} — по фильтрам "
            + ", ".join(f"{f} {k}" for f, k in on.items()))
    print(line)
    # кроп 150 px ×3 вокруг самого плотного скопления точек
    dens = ndimage.uniform_filter(red.astype(np.float32), 150)
    y, x = np.unravel_index(int(np.argmax(dens)), dens.shape)
    sl = (slice(max(y - 75, 0), y + 75), slice(max(x - 75, 0), x + 75))
    t = np.clip(img[sl] / 255 * 3, 0, 1)
    over = t.copy()
    colors = {"F150W": (0.3, 0.5, 1), "F200W": (0, 1, 1), "F356W": (1, 0.6, 0), "F480M": (1, 0.1, 0.1)}
    for f, col in colors.items():
        over[holes[f][sl]] = col
    mark = img[sl] / 255
    mark = np.clip(mark * 3, 0, 1)
    mark[red[sl]] = (1, 1, 1)
    contact_sheet([t[::-1], over[::-1], mark[::-1]],
                  [f"итог ×3, кроп 150 px у (строка PNG {meta['shape'][0] - 1 - y}, столбец {x})",
                   "дыры по фильтрам: F150W синий, F200W голубой, F356W оранжевый, F480M красный",
                   "красные точки (белым)"],
                  out / "04_красные_точки.png", cols=3, tile_inches=5, interpolation="nearest",
                  title="OMC3-N финал: красные точки у протозвезды и маски заполнения — " + line)
    (out / "04_красные_точки.txt").write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
