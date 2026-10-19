"""3j: кропы 80 px ×3 трёх звёзд с насыщенным ядром и трёх ярких звёзд поля — варианты рядом с NASA.

Запуск: uv run python scripts/stars_sheet_3j.py <папка> <имя> <png>=<подпись> [...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.clean import clean_dir
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import OURS_STARS, nasa_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])


def field_stars(img: np.ndarray, n: int = 3) -> list[tuple[int, int]]:
    """Яркие компактные звёзды на пустом небе (вне маски туманности, далеко от звёзд с насыщенным ядром)."""
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    sd = np.load(clean_dir(cfg, P) / "nebula_sd.npy")[::-1]
    g = ndimage.gaussian_filter(img @ LUMA, 1.5)
    pk = np.argwhere((g == ndimage.maximum_filter(g, 41)) & (sd < -60))
    pk = [(int(x), int(y)) for y, x in pk if 50 < x < img.shape[1] - 50 and 50 < y < img.shape[0] - 50
          and all(np.hypot(x - sx, y - sy) > 150 for sx, sy in OURS_STARS)]
    pk.sort(key=lambda p: -g[p[1], p[0]])
    return pk[:n]


def main() -> None:
    out, name = pathlib.Path(sys.argv[1]), sys.argv[2]
    variants = [tuple(a.rsplit("=", 1)) for a in sys.argv[3:]]
    imgs = [(lab, np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255) for p, lab in variants]
    nasa = nasa_on_grid(imgs[0][1], "data/reference/ngc3132_nasa.jpg")[::-1]
    stars = [("центральная", OURS_STARS[0]), ("внизу справа", OURS_STARS[1]), ("слева", OURS_STARS[2])]
    stars += [(f"поле {i + 1}", p) for i, p in enumerate(field_stars(imgs[0][1]))]
    tiles, labels = [], []
    for sname, (x, y) in stars:
        for lab, im in [("NASA", nasa)] + imgs:
            c = im[y - 40:y + 40, x - 40:x + 40]
            tiles.append(np.kron(c, np.ones((3, 3, 1)))[::-1])
            labels.append(f"{sname} ({x}, {y}) · {lab}")
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(imgs) + 1, tile_inches=3.0, interpolation="nearest",
                  title="Звёзды 80 px ×3: спад плавный, без диска и ступеньки на краю, без каймы на лучах")
    print("звёзды поля:", stars[3:])


if __name__ == "__main__":
    main()
