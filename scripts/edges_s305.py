"""Фаза 4, шаг 4d: края и небо готовой картинки Sh2-305 (bin 1) — те же области, что лист 79 (scripts/skyref2_s305.py):
медиана R/G/B в четырёх углах 600 px и в нижней полосе 300 px (поле всех фильтров, без звёзд) и неба в кропе sky.

Запуск: uv run python scripts/edges_s305.py <png bin 1>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import display_crop

Image.MAX_IMAGE_PIXELS = None
BAND, CSQ = 300, 600


def main() -> None:
    img = np.asarray(Image.open(sys.argv[1]).convert("RGB"), dtype=np.float32)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(c, P)
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    stars = ndimage.binary_dilation(np.load(clean_dir(c, P) / "star_weight.npy") > 0.01, iterations=3)
    sky = load_sky_mask(c, P)
    H, W = full.shape
    ok_img = (full & ~stars)[::-1]
    corners = np.zeros((H, W), bool)
    corners[:CSQ, :CSQ] = corners[:CSQ, -CSQ:] = corners[-CSQ:, :CSQ] = corners[-CSQ:, -CSQ:] = True
    rows = np.nonzero(full[::-1].any(1))[0]
    bottom = np.zeros((H, W), bool)
    bottom[rows[-1] - BAND + 1:rows[-1] + 1, :] = True
    k = {x.name: x for x in with_resolved_crops(c, P).report.crops}["sky"]
    gs = display_crop((H, W), k.x, k.y, k.size, 1)
    ssl = (slice(H - gs[0].stop, H - gs[0].start), gs[1])
    sky_in = (sky & full)[::-1][ssl] & ok_img[ssl]
    med = lambda v: " / ".join(f"{np.median(v[:, i]):.0f}" for i in range(3))
    print(f"углы 600 px {med(img[corners & ok_img])} · нижние 300 px {med(img[bottom & ok_img])} · небо sky {med(img[ssl][sky_in])}")


if __name__ == "__main__":
    main()
