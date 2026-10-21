"""Фаза 4: проверка пробы «опорное небо по кропу sky» на полосы и ступеньки — лист 78b, bin 1.

Финал против пробы: sky ×5, sky2 ×3, seam ×2, gap212 ×5 при 100%, рядом карта областей CON F444W и F182M.

Запуск: uv run python scripts/skyref_check_s305.py <png финала> <png пробы> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import aligned_dir, load_aligned, with_resolved_crops
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    fin, probe, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    load_aligned(c, P)
    imgs = [("финал", np.asarray(Image.open(fin).convert("RGB"), dtype=np.float32) / 255),
            ("проба", np.asarray(Image.open(probe).convert("RGB"), dtype=np.float32) / 255)]
    H, W = imgs[0][1].shape[:2]
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    tl = {f: np.load(aligned_dir(c, P) / f"{f}.tiles.npy") for f in ("F444W", "F182M")}
    pal = np.random.default_rng(0).uniform(0.15, 1, (4096, 3)).astype(np.float32)
    tiles, labels = [], []
    for name, k in (("sky", 5), ("sky2", 3), ("seam", 2), ("gap212", 5)):
        cr = crops[name]
        sl = display_crop((H, W), cr.x, cr.y, cr.size, 1)                    # срез сетки
        sli = (slice(H - sl[0].stop, H - sl[0].start), sl[1])                 # строки картинки
        for lab, im in imgs:
            tiles.append(np.clip(im[sli] * k, 0, 1)[::-1]); labels.append(f"{lab} · {name} ×{k}")
        for f in ("F444W", "F182M"):
            t = tl[f][sl]
            m = pal[np.where(t >= 0, t, -1).astype(int) % 4096]
            m[t < 0] = 0
            tiles.append(m); labels.append(f"{name} · области CON {f}")
    contact_sheet(tiles, labels, out / "78b_полосы_и_стыки.png", cols=4, tile_inches=4.6, interpolation="nearest",
                  title="Проба «опорное небо по кропу sky» против финала: небо, стык, щель SW при 100% с картами CON")


if __name__ == "__main__":
    main()
