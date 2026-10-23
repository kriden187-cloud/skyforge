"""Фаза 8: «phase-7 | phase-8» — кропы отчёта двух прогонов рядом при 100% и ×5 (центр кропа, пиксели без сглаживания).

Кропы report.crops заданы на небе (ra, dec) и одного размера, поэтому crop_<имя>.png двух прогонов — одно и то же место
неба, даже если кадры обрезаны по-разному.

Запуск: uv run python scripts/compare_runs.py <прогон A> <прогон B> <папка> <номер листа> <подпись A> <подпись B> <кроп> [...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.preview import contact_sheet


def main() -> None:
    run_a, run_b, out, num, la, lb, *crops = sys.argv[1:]
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tiles, labels = [], []
    for name in crops:
        imgs = [np.asarray(Image.open(pathlib.Path(r) / f"crop_{name}.png").convert("RGB"), dtype=np.float32) / 255
                for r in (run_a, run_b)]
        for img, lab in zip(imgs, (la, lb)):
            tiles.append(img)
            labels.append(f"{name} — {lab}, 100%")
        for img, lab in zip(imgs, (la, lb)):
            h, w = img.shape[:2]
            s = max(8, min(h, w) // 5)
            c = img[h // 2 - s // 2:h // 2 + s // 2, w // 2 - s // 2:w // 2 + s // 2]
            tiles.append(np.kron(c, np.ones((5, 5, 1), dtype=np.float32)))
            labels.append(f"{name} — {lab}, ×5 (центр)")
    contact_sheet(tiles, labels, out / f"{num}_{la}_{lb}.png", cols=4, tile_inches=4.0, interpolation="nearest",
                  title=f"OMC3-N: {la} | {lb} — кропы на небе при 100% и ×5")


if __name__ == "__main__":
    main()
