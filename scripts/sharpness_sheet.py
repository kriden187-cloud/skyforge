"""Фаза 8, лист 39: три варианта резкости на одних и тех же местах неба при 100% и мера мелкой фактуры.

Варианты — прогоны с кропами report.crops (crop_<имя>.png). Кропы заданы в px сетки, поэтому у прогона на вдвое более
мелкой сетке (0,031″) кроп покрывает вдвое меньше неба: у остальных берётся центральная половина — место неба одно.
Мера мелкой фактуры — σ(L − G(L, σ 0,1″)) / среднее L по яркости L = среднее RGB: доля яркости в деталях мельче 0,1″.
Выводов нет — выбор владельца.

Запуск: uv run python scripts/sharpness_sheet.py <папка> <прогон:подпись:″/px> [...] -- <кроп> [...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.preview import contact_sheet

FINE_ARCSEC = 0.1


def main() -> None:
    args = sys.argv[1:]
    out = pathlib.Path(args[0])
    sep = args.index("--")
    variants = [a.split(":") for a in args[1:sep]]
    crops = args[sep + 1:]
    finest = min(float(v[2]) for v in variants)
    tiles, labels, rows = [], [], []
    for name in crops:
        for run, label, scale in variants:
            img = np.asarray(Image.open(pathlib.Path(run) / f"crop_{name}.png").convert("RGB"), dtype=np.float32) / 255
            k = float(scale) / finest                      # во сколько раз пиксель крупнее самого мелкого
            if k > 1.01:                                   # то же небо: центральная доля 1/k
                h, w = img.shape[:2]
                sh, sw = int(round(h / k)), int(round(w / k))
                img = img[(h - sh) // 2:(h - sh) // 2 + sh, (w - sw) // 2:(w - sw) // 2 + sw]
            lum = img.mean(axis=2)
            fine = float(np.std(lum - ndimage.gaussian_filter(lum, FINE_ARCSEC / float(scale))) / max(lum.mean(), 1e-6))
            rows.append(f"{name}\t{label}\t{fine:.4f}")
            tiles.append(img)
            labels.append(f"{name} — {label}\n100% ({img.shape[1]}×{img.shape[0]} px), фактура < 0,1″: {fine:.4f}")
    out.mkdir(parents=True, exist_ok=True)
    contact_sheet(tiles, labels, out / "39_резкость.png", cols=len(variants), tile_inches=5.0, interpolation="nearest",
                  title="OMC3-N phase-8: три варианта резкости, одно место неба при 100% каждого; мера — σ(L − размытие "
                        "0,1″) / среднее L")
    (out / "39_резкость.tsv").write_text("кроп\tвариант\tфактура < 0,1″\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
