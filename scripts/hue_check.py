"""Фаза 8: цвет туманности в кропах двух прогонов — оттенок h и насыщенность C в CIELAB по пикселям туманности
(светлота L 20–95: без неба и выжженного). Сдвиг оттенка к красному — уменьшение h к ~40° (красный в CIELAB около 40°,
оранжевый ~60°, жёлтый ~90°, голубой ~200°, синий ~280°).

Запуск: uv run python scripts/hue_check.py <прогон A> <прогон B> <кроп> [...]
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from PIL import Image
from skimage.color import rgb2lab


def hue(run: str, name: str) -> tuple[float, float, float]:
    rgb = np.asarray(Image.open(pathlib.Path(run) / f"crop_{name}.png").convert("RGB"), dtype=np.float64) / 255
    lab = rgb2lab(rgb)
    m = (lab[..., 0] > 20) & (lab[..., 0] < 95)
    a, b = float(np.mean(lab[..., 1][m])), float(np.mean(lab[..., 2][m]))
    return math.degrees(math.atan2(b, a)) % 360, math.hypot(a, b), float(np.mean(lab[..., 0][m]))


def main() -> None:
    ra, rb, *crops = sys.argv[1:]
    for name in crops:
        ha, ca, la = hue(ra, name)
        hb, cb, lb = hue(rb, name)
        dh = (hb - ha + 180) % 360 - 180
        print(f"{name}: оттенок {ha:.1f}° → {hb:.1f}° (Δ {dh:+.1f}°), насыщенность {ca:.1f} → {cb:.1f}, "
              f"светлота {la:.1f} → {lb:.1f}")


if __name__ == "__main__":
    main()
