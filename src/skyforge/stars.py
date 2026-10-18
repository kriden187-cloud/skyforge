"""Насыщенные ядра звёзд.

В i2d уровня 3 нет слоя DQ: пиксели, которые конвейер STScI отбраковал (в том числе
насыщенные), приходят как NaN с WHT=0. Ядро яркой звезды — небольшая дыра NaN внутри
поля. Если её оставить, в одних фильтрах ядро пустое, в других яркое, и после
смешивания в центре звезды цветная точка.

Дыру заполняем кольцами снаружи внутрь: каждый пиксель кольца получает максимум уже
известных соседей. Ядро становится не темнее своего края, и после растяжки все фильтры
упираются в белую точку — ядро белое.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_EIGHT = np.ones((3, 3), dtype=bool)


def core_area_px(max_core_arcsec: float, pixel_scale_arcsec: float) -> int:
    r = max_core_arcsec / 2 / pixel_scale_arcsec
    return max(1, int(math.ceil(math.pi * r * r)))


def find_core_holes(a: np.ndarray, max_area_px: int) -> np.ndarray:
    """Дыры NaN внутри поля, не больше max_area_px. Пустые края мозаики и длинные пропуски не трогаем."""
    nan = ~np.isfinite(a)
    labels, n = ndimage.label(nan, structure=_EIGHT)
    if n == 0:
        return np.zeros(a.shape, dtype=bool)
    sizes = ndimage.sum_labels(nan, labels, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes <= max_area_px
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep[border] = False
    return keep[labels]


def fill_rings(a: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Заполнить дыры снаружи внутрь максимумом известных соседей. Остальные NaN не меняются."""
    out = np.array(a, dtype=np.float32, copy=True)
    labels, _ = ndimage.label(holes, structure=_EIGHT)
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        # Окно с запасом в пиксель, чтобы видеть край дыры.
        win = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        sub = out[win]
        todo = holes[win].copy()
        while todo.any():
            known = np.where(np.isfinite(sub) & ~todo, sub, -np.inf)
            nb = ndimage.maximum_filter(known, footprint=_EIGHT, mode="constant", cval=-np.inf)
            ring = todo & np.isfinite(nb)
            if not ring.any():
                break  # дыра без известных соседей: оставляем NaN
            sub[ring] = nb[ring]
            todo &= ~ring
    return out
