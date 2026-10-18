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
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

from dataclasses import dataclass

from skyforge.config import StarsConfig
from skyforge.stretch import sample

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


@dataclass
class StarMask:
    """Растушёванная маска звёзд (0–1) и окно местного фона в пикселях той же сетки."""

    weight: np.ndarray
    window_px: int
    detect: str  # фильтр, в котором искались звёзды: по нему считается доля звёздного света на туманности

    def crop(self, sl: tuple[slice, slice]) -> StarMask:
        return StarMask(self.weight[sl], self.window_px, self.detect)

    def shrink(self, factor: int) -> StarMask:
        from skyforge.preview import nan_block_mean

        return StarMask(nan_block_mean(self.weight, factor), max(3, self.window_px // max(factor, 1)), self.detect)


def star_mask(a: np.ndarray, s: StarsConfig, scale_arcsec: float) -> tuple[np.ndarray, np.ndarray]:
    """Маска звёзд поля и маска плотной туманности для одного фильтра.

    Звезда — компактная связная область выше mask_sigma с пиком выше peak_sigma и площадью
    не больше max_star_area_arcsec2. Всё, что больше, — плотная часть туманности: её пиксели
    из маски исключаются жёстко. Слабые волокна защищает уже не маска, а доля звёздного
    света в пикселе (compose.desaturate_stars).
    """
    smp = sample(a)
    med, _, std = sigma_clipped_stats(smp[np.isfinite(smp)], sigma=3.0)
    excess = np.where(np.isfinite(a), a - np.float32(med), 0)
    det = excess > s.mask_sigma * std
    labels, n = ndimage.label(det, structure=_EIGHT)
    if n == 0:
        empty = np.zeros(a.shape, dtype=bool)
        return empty, empty
    idx = np.arange(1, n + 1)
    area = ndimage.sum_labels(det, labels, index=idx)
    peak = ndimage.maximum(excess, labels, index=idx)
    max_area = s.max_star_area_arcsec2 / scale_arcsec**2
    is_star = np.zeros(n + 1, dtype=bool)
    is_star[1:] = (area <= max_area) & (peak >= s.peak_sigma * std)
    is_nebula = np.zeros(n + 1, dtype=bool)
    is_nebula[1:] = area > max_area
    stars = is_star[labels]
    r = int(round(s.halo_arcsec / scale_arcsec))
    if r > 0:
        disk = np.hypot(*np.indices((2 * r + 1, 2 * r + 1)) - r) <= r  # круглый ореол, не квадрат
        stars = ndimage.maximum_filter(stars.astype(np.uint8), footprint=disk) > 0
    return stars, is_nebula[labels]


def star_weight(masks: dict[str, tuple[np.ndarray, np.ndarray]], detect: str, s: StarsConfig,
                scale_arcsec: float) -> StarMask:
    """Звёзды из фильтра detect, плотная туманность — объединение по всем фильтрам; край растушёван."""
    stars = masks[detect][0]
    nebula = np.logical_or.reduce([m[1] for m in masks.values()])
    w = (stars & ~nebula).astype(np.float32)
    feather = s.halo_arcsec / scale_arcsec / 3
    if feather > 0:
        w = ndimage.gaussian_filter(w, feather)
    w[nebula] = 0  # растушёвка не должна заползать на туманность
    return StarMask(w, max(3, int(round(s.local_window_arcsec / scale_arcsec))), detect)
