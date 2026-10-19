"""Поле статистики (3h, п.4): где считаются пороги, процентили и цвет неба.

При align.crop: union кадр шире поля, где есть все фильтры, а края шумнее и с неполным набором
фильтров. Если пороги масок, процентили растяжки, чёрную точку и карту цвета неба считать по
всему кадру, они зависят от того, насколько кадр расширен, и внутри прежней границы итог
меняется. Поэтому статистика считается в прямоугольнике общего поля (intersection из метаданных
align) и только по пикселям, где есть все фильтры; к краям union результат применяется как есть.

Блочные операции (медиана фона под звездой, заполнение звёзд) ведут сетку блоков от начала
этого прямоугольника — тогда блоки внутри прежней границы одни и те же при любом расширении кадра.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class StatsField:
    rows: slice
    cols: slice
    full: np.ndarray  # bool, размер прямоугольника: есть все фильтры

    @property
    def origin(self) -> tuple[int, int]:
        return int(self.rows.start or 0), int(self.cols.start or 0)

    def values(self, a: np.ndarray) -> np.ndarray:
        """Значения для статистики: прямоугольник поля, вне полного набора фильтров — NaN."""
        x = np.asarray(a, dtype=np.float32)[self.rows, self.cols]
        return np.where(self.full, x, np.nan)

    def mask(self, m: np.ndarray) -> np.ndarray:
        return np.asarray(m, dtype=bool)[self.rows, self.cols] & self.full

    def paste(self, sub: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Карта, посчитанная в прямоугольнике поля, → весь кадр; за краем поля — значения края."""
        r0, c0 = self.origin
        pad = [(r0, shape[0] - r0 - sub.shape[0]), (c0, shape[1] - c0 - sub.shape[1])] + [(0, 0)] * (sub.ndim - 2)
        return np.pad(sub, pad, mode="edge")


def make_field(arrays: list[np.ndarray], rect: list[int] | None) -> StatsField:
    """rect — [r0, r1, c0, c1] общего поля в координатах кадра (meta["intersection"] этапа align);
    None — весь кадр (старый кеш, кадр intersection)."""
    h, w = np.asarray(arrays[0]).shape[:2]
    r0, r1, c0, c1 = rect if rect else (0, h, 0, w)
    rows, cols = slice(r0, r1), slice(c0, c1)
    full = np.logical_and.reduce([np.isfinite(np.asarray(a)[rows, cols]) for a in arrays])
    return StatsField(rows, cols, full)


def block_median_map(a: np.ndarray, window_px: int, origin: tuple[int, int] = (0, 0),
                     fill_empty: bool = False) -> np.ndarray:
    """Медиана местного фона в окне window_px, на уменьшенной копии (блоки ~window/16): средние блоков
    без NaN → медиана по окну → линейная интерполяция до полного размера по центрам блоков.
    Сетка блоков начинается в origin (а не в углу кадра). fill_empty — пустые блоки (все NaN)
    заполняются ближайшим непустым."""
    a = np.asarray(a, dtype=np.float32)
    h, w = a.shape
    f = max(1, window_px // 16)
    pr, pc = (-origin[0]) % f, (-origin[1]) % f
    er, ec = (-(h + pr)) % f, (-(w + pc)) % f
    x = np.pad(a, ((pr, er), (pc, ec)), constant_values=np.nan)
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            small = np.nanmean(x.reshape(x.shape[0] // f, f, x.shape[1] // f, f), axis=(1, 3))
    bad = ~np.isfinite(small)
    if fill_empty and bad.any() and not bad.all():
        small = small[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    med = ndimage.median_filter(small, size=max(3, window_px // f) | 1, mode="nearest")
    yy = (np.arange(x.shape[0]) - (f - 1) / 2) / f
    xx = (np.arange(x.shape[1]) - (f - 1) / 2) / f
    up = ndimage.map_coordinates(med, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest")
    return up[pr:pr + h, pc:pc + w].astype(np.float32)
