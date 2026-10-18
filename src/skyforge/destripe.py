"""Полосы 1/f детекторов NIRCam: смещения целых строк (и иногда столбцов) внутри каждой экспозиции.

Работает на исходной сетке фильтра, до совмещения: полосы идут вдоль строк детектора, а после
поворота или пересэмплирования строка сетки уже не строка детектора. Внутри каждой области
слоя CON (свой набор экспозиций — свой рисунок полос) для каждой строки берётся медиана
пикселей вне маски источников. Строкам, где свободных пикселей мало, достаётся медиана,
интерполированная от соседних строк той же области.

По умолчанию (window_rows: 0) медиана строки вычитается целиком: вместе с полосами уходит и
уровень области, поэтому узкие полосы дизеринга тоже выравниваются. С window_rows: N вычитается
только отклонение от скользящей медианы по N строкам — уровень и плавный ход фона остаются,
а уровни областей выравнивает clean. Области без единой свободной строки (целиком под маской)
не трогаются: их пьедестал сшивается с соседями по границе в clean.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import ndimage

from skyforge.config import Config, DestripeConfig


def destripe_mask(a: np.ndarray, cfg: Config, scale_arcsec: float) -> np.ndarray:
    """Туманность и звёзды — существующие маски (маска плиток фона и маска звёзд), плюс grow_px."""
    from skyforge.background import source_mask
    from skyforge.stars import star_mask

    from astropy.stats import sigma_clipped_stats

    from skyforge.stretch import sample

    mask = source_mask(a, cfg.background, scale_arcsec, for_tiles=True)
    mask |= star_mask(a, cfg.stars, scale_arcsec)[0]
    c = cfg.clean.destripe
    smp = sample(a)
    med, _, std = sigma_clipped_stats(smp[np.isfinite(smp)], sigma=3.0)
    bright = np.nan_to_num(a, nan=-np.inf) > med + c.pixel_sigma * std
    if c.pixel_grow_px:
        bright = ndimage.binary_dilation(bright, iterations=c.pixel_grow_px)
    mask |= bright
    g = cfg.clean.destripe.mask_grow_px
    if g > 0:
        mask = ndimage.binary_dilation(mask, iterations=g)
    return mask | ~np.isfinite(a)


def _regions(labels: np.ndarray | None, finite: np.ndarray, min_region_px: int) -> np.ndarray:
    """Области CON; мелкие (точки после отбраковки лучей) приписываются ближайшей крупной."""
    if labels is None:
        return np.where(finite, 0, -1).astype(np.int32)
    lab = np.where(finite, labels, -1).astype(np.int32)
    counts = np.bincount(lab[lab >= 0].ravel(), minlength=int(lab.max()) + 1)
    big = counts >= min_region_px
    if not big.any():
        return np.where(finite, 0, -1).astype(np.int32)
    is_big = (lab >= 0) & big[np.clip(lab, 0, None)]
    if (finite & ~is_big).any():
        idx = ndimage.distance_transform_edt(~is_big, return_distances=False, return_indices=True)
        lab = np.where(finite, lab[tuple(idx)], -1)
    return lab


def _line_offsets(vals: np.ndarray, c: DestripeConfig, axis: int,
                  extent: np.ndarray | None = None) -> np.ndarray | None:
    """Медиана по линиям (строкам при axis=1) с интерполяцией через пустые линии.
    extent — длина каждой линии внутри области: линия, свободная меньше чем на min_free_fraction,
    считается пустой и получает значение от соседей."""
    counts = np.isfinite(vals).sum(axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(vals, axis=axis)
    good = counts >= c.min_pixels
    if extent is not None:
        good &= counts >= c.min_free_fraction * extent
    if not good.any():
        # Узкая область (полоса на краю дизеринга): ни одна строка не набирает min_pixels. Берём медиану
        # по группам соседних строк такой высоты, чтобы пикселей хватило; иначе область осталась бы
        # на прежнем уровне неба, а соседи ушли бы в ноль — светлый прямоугольник.
        per_line = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
        if per_line == 0:
            return None
        k = int(np.ceil(c.min_pixels / per_line))
        if k >= med.size:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                level = np.nanmedian(vals)
            return None if not np.isfinite(level) else np.full(med.size, level, dtype=np.float32)
        lines = vals if axis == 1 else vals.T
        pad = (-lines.shape[0]) % k
        padded = np.vstack([lines, np.full((pad, lines.shape[1]), np.nan, dtype=lines.dtype)])
        groups = padded.reshape(-1, k * lines.shape[1])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gmed = np.nanmedian(groups, axis=1)
        gcount = np.isfinite(groups).sum(axis=1)
        med = np.repeat(gmed, k)[: med.size]
        good = np.repeat(gcount >= c.min_pixels, k)[: med.size]
        if not good.any():
            return None
    idx = np.arange(med.size)
    return np.interp(idx, idx[good], med[good]).astype(np.float32)


def destripe(a: np.ndarray, labels: np.ndarray | None, mask: np.ndarray, c: DestripeConfig) -> tuple[np.ndarray, dict[str, Any]]:
    out = np.array(a, dtype=np.float32, copy=True)
    finite = np.isfinite(out)
    regions = _regions(labels, finite, c.min_pixels * c.min_region_rows)
    n_reg = 0
    passes = [1, 0] if c.columns else [1]
    for axis in passes:  # 1 — строки, 0 — столбцы
        # find_objects(regions + 1): элемент r — рамка области r (метка −1 «вне поля» становится 0 и пропускается).
        for r, sl in enumerate(ndimage.find_objects(regions + 1)):
            if sl is None:
                continue
            reg = regions[sl] == r
            sub = out[sl]
            vals = np.where(reg & ~mask[sl], sub, np.nan)
            off = _line_offsets(vals, c, axis, reg.sum(axis=axis))
            if off is None:
                continue
            # reflect, а не nearest: у края окно не должно заполняться копиями одной (шумной) строки.
            if c.window_rows:
                off = off - ndimage.median_filter(off, size=min(c.window_rows, off.size) | 1, mode="reflect")
            shaped = off[:, None] if axis == 1 else off[None, :]
            sub -= np.where(reg, shaped, 0).astype(np.float32)
            n_reg += axis == 1
    return out, {"regions": n_reg, "columns": c.columns}


def step_metric(a: np.ndarray, mask: np.ndarray, labels: np.ndarray | None, c: DestripeConfig,
                axis: int = 1) -> float:
    """Мера полос: среднее |медиана строки − медиана соседней строки| по пустому небу внутри областей CON
    (axis=0 — то же по столбцам). Только быстрая часть: плавный ход фона в неё не входит."""
    regions = _regions(labels, np.isfinite(a), c.min_pixels * c.min_region_rows)
    total, n = 0.0, 0
    for r, sl in enumerate(ndimage.find_objects(regions + 1)):
        if sl is None:
            continue
        reg = regions[sl] == r
        vals = np.where(reg & ~mask[sl], a[sl], np.nan)
        counts = np.isfinite(vals).sum(axis=axis)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(vals, axis=axis)
        good = counts >= c.min_pixels
        both = good[1:] & good[:-1]
        if both.any():
            d = np.abs(np.diff(med))[both]
            total += float(d.sum())
            n += int(d.size)
    return total / n if n else 0.0
