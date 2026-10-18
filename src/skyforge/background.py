"""Фон до растяжки: ступеньки между плитками мозаики и крупные градиенты.

Шаг 1, плитки. В i2d слой CON говорит, какие экспозиции легли в каждый пиксель.
Области с одинаковым набором — прямоугольники и полосы от шаблона дизеринга, и у
каждой свой пьедестал (у NGC 3132 разброс до ±1σ шума фона). Каждой области —
постоянная поправка: медиана её свободных от источников пикселей минус общий фон.
Области почти целиком под туманностью берут поправку ближайшей соседки.

Шаг 2, гладкая модель:
1. Маска источников: кадр слегка сглаживается (так ловится и слабая туманность),
   всё выше mask_sigma шума фона маскируется и расширяется на mask_dilate_arcsec.
2. Сетка блоков box_arcsec: в каждом медиана с отсечкой по σ по свободным пикселям;
   блоки, где свободных меньше min_free, пропускаются и заполняются от соседей.
3. Сетка сглаживается медианой filter_size×filter_size блоков и интерполируется до
   полного размера (photutils.Background2D).
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter

from skyforge.config import BackgroundConfig
from skyforge.log import SkyforgeError
from skyforge.stretch import sample

CLIP_SIGMA = 3.0  # отсечка выбросов при оценке шума фона


def source_mask(a: np.ndarray, c: BackgroundConfig, scale_arcsec: float, for_tiles: bool = False) -> np.ndarray:
    """for_tiles — маска для поправок плиток. Порог по сглаженному кадру считается в единицах
    шума отдельного пикселя: иначе сама ступенька пьедестала (доли σ, но на тысячах пикселей)
    после сглаживания становится «источником» и прячет плитку под маску. Зато вокруг туманности
    маска шире (tile_mask_dilate_arcsec): полосы CON тянутся до края кадра, и оценивать их
    надёжнее по дальним частям, чем по ореолу."""
    finite = np.isfinite(a)
    smp = sample(a)
    med, _, std = sigma_clipped_stats(smp[np.isfinite(smp)], sigma=CLIP_SIGMA)  # std — шум пикселя
    s = np.where(finite, a - np.float32(med), 0).astype(np.float32)
    smooth_px = c.mask_smooth_arcsec / scale_arcsec
    if smooth_px > 0:
        s = gaussian_filter(s, smooth_px)
    if not for_tiles:
        smp_s = sample(np.where(finite, s, np.nan))
        _, _, std = sigma_clipped_stats(smp_s[np.isfinite(smp_s)], sigma=CLIP_SIGMA)
    mask = s > c.mask_sigma * std
    dilate = c.tile_mask_dilate_arcsec if for_tiles else c.mask_dilate_arcsec
    r = int(round(dilate / scale_arcsec))
    if r > 0:
        mask = maximum_filter(mask.astype(np.uint8), size=2 * r + 1) > 0
    return mask & finite


def background_model(a: np.ndarray, c: BackgroundConfig, scale_arcsec: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Модель фона того же размера, что и кадр, маска источников и размер блока в пикселях."""
    box = max(4, int(round(c.box_arcsec / scale_arcsec)))
    mask = source_mask(a, c, scale_arcsec)
    try:
        bkg = Background2D(
            a, box,
            mask=mask,
            coverage_mask=~np.isfinite(a),
            fill_value=np.nan,
            exclude_percentile=100 * (1 - c.min_free),
            filter_size=(c.filter_size, c.filter_size),
            bkg_estimator=MedianBackground(),
        )
    except ValueError as e:
        raise SkyforgeError(
            f"не получилось построить модель фона ({e}). Туманность занимает почти весь кадр: "
            f"уменьши background.box_arcsec или background.min_free"
        ) from e
    return bkg.background.astype(np.float32), mask, box


def tile_pedestal(a: np.ndarray, tiles: np.ndarray, c: BackgroundConfig,
                  scale_arcsec: float) -> tuple[np.ndarray, dict[str, int]]:
    """Карта пьедесталов по областям CON того же размера, что и кадр (NaN вне поля)."""
    finite = np.isfinite(a) & (tiles >= 0)
    free = finite & ~source_mask(a, c, scale_arcsec, for_tiles=True)
    if not free.any():
        return np.zeros_like(a), {"tiles": 0, "measured": 0}
    lab = np.where(free, tiles, -1)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    min_px = max(1, int(round(c.tile_min_area_arcsec2 / scale_arcsec**2)))
    good = ids[counts >= min_px]
    level = float(np.median(a[free]))
    offsets = np.full(int(tiles.max()) + 1, np.nan, dtype=np.float32)
    if good.size:
        offsets[good] = np.asarray(ndimage.median(a, labels=lab, index=good), dtype=np.float32) - level
    ped = np.where(tiles >= 0, offsets[np.clip(tiles, 0, None)], np.nan).astype(np.float32)
    unknown = finite & ~np.isfinite(ped)
    if unknown.any() and np.isfinite(ped).any():
        # Поправка ближайшей измеренной области.
        idx = ndimage.distance_transform_edt(~np.isfinite(ped), return_distances=False, return_indices=True)
        ped = ped[tuple(idx)]
    ped[~finite] = np.nan
    return ped, {"tiles": int(ids.size), "measured": int(good.size)}
