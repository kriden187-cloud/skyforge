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
    area = np.bincount(np.where(finite, tiles, -1)[finite].ravel(), minlength=int(tiles.max()) + 1)
    frac = counts / np.maximum(area[ids], 1)
    good = ids[(counts >= min_px) & (frac >= c.tile_min_free_fraction)]
    if good.size == 0 and ids.size:
        good = ids[[int(np.argmax(counts))]]  # ни одна не прошла порог — опора на самую свободную, остальные по границам
    level = float(np.median(a[free]))
    offsets = np.full(int(tiles.max()) + 1, np.nan, dtype=np.float32)
    if good.size:
        offsets[good] = np.asarray(ndimage.median(a, labels=lab, index=good), dtype=np.float32) - level
    # Область без своего неба (целиком под маской — ореол туманности или сама яркая плитка):
    # поправку даёт ступенька на границе с уже измеренными соседями — она от маски не зависит.
    n_matched = 0
    if np.isfinite(offsets).any() and not np.isfinite(offsets[np.unique(tiles[finite])]).all():
        # Яркое — относительно медианы 9×9 вокруг: лучи, звёзды, волокна. Ровная приподнятая плитка
        # (ступенька пьедестала) так не отмечается — медианный фильтр сохраняет ступеньку.
        filled = np.where(np.isfinite(a), a, np.float32(np.nanmedian(a[free]))).astype(np.float32)
        hp = filled - ndimage.median_filter(filled, size=9)
        _, _, std = sigma_clipped_stats(hp[free][:: max(1, int(free.sum()) // 2_000_000)], sigma=CLIP_SIGMA)
        bright = ndimage.binary_dilation(hp > c.tile_pixel_sigma * std, iterations=2) | ~np.isfinite(a)
        steps = boundary_steps(a, np.where(finite, tiles, -1), bright, c.tile_band_px)
        matched = propagate(offsets, steps, area)
        n_matched = int(np.isfinite(matched).sum() - np.isfinite(offsets).sum())
        offsets = matched
    ped = np.where(tiles >= 0, offsets[np.clip(tiles, 0, None)], np.nan).astype(np.float32)
    unknown = finite & ~np.isfinite(ped)
    if unknown.any() and np.isfinite(ped).any():
        # Без общей границы с измеренными — поправка ближайшей измеренной области.
        idx = ndimage.distance_transform_edt(~np.isfinite(ped), return_distances=False, return_indices=True)
        ped = ped[tuple(idx)]
    ped[~finite] = np.nan
    return ped, {"tiles": int(ids.size), "measured": int(good.size), "matched": n_matched}


def boundary_steps(a: np.ndarray, regions: np.ndarray, excluded: np.ndarray, band_px: int,
                   min_pairs: int = 100) -> dict[tuple[int, int], tuple[float, int]]:
    """Ступеньки на общих границах областей: {(i, j): (o_i − o_j, число пар)}.

    Для пар пикселей по разные стороны границы (до band_px от неё) медиана разности даёт скачок.
    Плавный склон туманности добавляет вклад, пропорциональный расстоянию между пикселями пары,
    поэтому медианы считаются отдельно для каждого расстояния и продолжаются прямой к нулю.
    """
    n = int(regions.max()) + 1
    if n < 2:
        return {}
    ok = np.isfinite(a) & ~excluded & (regions >= 0)
    keys, vals, seps = [], [], []
    h, w = regions.shape
    for axis in (0, 1):
        ia, ib = (regions[:-1, :], regions[1:, :]) if axis == 0 else (regions[:, :-1], regions[:, 1:])
        ys, xs = np.nonzero((ia != ib) & (ia >= 0) & (ib >= 0))
        la, lb = ia[ys, xs], ib[ys, xs]
        for k in range(band_px):
            if axis == 0:
                ya, yb, xa, xb = ys - k, ys + 1 + k, xs, xs
            else:
                ya, yb, xa, xb = ys, ys, xs - k, xs + 1 + k
            inb = (ya >= 0) & (yb < h) & (xa >= 0) & (xb < w)
            ya, yb, xa, xb, ka, kb = ya[inb], yb[inb], xa[inb], xb[inb], la[inb], lb[inb]
            good = (regions[ya, xa] == ka) & (regions[yb, xb] == kb) & ok[ya, xa] & ok[yb, xb]
            keys.append(ka[good].astype(np.int64) * n + kb[good])
            vals.append(a[ya[good], xa[good]] - a[yb[good], xb[good]])
            seps.append(np.full(int(good.sum()), 2 * k + 1, dtype=np.int16))
    keys_all = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int64)
    if keys_all.size == 0:
        return {}
    vals_all, seps_all = np.concatenate(vals), np.concatenate(seps)
    order = np.argsort(keys_all, kind="stable")
    keys_all, vals_all, seps_all = keys_all[order], vals_all[order], seps_all[order]
    uniq, start = np.unique(keys_all, return_index=True)
    out: dict[tuple[int, int], tuple[float, int]] = {}
    for u, s0, s1 in zip(uniq, start, list(start[1:]) + [len(keys_all)]):
        if s1 - s0 < min_pairs:
            continue
        i, j = divmod(int(u), n)
        step = _step_at_zero(vals_all[s0:s1], seps_all[s0:s1])
        # пары (i, j) и (j, i) — одна граница, собранная с двух сторон; объединяем с весами
        if (j, i) in out:
            prev, m = out.pop((j, i))
            step = (-prev * m + step * (s1 - s0)) / (m + s1 - s0)
            out[(i, j)] = (step, m + s1 - s0)
        else:
            out[(i, j)] = (step, int(s1 - s0))
    return out


def propagate(known: np.ndarray, steps: dict[tuple[int, int], tuple[float, int]], area: np.ndarray) -> np.ndarray:
    """Уровни неизмеренных областей по ступенькам к уже известным соседям — от больших областей к малым.

    Область получает средневзвешенное (по числу пар) из «уровень соседа + ступенька» по всем уже
    известным соседям и сама становится опорой для следующих. Проходы повторяются, пока что-то
    находится: цепочка узких полос дизеринга выравнивается от широкой измеренной плитки внутрь.
    """
    out = np.array(known, dtype=np.float32, copy=True)
    nbrs: dict[int, list[tuple[int, float, int]]] = {}
    for (i, j), (step, m) in steps.items():
        nbrs.setdefault(i, []).append((j, step, m))      # o_i = o_j + step
        nbrs.setdefault(j, []).append((i, -step, m))     # o_j = o_i − step
    todo = sorted((r for r in nbrs if not np.isfinite(out[r])), key=lambda r: -int(area[r]) if r < area.size else 0)
    changed = True
    while todo and changed:
        changed = False
        rest = []
        for r in todo:
            est = [(float(out[j]) + st, m) for j, st, m in nbrs[r] if np.isfinite(out[j])]
            if not est:
                rest.append(r)
                continue
            wsum = sum(m for _, m in est)
            out[r] = sum(v * m for v, m in est) / wsum
            changed = True
        todo = rest
    return out


def match_regions(a: np.ndarray, regions: np.ndarray, excluded: np.ndarray, band_px: int,
                  known: np.ndarray | None = None, min_pairs: int = 100) -> np.ndarray:
    """Уровни всех областей по ступенькам на границах. Без known — опора на самую большую область (0)."""
    n = int(regions.max()) + 1
    area = np.bincount(regions[regions >= 0].ravel(), minlength=n)
    out = np.full(n, np.nan, dtype=np.float32) if known is None else np.array(known, dtype=np.float32, copy=True)
    if not np.isfinite(out).any():
        out[int(np.argmax(area))] = 0.0
    steps = boundary_steps(a, regions, excluded, band_px, min_pairs)
    res = propagate(out, steps, area)
    if known is None and np.isfinite(res).any():
        res = res - np.nanmean(res)  # без внешней опоры: средняя поправка — ноль
    return res


def _step_at_zero(vals: np.ndarray, seps: np.ndarray) -> float:
    """Скачок на самой границе: медианы разностей по расстояниям, прямая, значение при расстоянии 0."""
    xs, ys = [], []
    for sep in np.unique(seps):
        v = vals[seps == sep]
        if v.size >= 10:
            xs.append(float(sep))
            ys.append(float(np.median(v)))
    if len(xs) < 3:
        return float(np.median(vals))
    _, intercept = np.polyfit(xs, ys, 1)
    return float(intercept)
