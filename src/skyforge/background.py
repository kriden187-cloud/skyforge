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

from typing import Any

import numpy as np
from astropy.stats import sigma_clipped_stats
from photutils.background import Background2D, MedianBackground
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter

from skyforge.config import BackgroundConfig
from skyforge.log import SkyforgeError
from skyforge.stretch import sample

CLIP_SIGMA = 3.0  # отсечка выбросов при оценке шума фона


def source_mask(a: np.ndarray, c: BackgroundConfig, scale_arcsec: float, for_tiles: bool = False,
                field: Any = None) -> np.ndarray:
    """for_tiles — маска для поправок плиток. Порог по сглаженному кадру считается в единицах
    шума отдельного пикселя: иначе сама ступенька пьедестала (доли σ, но на тысячах пикселей)
    после сглаживания становится «источником» и прячет плитку под маску. Зато вокруг туманности
    маска шире (tile_mask_dilate_arcsec): полосы CON тянутся до края кадра, и оценивать их
    надёжнее по дальним частям, чем по ореолу."""
    # field (skyforge.field.StatsField) — медиана и шум считаются только в поле, где есть все фильтры
    finite = np.isfinite(a)
    smp = sample(field.values(a) if field is not None else a)
    med, _, std = sigma_clipped_stats(smp[np.isfinite(smp)], sigma=CLIP_SIGMA)  # std — шум пикселя
    s = np.where(finite, a - np.float32(med), 0).astype(np.float32)
    smooth_px = c.mask_smooth_arcsec / scale_arcsec
    if smooth_px > 0:
        s = gaussian_filter(s, smooth_px)
    if not for_tiles:
        smp_s = sample(field.values(np.where(finite, s, np.nan)) if field is not None else np.where(finite, s, np.nan))
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


def nexp_regions(nexp: np.ndarray, finite: np.ndarray, smooth_px: int = 1, min_area_px: int = 0) -> np.ndarray:
    """Связные области одинакового числа экспозиций: метки 0…n−1, вне данных и в клочках −1.
    smooth_px — медиана по NEXP до разметки: отбраковка отдельных пикселей в стеке дробит NEXP на крапины;
    клочки меньше min_area_px областью не считаются (поправку берут от ближайшей)."""
    if smooth_px > 1:
        nexp = ndimage.median_filter(nexp, size=smooth_px)
    out = np.full(nexp.shape, -1, dtype=np.int32)
    n = 0
    for v in np.unique(nexp[finite]):
        lab, k = ndimage.label(finite & (nexp == v))
        if not k:
            continue
        sizes = np.bincount(lab.ravel(), minlength=k + 1)
        keep = np.flatnonzero(sizes[1:] >= max(1, min_area_px)) + 1
        remap = np.full(k + 1, -1, dtype=np.int32)
        remap[keep] = np.arange(n, n + keep.size, dtype=np.int32)
        m = lab > 0
        out[m] = remap[lab[m]]
        n += keep.size
    return out


def nexp_levels(a: np.ndarray, regions: np.ndarray, c: BackgroundConfig,
                scale_arcsec: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Поправка уровня на каждую область NEXP (3k): скачки на общих границах (boundary_steps, полоса
    nexp_band_px) и распространение от самой большой области (поправка 0) к соседям (propagate).
    В скачок не идут яркие пиксели (лучи, звёзды, волокна — как в сшивке 3d) и, при scale_arcsec, маска
    источников с отступом (туманность: скачок по её структуре и по крапинам отбраковки в ней — не уровень);
    плавный ход фона и ореол снимает продолжение к границе. Область без измеренной границы по небу —
    поправка ближайшей измеренной."""
    finite = np.isfinite(a) & (regions >= 0)
    if not finite.any():
        return np.zeros_like(a), {"regions": 0, "matched": 0}
    data = np.isfinite(a)
    n = int(regions.max()) + 1
    area = np.bincount(regions[finite].ravel(), minlength=n)
    filled = np.where(np.isfinite(a), a, np.float32(np.nanmedian(a[finite]))).astype(np.float32)
    hp = filled - ndimage.median_filter(filled, size=9)
    _, _, std = sigma_clipped_stats(hp[finite][:: max(1, int(finite.sum()) // 2_000_000)], sigma=CLIP_SIGMA)
    bright = ndimage.binary_dilation(hp > c.tile_pixel_sigma * std, iterations=2) | ~np.isfinite(a)
    if scale_arcsec is not None:
        bright |= source_mask(a, c, scale_arcsec, for_tiles=True)
    known = np.full(n, np.nan, dtype=np.float32)
    known[int(np.argmax(area))] = 0.0
    steps = boundary_steps(a, np.where(finite, regions, -1), bright, c.nexp_band_px)
    off = propagate(known, steps, area)
    ped = np.where(regions >= 0, off[np.clip(regions, 0, None)], np.nan).astype(np.float32)
    unknown = data & ~np.isfinite(ped)   # клочки и области без измеренной границы
    if unknown.any() and np.isfinite(ped).any():
        idx = ndimage.distance_transform_edt(~np.isfinite(ped), return_distances=False, return_indices=True)
        ped = ped[tuple(idx)]
    ped[~data] = np.nan
    meas = np.isfinite(off)
    return ped, {"regions": n, "matched": int(meas.sum()), "boundaries": len(steps),
                 "max_abs": round(float(np.nanmax(np.abs(off))) if meas.any() else 0.0, 6)}


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


# --- фаза 3e: плоскость вместо константы ---

PLANE_MAX_POINTS = 20_000  # точек на подгонку плоскости одной области (детерминированное прореживание)
PLANE_ITERS, PLANE_CLIP = 6, 3.0


def fit_plane(x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Робастная плоскость v ≈ a + b·x + c·y: МНК с отбрасыванием точек дальше PLANE_CLIP·σ (по MAD)."""
    sel = np.ones(v.size, dtype=bool)
    coef = np.array([float(np.median(v)), 0.0, 0.0])
    for _ in range(PLANE_ITERS):
        A = np.column_stack([np.ones(int(sel.sum())), x[sel], y[sel]])
        coef, *_ = np.linalg.lstsq(A, v[sel], rcond=None)
        r = v - (coef[0] + coef[1] * x + coef[2] * y)
        mid = float(np.median(r[sel]))  # МНК с выбросами смещён: отсекаем от медианы остатков, не от нуля
        mad = 1.4826 * float(np.median(np.abs(r[sel] - mid)))
        new = np.abs(r - mid) <= PLANE_CLIP * max(mad, 1e-12)
        if new.sum() < 3 or np.array_equal(new, sel):
            break
        sel = new
    return coef


def adjacency(regions: np.ndarray) -> dict[tuple[int, int], int]:
    """Длина общей границы (число соседних пар пикселей) для каждой пары областей."""
    out: dict[tuple[int, int], int] = {}
    n = int(regions.max()) + 1
    for ia, ib in ((regions[:-1, :], regions[1:, :]), (regions[:, :-1], regions[:, 1:])):
        m = (ia != ib) & (ia >= 0) & (ib >= 0)
        lo, hi = np.minimum(ia[m], ib[m]), np.maximum(ia[m], ib[m])
        keys, counts = np.unique(lo.astype(np.int64) * n + hi, return_counts=True)
        for k, cnt in zip(keys, counts):
            i, j = divmod(int(k), n)
            out[(i, j)] = out.get((i, j), 0) + int(cnt)
    return out


def _bright_mask(a: np.ndarray, free: np.ndarray, c: BackgroundConfig) -> np.ndarray:
    """Лучи, звёзды, волокна — относительно медианы 9×9; ровная приподнятая плитка не отмечается."""
    filled = np.where(np.isfinite(a), a, np.float32(np.nanmedian(a[free]))).astype(np.float32)
    hp = filled - ndimage.median_filter(filled, size=9)
    _, _, std = sigma_clipped_stats(hp[free][:: max(1, int(free.sum()) // 2_000_000)], sigma=CLIP_SIGMA)
    return ndimage.binary_dilation(hp > c.tile_pixel_sigma * std, iterations=2) | ~np.isfinite(a)


def tile_planes(a: np.ndarray, tiles: np.ndarray, c: BackgroundConfig,
                scale_arcsec: float) -> tuple[np.ndarray, dict[str, int]]:
    """Поправка фона по областям CON — плоскость a + b·x + c·y на каждую (NaN вне поля).

    1. Область с достаточным свободным небом, протяжённым по обеим осям, — своя робастная плоскость.
    2. Остальные — фронтом от решённых, от больших областей к малым: наклон — среднее соседей
       (с весом по длине общей границы), уровень — по ступеньке на границе с уже исправленными соседями.
    3. Где ступеньку не измерить (внутри туманности: яркие пиксели исключены) — продолжается
       плоскость соседей: средние a, b, c с весом по длине границы.
    4. Область без решённых соседей — плоскость ближайшей решённой.
    Поправка = плоскость − общий уровень неба: средний фон кадра не меняется.
    """
    finite = np.isfinite(a) & (tiles >= 0)
    free = finite & ~source_mask(a, c, scale_arcsec, for_tiles=True)
    if not free.any():
        return np.zeros_like(a), {"tiles": 0, "measured": 0, "matched": 0, "continued": 0}
    regions = np.where(finite, tiles, -1).astype(np.int32)
    n = int(regions.max()) + 1
    area = np.bincount(regions[finite].ravel(), minlength=n)
    lab = np.where(free, regions, -1)
    counts = np.bincount(lab[lab >= 0].ravel(), minlength=n)
    min_px = max(1, int(round(c.tile_min_area_arcsec2 / scale_arcsec**2)))
    min_ext = c.tile_min_extent_arcsec / scale_arcsec
    level = float(np.median(a[free]))
    planes = np.full((n, 3), np.nan)

    ys_all, xs_all = np.nonzero(lab >= 0)
    labs = lab[ys_all, xs_all]
    order = np.argsort(labs, kind="stable")
    ys_all, xs_all, labs = ys_all[order], xs_all[order], labs[order]
    starts = np.searchsorted(labs, np.arange(n + 1))
    for r in np.argsort(-counts, kind="stable"):
        cnt = counts[r]
        if cnt < min_px or cnt < c.tile_min_free_fraction * area[r]:
            continue
        ys, xs = ys_all[starts[r]:starts[r + 1]], xs_all[starts[r]:starts[r + 1]]
        if np.ptp(xs) < min_ext or np.ptp(ys) < min_ext:
            continue  # узкая: наклон не измерить — от соседей
        step = max(1, ys.size // PLANE_MAX_POINTS)
        ys, xs = ys[::step], xs[::step]
        planes[r] = fit_plane(xs.astype(np.float64), ys.astype(np.float64), a[ys, xs].astype(np.float64))
    if not np.isfinite(planes[:, 0]).any():
        r = int(np.argmax(counts))
        planes[r] = [float(np.median(a[lab == r])), 0.0, 0.0]
    measured = int(np.isfinite(planes[:, 0]).sum())

    nbrs: dict[int, list[tuple[int, int]]] = {}
    for (i, j), m in adjacency(regions).items():
        nbrs.setdefault(i, []).append((j, m))
        nbrs.setdefault(j, []).append((i, m))
    bright = _bright_mask(a, free, c)
    yy, xx = np.indices(a.shape)
    idx = np.clip(regions, 0, None)
    n_matched = n_continued = 0
    while True:
        solved = np.isfinite(planes[:, 0])
        frontier = [r for r in range(n) if area[r] and not solved[r] and any(solved[j] for j, _ in nbrs.get(r, []))]
        if not frontier:
            break
        frontier.sort(key=lambda r: -area[r])
        slopes = {}
        for r in frontier:
            ws = [(planes[j], m) for j, m in nbrs[r] if solved[j]]
            slopes[r] = sum(p * m for p, m in ws) / sum(m for _, m in ws)
        p0, pb, pc = np.full(n, np.nan), np.zeros(n), np.zeros(n)
        p0[solved], pb[solved], pc[solved] = planes[solved, 0], planes[solved, 1], planes[solved, 2]
        for r, s in slopes.items():
            p0[r], pb[r], pc[r] = 0.0, s[1], s[2]
        w = a - (p0[idx] + pb[idx] * xx + pc[idx] * yy)
        w[(regions < 0) | ~np.isfinite(p0[idx])] = np.nan
        steps = boundary_steps(w, regions, bright | ~np.isfinite(w), c.tile_band_px)
        for r in frontier:
            est = []
            for (i, j), (st, m) in steps.items():
                if i == r and solved[j]:
                    est.append((st, m))
                elif j == r and solved[i]:
                    est.append((-st, m))
            if est:
                a0 = sum(v * m for v, m in est) / sum(m for _, m in est)
                planes[r] = [a0, slopes[r][1], slopes[r][2]]
                n_matched += 1
            else:
                planes[r] = slopes[r]  # внутри туманности: продолжить плоскость соседей
                n_continued += 1

    corr = planes[idx, 0] + planes[idx, 1] * xx + planes[idx, 2] * yy - level
    corr = np.where(finite, corr, np.nan).astype(np.float32)
    unknown = finite & ~np.isfinite(corr)
    if unknown.any() and np.isfinite(corr).any():
        has = np.isfinite(corr)
        near = ndimage.distance_transform_edt(~has, return_distances=False, return_indices=True)
        src = np.clip(regions[tuple(near)], 0, None)
        alt = planes[src, 0] + planes[src, 1] * xx + planes[src, 2] * yy - level
        corr = np.where(unknown, alt, corr).astype(np.float32)
    corr[~finite] = np.nan
    return corr, {"tiles": int((area > 0).sum()), "measured": measured, "matched": n_matched, "continued": n_continued}


def pixel_noise(a: np.ndarray, ok: np.ndarray) -> float:
    """Шум пикселя по разностям пикселей через 3 px (1.4826·MAD/√2): плавный ход фона и градиенты не влияют.
    Разброс всего неба (σ с отсечкой) ими раздувается — и скачок «в σ» менялся от вычитания гладкой модели."""
    lag = 3  # после проекции и стека соседние пиксели коррелированы — разность через 3 px их уже не связывает
    pair = ok[:, lag:] & ok[:, :-lag]
    d = (a[:, lag:] - a[:, :-lag])[pair]
    if d.size < 100:
        return 0.0
    d = d[:: max(1, d.size // 2_000_000)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2))


def boundary_jumps(a: np.ndarray, regions: np.ndarray, sky: np.ndarray, band_px: int) -> tuple[float, int]:
    """Средний |скачок| фона поперёк границ областей CON на пустом небе, в единицах σ шума фона."""
    ok = sky & np.isfinite(a) & (regions >= 0)
    if ok.sum() < 1000:
        return 0.0, 0
    steps = boundary_steps(a, np.where(np.isfinite(a), regions, -1), ~ok, band_px)
    if not steps:
        return 0.0, 0
    std = pixel_noise(a, ok)
    tot = sum(abs(s) * m for s, m in steps.values())
    cnt = sum(m for _, m in steps.values())
    return (round(tot / cnt / std, 4) if std else 0.0), len(steps)



def block_model(a: np.ndarray, c: BackgroundConfig, scale_arcsec: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Гладкий фон по блокам blocks_px (3g, для своей мозаики): медиана свободного неба в блоке;
    блоки, где свободного неба меньше blocks_min_free (туманность, звёзды), не меряются, а продолжаются
    от соседних измеренных (последовательное заполнение средним соседей); затем медиана 3×3 блоков и
    гладкая (бикубическая) интерполяция до полного размера."""
    from skyforge.config import StarsConfig  # noqa: F401  (тип для подсказки)
    from skyforge.stars import star_mask

    finite = np.isfinite(a)
    mask = source_mask(a, c, scale_arcsec, for_tiles=True)
    if c.blocks_mask_grow_px:
        mask = ndimage.binary_dilation(mask, iterations=c.blocks_mask_grow_px)
    free = finite & ~mask
    model, _ = block_map(a, free, c.blocks_px, c.blocks_min_free)
    return model, mask, c.blocks_px


def block_map(a: np.ndarray, free: np.ndarray, b: int, min_free: float,
              origin: tuple[int, int] | None = None) -> tuple[np.ndarray, int]:
    """Гладкая карта по блокам b×b: медиана a по пикселям free в блоке, где их не меньше min_free; остальные
    блоки продолжаются от измеренных соседей (слоями, средним соседей), затем медиана 3×3 блоков и
    бикубическая интерполяция до полного размера. Возвращает карту и число измеренных блоков.
    origin — сетка блоков идёт от этой точки (кадр дополняется пустыми строками и столбцами до кратного b)."""
    if origin is not None:
        H0, W0 = a.shape
        pr, pc = (-origin[0]) % b, (-origin[1]) % b
        er, ec = (-(H0 + pr)) % b, (-(W0 + pc)) % b
        ap = np.pad(np.asarray(a, dtype=np.float32), ((pr, er), (pc, ec)), constant_values=np.nan)
        fp = np.pad(free, ((pr, er), (pc, ec)), constant_values=False)
        m, n = block_map(ap, fp, b, min_free)
        return m[pr:pr + H0, pc:pc + W0], n
    H, W = a.shape
    ny, nx = (H + b - 1) // b, (W + b - 1) // b
    grid = np.full((ny, nx), np.nan)
    for j in range(ny):
        for i in range(nx):
            sl = (slice(j * b, min((j + 1) * b, H)), slice(i * b, min((i + 1) * b, W)))
            m = free[sl]
            if m.mean() >= min_free:
                grid[j, i] = float(np.median(a[sl][m]))
    measured = int(np.isfinite(grid).sum())
    if measured == 0:
        return np.zeros_like(a, dtype=np.float32), 0
    # продолжить под маску от соседей, слоями
    while not np.isfinite(grid).all():
        known = np.isfinite(grid)
        s_ = ndimage.convolve(np.where(known, grid, 0.0), np.ones((3, 3)), mode="constant")
        n_ = ndimage.convolve(known.astype(float), np.ones((3, 3)), mode="constant")
        ring = ~known & (n_ > 0)
        if not ring.any():
            break
        grid[ring] = s_[ring] / n_[ring]
    grid = ndimage.median_filter(np.nan_to_num(grid, nan=float(np.nanmedian(grid))), size=3, mode="nearest")
    zy, zx = H / ny, W / nx
    model = ndimage.zoom(grid, (zy, zx), order=3, mode="nearest", grid_mode=True)[:H, :W]
    if model.shape != a.shape:
        model = np.pad(model, ((0, H - model.shape[0]), (0, W - model.shape[1])), mode="edge")
    return model.astype(np.float32), measured
