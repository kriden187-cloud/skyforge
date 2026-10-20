"""Следы космических лучей и тонкие зазоры между чипами.

Луч. У Hubble часть фильтров снята одной экспозицией, и сложение следы не убрало. След — резкий
выброс над мелкой структурой (как в L.A.Cosmic: остаток после медианы 3×3, который заметно больше
перепада между медианами 3×3 и 7×7). Звёзды в WFPC2 тоже резкие, поэтому решает сверка фильтров:
настоящий источник есть во всех фильтрах на том же месте неба, а луч — только в одном.

Зазор. Между чипами WFPC2 в одиночной экспозиции остаётся пустая полоса в несколько пикселей.
Это NaN-пиксели, которые исчезают при морфологическом замыкании маски данных окном
ширины зазора; заполняются средним известных соседей снаружи внутрь.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

from skyforge.config import CosmicsConfig
from skyforge.stretch import sample


BG_WINDOW = 9   # окно медианы местного фона, px сетки
LONE_PASSES = 3  # проходов отбора одиноких пятен (лучи), до финальной сверки
NOISE_BLOCK = 32  # блок локальной оценки шума превышения, px сетки
COSMICS_VERSION = 8  # поднять при смене алгоритма — входит в ключ кеша clean (только если очистка лучей включена)


def _excess(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(кадр без NaN, превышение над медианой BG_WINDOW×BG_WINDOW, карта σ шума превышения).

    σ — местная: MAD превышения в блоках NOISE_BLOCK (без NaN), сглаженная. Одна σ на кадр раздувалась фактурой
    туманности: на небе порог выходил высоким (лучи пропускались), а в туманности её мелкие детали, разные
    в разных линиях, проходили как лучи (удалено 11% пикселей туманности против 0,2–3% неба)."""
    fill = float(np.nanmedian(sample(a)))
    fin = np.isfinite(a)
    f = np.where(fin, a, fill).astype(np.float32)
    exc = f - ndimage.median_filter(f, BG_WINDOW)
    return f, exc, _sigma_map(exc, fin)


def _sigma_map(x: np.ndarray, fin: np.ndarray) -> np.ndarray:
    """Местная σ карты x: MAD в блоках NOISE_BLOCK (без NaN), сглаженная, продолженная в пустые блоки."""
    H, W = x.shape
    b = NOISE_BLOCK
    nb_y, nb_x = (H + b - 1) // b, (W + b - 1) // b
    pad = np.full((nb_y * b, nb_x * b), np.nan, dtype=np.float32)
    pad[:H, :W] = np.where(fin, x, np.nan)
    blocks = pad.reshape(nb_y, b, nb_x, b).transpose(0, 2, 1, 3).reshape(nb_y, nb_x, b * b)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(blocks, axis=2)
        mad = 1.4826 * np.nanmedian(np.abs(blocks - med[..., None]), axis=2)
    bad = ~np.isfinite(mad) | (mad <= 0)
    if bad.all():
        mad = np.ones_like(mad)
    elif bad.any():
        mad = mad[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    mad = ndimage.median_filter(mad, 3, mode="nearest")
    yy = (np.arange(H) - (b - 1) / 2) / b
    xx = (np.arange(W) - (b - 1) / 2) / b
    return ndimage.map_coordinates(mad, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest").astype(np.float32)


def find_cosmics(layers: dict[str, np.ndarray], c: CosmicsConfig, scale_arcsec: float,
                 check_trusted: bool = False, smoothed_sigma: bool = False) -> dict[str, np.ndarray]:
    """Маска следов для каждого фильтра (3б, одна экспозиция на фильтр).

    Пятно — связная область превышения > grow_sigma·σ с затравкой > sigma·σ. Это след, если: (1) не во всех
    других фильтрах на его месте (с допуском protect_arcsec) есть источник — сглаженное 3×3 превышение
    > others_sigma·σ (звезда — континуум, есть во всех; «хотя бы в одном» давало взаимную защиту лучей разных
    фильтров, поймано 56–72% на синтетике против 98%); пятно защищено и тогда, когда источник есть в фильтре,
    свободном от лучей (skip_filters — сложен из нескольких экспозиций): звезда, слабая в одной линии, иначе
    принималась за луч (3б: в F656N удалялись пики звёзд, которых нет в F502N); (2) площадь ≤ max_area_arcsec2; (3) пик / среднее 5×5 вокруг пика ≥ sharpness (узел
    туманности, яркий только в одной линии, плавный). Звезда есть во всех фильтрах — её защищает (1).
    Первый вариант (черновик: остаток после медианы 3×3) пропускал длинные и толстые следы: внутри следа шириной
    2 px медиана 3×3 сама яркая.

    check_trusted — чистить и фильтры skip_filters (3б шаг 3, только для прохода по небу): в двух экспозициях без
    сдвига остаются лучи и тёплые пиксели, совпавшие в обеих. Пятно в таком фильтре — луч, если его нет ни в одном
    из остальных фильтров (звезда, слабая в одной линии, есть хотя бы в другой).

    smoothed_sigma — σ для «источник есть» мерится по самой карте средних 3×3 (3б шаг 4, проход по небу): шум
    drizzle-кадров коррелирован, σ среднего 3×3 у WFPC2 — 0,45σ пикселя, а не σ/3; порог 5σ был на деле 3,7σ, и шум
    фильтра без лучей случайно защищал ~2,4% точек неба (на небе итога оставалось 65 следов)."""
    parts = {f: _excess(a) for f, a in layers.items()}
    protect = max(0, int(round(c.protect_arcsec / scale_arcsec)))
    max_area = max(4, int(round(c.max_area_arcsec2 / scale_arcsec ** 2)))
    grow = (lambda m: ndimage.maximum_filter(m.astype(np.uint8), size=2 * protect + 1) > 0) if protect else (lambda m: m)
    raw_seen = {}
    for f, (_, exc, sig) in parts.items():
        m3 = ndimage.uniform_filter(exc, 3)
        if smoothed_sigma:
            raw_seen[f] = m3 > c.others_sigma * _sigma_map(m3, np.isfinite(layers[f]))
        else:   # σ среднего 3×3 — σ пикселя / 3 (так принято на шаге 2; при коррелированном шуме занижена)
            raw_seen[f] = m3 > c.others_sigma * sig / 3

    def components(f: str, seen_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, exc, noise = parts[f]
        lab, n = ndimage.label(exc > c.grow_sigma * noise, structure=np.ones((3, 3)))   # noise — карта σ
        ids = np.arange(1, n + 1)
        if not n:
            return lab, ids, np.zeros(0, dtype=bool)
        peak = np.asarray(ndimage.maximum(exc, lab, ids))
        snr = np.asarray(ndimage.maximum(exc / noise, lab, ids))
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        mean5 = ndimage.uniform_filter(exc, 5)
        pos = ndimage.maximum_position(exc, lab, ids)
        sharp = peak / np.maximum(np.array([mean5[p] for p in pos]), 1e-12)
        # защита — в точке пика пятна: у звезды пик совпадает с источником в других фильтрах, а длинный след,
        # задевший где-то фактуру туманности, по всей площади оказался бы защищён (3б: следы на кропах sky, edge)
        in_other = np.array([bool(seen_other[p]) for p in pos], dtype=bool)
        return lab, ids, (snr > c.sigma) & ~in_other & (area <= max_area) & (sharp >= c.sharpness)

    # проход 1: одинокие пятна (нет ничего в других фильтрах) — лучи; их убираем из «источник есть» своего фильтра,
    # иначе лучи одного фильтра (их тысячи) закрывали бы лучи другого: «источник есть» покрывал 11–12% кадра
    trusted = [g for g in layers if g in c.skip_filters]   # свободны от лучей (сложены из нескольких экспозиций)

    def protected(f: str, seen: dict) -> np.ndarray:
        """Источник есть в свободном от лучей фильтре — или во всех остальных фильтрах."""
        others = [seen[g] for g in layers if g != f]
        if not others:
            return np.zeros_like(raw_seen[f])
        m = np.logical_and.reduce(others)
        for g in trusted:
            if g != f:
                m = m | seen[g]
        return m

    def protected_any(f: str, seen: dict) -> np.ndarray:
        """Источник есть хотя бы в одном из остальных фильтров — для фильтра без лучей при check_trusted."""
        others = [seen[g] for g in layers if g != f]
        return np.logical_or.reduce(others) if others else np.zeros_like(raw_seen[f])

    def lone_spots(f: str, seen: dict) -> np.ndarray:
        if f in trusted:
            if not check_trusted:     # в фильтре без лучей одиноких пятен-лучей нет: его источники настоящие
                return np.zeros(raw_seen[f].shape, dtype=bool)
            lab, ids, cr = components(f, protected_any(f, seen))
        else:
            lab, ids, cr = components(f, protected(f, seen))
        return np.isin(lab, ids[cr])

    seen = {g: grow(raw_seen[g]) for g in layers}
    for _ in range(LONE_PASSES):  # каждый проход уточняет «источник есть»: лучи, найденные в одном фильтре,
        lone = {f: lone_spots(f, seen) for f in layers}   # перестают защищать совпавшие с ними лучи других фильтров
        seen = {g: grow(raw_seen[g] & ~ndimage.binary_dilation(lone[g], iterations=2)) for g in layers}
    masks = {}
    for f in layers:
        m = lone_spots(f, seen)
        if c.grow_px:
            m = ndimage.binary_dilation(m, iterations=c.grow_px)
        masks[f] = m & np.isfinite(layers[f])
    return masks


QUIET_WINDOW = 25  # окно медианы местного фона для «тихого неба», px сетки (шире звезды и точечного артефакта)


def quiet_sky(a: np.ndarray, sky: np.ndarray, k: float) -> np.ndarray:
    """3б шаг 3: «тихое небо» — местный фон (медиана QUIET_WINDOW) не выше уровня неба + k·σ шума пикселя.

    Уровень — нижняя четверть местного фона по маске неба: маска неба — всё вне маски туманности, и в неё входит
    слабая внешняя оболочка (F658N: медиана по маске неба 0,00118 против 0,00087 вдали от туманности). σ — шум
    пикселя над местным фоном (MAD), его оболочка не раздувает. Точечный артефакт медиана не видит."""
    fin = np.isfinite(a)
    sel = sky & fin
    fill = float(np.median(a[sel])) if sel.any() else 0.0
    f = np.where(fin, a, fill).astype(np.float32)
    bg = ndimage.median_filter(f, QUIET_WINDOW)
    if not sel.any():
        return np.zeros(a.shape, dtype=bool)
    level = float(np.percentile(bg[sel], 25))
    r = (f - bg)[sel]
    sig = float(1.4826 * np.median(np.abs(r - np.median(r))))
    return fin & (bg - level < k * sig)


def remove_cosmics(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """След → местный фон: медиана BG_WINDOW×BG_WINDOW, во втором проходе — без самого следа."""
    if not mask.any():
        return a
    f = np.where(np.isfinite(a), a, np.nanmedian(sample(a))).astype(np.float32)
    tmp = f.copy()
    tmp[mask] = ndimage.median_filter(f, BG_WINDOW)[mask]
    out = np.array(a, dtype=np.float32, copy=True)
    out[mask] = ndimage.median_filter(tmp, BG_WINDOW)[mask]
    return out


def fill_gaps_linear(a: np.ndarray, max_half_px: int) -> tuple[np.ndarray, np.ndarray]:
    """3б: швы чипов и отбракованные чёрточки — линейная интерполяция поперёк шва от обеих сторон.

    Для пустого пикселя по четырём направлениям (строка, столбец, две диагонали) ищутся ближайшие данные
    вперёд и назад, не дальше 2·max_half_px + 1 шагов; берётся направление с самым коротким пролётом
    (поперёк шва), значение — линейно между краями. Там, где данных нет с одной из сторон (край поля,
    угол вне поля), пролёт шире предела или рядом пустота глубже max_half_px, пиксель остаётся пустым.
    Возвращает (кадр, маска заполненных)."""
    a = np.asarray(a, dtype=np.float32)
    nan = ~np.isfinite(a)
    if not nan.any() or max_half_px <= 0:
        return a.copy(), np.zeros(a.shape, dtype=bool)
    L = 2 * max_half_px + 1
    H, W = a.shape
    best_span = np.full(a.shape, np.inf, dtype=np.float32)
    best_val = np.full(a.shape, np.nan, dtype=np.float32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ends = []
        for sgn in (1, -1):
            dist = np.full(a.shape, np.inf, dtype=np.float32)
            val = np.full(a.shape, np.nan, dtype=np.float32)
            for k in range(1, L + 1):
                sy, sx = sgn * dy * k, sgn * dx * k
                shifted = np.full(a.shape, np.nan, dtype=np.float32)
                ys = slice(max(0, -sy), H - max(0, sy))
                yt = slice(max(0, sy), H - max(0, -sy))
                xs = slice(max(0, -sx), W - max(0, sx))
                xt = slice(max(0, sx), W - max(0, -sx))
                shifted[ys, xs] = a[yt, xt]
                hit = np.isfinite(shifted) & ~np.isfinite(val)
                dist[hit] = k
                val[hit] = shifted[hit]
            ends.append((dist, val))
        (df, vf), (db, vb) = ends
        span = df + db
        ok = nan & np.isfinite(span) & (span <= L + 1) & (span < best_span)
        best_span[ok] = span[ok]
        best_val[ok] = ((vb * df + vf * db) / span)[ok]
    # пустота вокруг пикселя (окно ±max_half_px) нигде не глубже max_half_px: иначе это угол широкой полосы или
    # край поля, а не шов — диагональ через угол нашла бы данные с обеих сторон, но полоса шире предела
    depth = ndimage.distance_transform_edt(nan)
    thin = ndimage.maximum_filter(depth, size=2 * max_half_px + 1) <= max_half_px + 0.5
    filled = nan & thin & np.isfinite(best_val)
    out = a.copy()
    out[filled] = best_val[filled]
    return out, filled
