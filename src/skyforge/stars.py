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
import warnings

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

from dataclasses import dataclass
from typing import Any

from skyforge.config import PlateauConfig, StarsConfig
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


# Области отбракованных пикселей произвольной формы (после фазы 7, OMC3-N): у очень яркой протозвезды конвейер STScI
# отбраковывает не только круглое ядро, но и россыпь пикселей блоком с прямыми краями (F200W: 76 дыр, NaN 42% оболочки),
# и уцелевшие пиксели между дырами — тоже выбросы (разброс в 6 раз выше окрестности). Заполнение максимумом соседей
# (fill_rings) делает из такой области пёстрое пятно. Область — россыпь мелких дыр, связанных закрытием; крупные сплошные
# части дыр с ярким краем — ядра звёзд (и насыщенные узлы): в область не входят, вокруг них — кольцо защиты (крылья
# не сглаживаются), заполняются белым через fill_rings, как прежде. Сплошной комок россыпи без яркого края — часть
# области (у OMC3-N в блоках такие есть). Признаки области — все сразу: площадь россыпи; доля NaN не мала (полоса малого
# покрытия с редкими битыми пикселями — не область) и не сплошь;
# уцелевшие пиксели внутри шероховатее окрестности. Пороги — по сводке всех объектов 2026-10-22: у пяти принятых ни один
# кластер их не проходит (у ядер звёзд 1–5 дыр и NaN ≥ 0,82; россыпь NGC 3132 — шероховатость 0,7; швы Hubble — NaN ≥ 0,86).
AREA_LINK_ARCSEC = 0.3        # мелкие дыры ближе — одна область (радиус закрытия)
AREA_MAX_HOLE_ARCSEC2 = 100.0  # дыры больше — пропуски мозаики, не трогаем
AREA_CORE_CONTRAST = 1.5      # ядро: (край − даль) / (середина − даль) — край 1–2 px, середина 4–8 px, даль 15–25 px
                              # от сплошной части дыры; у ядер звёзд Sh2-305, W3, NGC 3132 — 1,7–22, у комков ~1
AREA_CORE_POINT_CONTRAST = 3.0  # подтверждение пиком в другом фильтре: крутизна вокруг самого пика (у звёзд 5–22)
AREA_CORE_MIN_SIGMA = 20.0    # …и край ярче дали больше чем на столько σ шума дали (у ядер — десятки и сотни σ)
AREA_CORE_GUARD_ARCSEC = 0.25  # кольцо защиты вокруг ядра — не в области
AREA_MIN_ARCSEC2 = 1.0        # оболочка области меньше — не область
AREA_MIN_SPECK_ARCSEC2 = 0.1  # площадь россыпи в оболочке меньше — не область (вместо числа дыр: плотная россыпь
                              # срастается в несколько крупных компонент)
AREA_MIN_NAN_FRACTION = 0.15  # реже — битые пиксели на полосе малого покрытия, не область (у OMC3-N 0,23–0,65)
AREA_MAX_NAN_FRACTION = 0.9   # сплошь — уже не россыпь (ядра отделяет core_parts; у OMC3-N до 0,65)
AREA_MIN_ROUGHNESS = 3.0      # размах разностей соседних уцелевших пикселей внутри / в кольце 6–12 px снаружи
                              # (16–84-й процентиль; блоки OMC3-N F200W 3,7–15, синтетика 17,6, NGC 3132 F090W 0,7;
                              # полоса малого покрытия с шумом ×2,8 и битыми пикселями — до 2,9)
AREA_OUTLIER_SIGMA = 4.0      # выбросы у края области: отклонение от местной медианы, в σ кольца
AREA_OUTLIER_BAND_ARCSEC = 0.15  # узкая полоса у края: дальше — уже окрестность (звёзды рядом не трогаем)
AREA_OUTLIER_MAX_PX = 4       # отдельная (не касается области) группа выбросов крупнее — источник, не выброс


def area_key() -> dict[str, float]:
    """Пороги областей для ключа кеша clean."""
    return {"link": AREA_LINK_ARCSEC, "max_hole": AREA_MAX_HOLE_ARCSEC2, "core_contrast": AREA_CORE_CONTRAST,
            "core_sigma": AREA_CORE_MIN_SIGMA, "guard": AREA_CORE_GUARD_ARCSEC, "min_area": AREA_MIN_ARCSEC2, "min_speck": AREA_MIN_SPECK_ARCSEC2,
            "min_nan": AREA_MIN_NAN_FRACTION, "max_nan": AREA_MAX_NAN_FRACTION, "rough": AREA_MIN_ROUGHNESS,
            "outlier_sigma": AREA_OUTLIER_SIGMA, "band": AREA_OUTLIER_BAND_ARCSEC, "outlier_px": AREA_OUTLIER_MAX_PX,
            "point_contrast": AREA_CORE_POINT_CONTRAST,
            "others": "hole or point source in another filter", "version": 6}


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def _pair_sigma(v: np.ndarray, m: np.ndarray, robust: bool = True) -> float:
    """σ по разностям соседних по строке пикселей, оба в маске m и конечны, делённая на √2. robust — MAD (шум без
    выбросов); иначе половина размаха 16–84-го процентиля — выбросы в неё входят (мера загрязнённости: у россыпи
    NGC 3132 F090W она 0,7 окрестности, а MAD там 2,9 — MAD выбросы и прячет)."""
    ok = m & np.isfinite(v)
    pair = ok[:, 1:] & ok[:, :-1]
    if pair.sum() < 8:
        return float("nan")
    d = (v[:, 1:] - v[:, :-1])[pair]
    if robust:
        return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))
    p16, p84 = np.percentile(d, [16, 84])
    return float((p84 - p16) / 2 / math.sqrt(2))


def solid_holes(a: np.ndarray, scale: float) -> np.ndarray:
    """Сплошные части дыр фильтра (открытие 3×3) — для подтверждения ядер в других фильтрах."""
    return ndimage.binary_opening(find_core_holes(a, int(AREA_MAX_HOLE_ARCSEC2 / scale ** 2)), structure=_EIGHT)


def _peak_test(sub: np.ndarray, d: np.ndarray, inner_max: float) -> bool:
    """Пик у сплошной части: внутреннее кольцо (до inner_max px от части) ярче дали (15–25 px) больше чем на
    AREA_CORE_MIN_SIGMA σ шума дали и круче середины (4–8 px) — (внутр. − даль) / (середина − даль) ≥ AREA_CORE_CONTRAST."""
    def med(lo: float, hi: float) -> float:
        v = sub[(d >= lo) & (d <= hi) & np.isfinite(sub)]
        return float(np.median(v)) if v.size > 5 else float("nan")

    inner, mid, far = med(0, inner_max), med(4, 8), med(15, 25)
    noise = _pair_sigma(sub, (d > 15) & (d <= 25))
    return bool(inner - far > AREA_CORE_MIN_SIGMA * noise and mid != far
                and (inner - far) / (mid - far) >= AREA_CORE_CONTRAST)


def _point_test(g: np.ndarray, near: np.ndarray) -> bool:
    """Точечный источник другого фильтра у части: самый яркий конечный пиксель в near — пик над далью (15–25 px от него)
    больше AREA_CORE_MIN_SIGMA σ и крутой: (пик − даль) / (кольцо 4–8 px − даль) ≥ AREA_CORE_POINT_CONTRAST. Кольца —
    вокруг пикселя, а не вокруг части: крупная часть на вытянутой нити туманности иначе проходила как «пик»
    (пятый раунд проверки, OMC3-N F480M)."""
    v = np.where(near & np.isfinite(g), g, -np.inf)
    if not np.isfinite(v).any():
        return False
    y, x = np.unravel_index(int(np.argmax(v)), v.shape)
    yy, xx = np.indices(g.shape)
    d = np.hypot(yy - y, xx - x)

    def med(lo: float, hi: float) -> float:
        w = g[(d >= lo) & (d <= hi) & np.isfinite(g)]
        return float(np.median(w)) if w.size > 5 else float("nan")

    peak, mid, far = float(g[y, x]), med(4, 8), med(15, 25)
    noise = _pair_sigma(g, (d >= 15) & (d <= 25))
    return bool(peak - far > AREA_CORE_MIN_SIGMA * noise and mid != far
                and (peak - far) / (mid - far) >= AREA_CORE_POINT_CONTRAST)


def core_parts(a: np.ndarray, holes: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]] | None = None,
               link_px: int = 1) -> np.ndarray:
    """Ядра среди дыр: сплошные части (открытие 3×3, с краем в пиксель — по пикселям, а не по компоненте: ядро,
    касающееся россыпи, отделяется от неё), у которых край ярче окрестности (_peak_test) и — если даны другие фильтры
    (others: [(кадр, сплошные дыры)]) — источник на том же месте хотя бы в одном из них: сплошная дыра рядом (link_px)
    или точечный источник (_point_test): насыщенная в одном фильтре звезда в другом — пик.
    OMC3-N F200W: комки блока на ярком краю туманности у тёмного неба проходили по яркости края, а в F150W, F356W и
    F480M там ровная туманность — не ядро; узел выброса (F480M — пик 217 над фоном 6) — ядро."""
    seeds = ndimage.binary_opening(holes, structure=_EIGHT)
    cores = np.zeros(a.shape, dtype=bool)
    lab, _ = ndimage.label(seeds, structure=_EIGHT)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        pad = 26
        r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        r1, c1 = min(sl[0].stop + pad, a.shape[0]), min(sl[1].stop + pad, a.shape[1])
        part = ndimage.binary_dilation(lab[r0:r1, c0:c1] == i + 1, structure=_EIGHT) & holes[r0:r1, c0:c1]
        d = ndimage.distance_transform_edt(~part)
        # край ярче дали и заметно (не шум ровного фона), и круче середины — пик, а не комок на ровном месте
        if not _peak_test(a[r0:r1, c0:c1], np.where(part, -1.0, d), 2):
            continue
        if others:
            near = ndimage.binary_dilation(part, structure=_disk(link_px))
            if not any((s[r0:r1, c0:c1] & near).any() or _point_test(g[r0:r1, c0:c1], near) for g, s in others):
                continue
        cores[r0:r1, c0:c1] |= part
    return cores


def flagged_areas(a: np.ndarray, scale: float, others: list[tuple[np.ndarray, np.ndarray]] | None = None
                  ) -> tuple[np.ndarray, list[dict]]:
    """Маска областей отбракованных пикселей произвольной формы (с выбросами у края) и сводка по каждой. Ядра звёзд
    (core_parts; others — другие фильтры: кадр и сплошные дыры) и кольцо защиты вокруг них в маску не входят никогда."""
    holes = find_core_holes(a, int(AREA_MAX_HOLE_ARCSEC2 / scale ** 2))
    region = np.zeros(a.shape, dtype=bool)
    if not holes.any():
        return region, []
    r = max(1, int(round(AREA_LINK_ARCSEC / scale)))
    cores = core_parts(a, holes, others, r)
    specks = holes & ~cores
    if not specks.any():
        return region, []
    g = max(1, int(round(AREA_CORE_GUARD_ARCSEC / scale)))
    guard = ndimage.binary_dilation(cores, structure=_disk(g)) if cores.any() else cores
    p = r + 1
    closed = ndimage.binary_closing(np.pad(specks, p), structure=_disk(r))[p:-p, p:-p] | specks
    hull = ndimage.binary_fill_holes(closed) & ~guard
    lab, _ = ndimage.label(hull, structure=_EIGHT)
    # +1: оболочка закрытия у прямого края блока сидит на пиксель внутри (крайний ряд входит, только если сам дыра)
    band = int(math.ceil(AREA_OUTLIER_BAND_ARCSEC / scale)) + 1
    pad = 12 + band
    info = []
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        r0, c0 = max(sl[0].start - pad, 0), max(sl[1].start - pad, 0)
        r1, c1 = min(sl[0].stop + pad, a.shape[0]), min(sl[1].stop + pad, a.shape[1])
        m = lab[r0:r1, c0:c1] == i + 1
        area = int(m.sum())
        if area * scale * scale < AREA_MIN_ARCSEC2:
            continue
        h = specks[r0:r1, c0:c1] & m
        n_holes = ndimage.label(h, structure=_EIGHT)[1]
        frac = float(h.sum()) / area
        if h.sum() * scale * scale < AREA_MIN_SPECK_ARCSEC2 or not AREA_MIN_NAN_FRACTION <= frac <= AREA_MAX_NAN_FRACTION:
            continue
        sub = a[r0:r1, c0:c1]
        d = ndimage.distance_transform_edt(~m)
        ring = (d > 6) & (d <= 12)
        s_out = _pair_sigma(sub, ring)
        w_out = _pair_sigma(sub, ring, robust=False)
        rough = _pair_sigma(sub, m, robust=False) / w_out if w_out > 0 else float("nan")
        if not rough >= AREA_MIN_ROUGHNESS:
            continue
        # выбросы у края: отклонение от медианы 7×7 по конечным пикселям вне области больше k σ кольца, только в узкой
        # полосе (AREA_OUTLIER_BAND_ARCSEC — пик звезды рядом в неё не попадает); группы, касающиеся области, — целиком
        # в пределах полосы, отдельные — не крупнее AREA_OUTLIER_MAX_PX
        known = np.where(np.isfinite(sub) & ~m, sub, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # окно целиком из NaN — медиана NaN, пиксель не выброс
            med = ndimage.generic_filter(known, np.nanmedian, size=7, mode="nearest")
        near = (d > 0) & (d <= band) & np.isfinite(sub)
        out_px = near & (np.abs(sub - med) > AREA_OUTLIER_SIGMA * s_out)
        lab_o, n_o = ndimage.label(out_px, structure=_EIGHT)
        ok_o = np.zeros(n_o + 1, dtype=bool)
        if n_o:
            sizes = ndimage.sum_labels(out_px, lab_o, index=np.arange(1, n_o + 1))
            touch = ndimage.maximum(ndimage.binary_dilation(m, structure=_EIGHT), lab_o, index=np.arange(1, n_o + 1))
            ok_o[1:] = (np.asarray(touch) > 0) | (sizes <= AREA_OUTLIER_MAX_PX)
        grown = (m | ok_o[lab_o]) & ~guard[r0:r1, c0:c1]
        region[r0:r1, c0:c1] |= grown
        info.append({"y": int((sl[0].start + sl[0].stop) // 2), "x": int((sl[1].start + sl[1].stop) // 2),
                     "area_px": area, "holes": int(n_holes), "nan_fraction": round(frac, 3),
                     "roughness": round(float(rough), 2), "outliers_px": int((grown & ~m).sum())})
    return region, info


def fill_smooth(a: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Заполнить область гладко по её краю — уравнение Лапласа (каждый пиксель — среднее четырёх соседей), граница —
    известные пиксели вокруг: плавный переход с градиентом окрестности, без своих краёв. Решается по 4-связным кускам
    (соседи в уравнении — четыре); пустые соседи вне области в уравнение не входят. Кусок без единого известного
    соседа не трогается (его пиксели остаются как были: NaN доделает fill_rings, известные — сохраняются)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import spsolve

    out = np.array(a, dtype=np.float32, copy=True)
    four = ndimage.generate_binary_structure(2, 1)
    lab, _ = ndimage.label(region, structure=four)
    for i, sl in enumerate(ndimage.find_objects(lab)):
        if sl is None:
            continue
        win = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
        sub = out[win]
        m = lab[win] == i + 1
        ys, xs = np.nonzero(m)
        n = len(ys)
        idx = -np.ones(m.shape, dtype=np.int64)
        idx[ys, xs] = np.arange(n)
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        rhs = np.zeros(n)
        n_known = 0
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = ys + dy, xs + dx
            inside = (ny >= 0) & (ny < m.shape[0]) & (nx >= 0) & (nx < m.shape[1])
            k = np.nonzero(inside)[0]
            j = idx[ny[k], nx[k]]
            unk = j >= 0
            rows.append(k[unk]); cols.append(j[unk]); vals.append(np.full(int(unk.sum()), -1.0))
            diag[k[unk]] += 1
            kv = k[~unk]
            v = sub[ny[kv], nx[kv]].astype(np.float64)
            fin = np.isfinite(v)
            rhs[kv[fin]] += v[fin]
            diag[kv[fin]] += 1
            n_known += int(fin.sum())
        if n_known == 0:
            continue                        # ни одного известного соседа: решения нет, кусок не трогаем
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        A = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        sub[ys, xs] = spsolve(A, rhs).astype(np.float32)
    return out


@dataclass
class StarMask:
    """Растушёванная маска звёзд (0–1) и окно местного фона в пикселях той же сетки."""

    weight: np.ndarray
    window_px: int
    detect: str  # фильтр, в котором искались звёзды: по нему считается доля звёздного света на туманности
    origin: tuple[int, int] = (0, 0)  # начало поля статистики: от него идёт сетка блоков местного фона

    def crop(self, sl: tuple[slice, slice]) -> StarMask:
        return StarMask(self.weight[sl], self.window_px, self.detect,
                        (self.origin[0] - (sl[0].start or 0), self.origin[1] - (sl[1].start or 0)))

    def shrink(self, factor: int) -> StarMask:
        from skyforge.preview import nan_block_mean

        f = max(factor, 1)
        return StarMask(nan_block_mean(self.weight, factor), max(3, self.window_px // f), self.detect,
                        (self.origin[0] // f, self.origin[1] // f))


def star_mask(a: np.ndarray, s: StarsConfig, scale_arcsec: float, field: object = None) -> tuple[np.ndarray, np.ndarray]:
    """Маска звёзд поля и маска плотной туманности для одного фильтра.

    Звезда — компактная связная область выше mask_sigma с пиком выше peak_sigma и площадью
    не больше max_star_area_arcsec2. Всё, что больше, — плотная часть туманности: её пиксели
    из маски исключаются жёстко. Слабые волокна защищает уже не маска, а доля звёздного
    света в пикселе (compose.desaturate_stars).
    """
    smp = sample(field.values(a) if field is not None else a)  # field — поле статистики (skyforge.field)
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
    if r > 0 and stars.any():
        # Круглый ореол через карту расстояний: то же, что расширение кругом радиуса r, но за линейное
        # время (круглое окно в maximum_filter не раскладывается по осям — 100 с на мозаику 4900²).
        stars = ndimage.distance_transform_edt(~stars) <= r
    return stars, is_nebula[labels]


SKY_STAR_WINDOW = 25  # окно медианы местного фона для звёзд неба, px сетки


def sky_stars(a: np.ndarray, outside: np.ndarray, quiet: np.ndarray, s: StarsConfig, scale_arcsec: float) -> np.ndarray:
    """3б: звёзды поля на небе — по местному фону и шуму неба, а не по σ всего кадра.

    В тесной рамке вокруг туманности (Hubble) σ кадра раздута туманностью в 8–17 раз — star_mask находил одну
    звезду неба из восьми. Здесь фон — медиана SKY_STAR_WINDOW, σ — MAD превышения на тихом небе; звезда — связная
    область выше mask_sigma·σ с пиком ≥ peak_sigma·σ, не больше max_star_area_arcsec2, пик вне маски туманности
    (outside) на тихом небе (quiet: пики на слабой оболочке за маской — её фактура). С ореолом halo_arcsec."""
    fin = np.isfinite(a)
    sel = outside & quiet & fin
    if not sel.any():
        return np.zeros(a.shape, dtype=bool)
    f = np.where(fin, a, float(np.median(a[sel]))).astype(np.float32)
    exc = f - ndimage.median_filter(f, SKY_STAR_WINDOW)
    r = exc[sel]
    sig = float(1.4826 * np.median(np.abs(r - np.median(r))))
    labels, n = ndimage.label((exc > s.mask_sigma * sig) & fin, structure=_EIGHT)
    if n == 0:
        return np.zeros(a.shape, dtype=bool)
    idx = np.arange(1, n + 1)
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndimage.maximum(exc, labels, idx))
    pos = ndimage.maximum_position(exc, labels, idx)
    on_sky = np.array([bool(outside[p] and quiet[p]) for p in pos], dtype=bool)
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = (area <= s.max_star_area_arcsec2 / scale_arcsec**2) & (peak >= s.peak_sigma * sig) & on_sky
    stars = keep[labels]
    rad = int(round(s.halo_arcsec / scale_arcsec))
    if rad > 0 and stars.any():
        stars = ndimage.distance_transform_edt(~stars) <= rad
    return stars & outside


def sky_weight(stars: np.ndarray, nebula_sd: np.ndarray, s: StarsConfig, scale_arcsec: float) -> np.ndarray:
    """Вес звёзд неба (0–1): растушёвка как у star_weight, на туманность не заходит."""
    w = stars.astype(np.float32)
    feather = s.halo_arcsec / scale_arcsec / 3
    if feather > 0:
        w = ndimage.gaussian_filter(w, feather)
    w[nebula_sd >= 0] = 0
    return w


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


PLATEAU_NOISE_K = 5.0   # допуск плато — не меньше стольких σ шума кадра
PLATEAU_VERSION = 5     # 2: допуск выше неба и местного шума (без этого «плато» — шум диска галактики);
                        # 3: следы растекания заряда выше доли уровня плато (pc.saturation); 4–5: кайма ниже неба


def plateau_cores(a: np.ndarray, pc: PlateauConfig) -> tuple[np.ndarray, dict[str, int]]:
    """Фаза 9 (stars.plateau_cores): плато насыщения наземного стека → дыра NaN (дальше — fill_cores, как ядро i2d).

    Кандидаты — локальные максимумы ярче pc.min_snr шумов (σ по разностям соседних пикселей) над медианой кадра, и
    допуск pc.tol от их высоты над медианой — не меньше PLATEAU_NOISE_K σ (иначе «плато» — просто шум гладкой туманности
    или диска галактики: на уровне 100σ допуск 1% меньше шума). Плато — связная область вокруг максимума в пределах
    допуска, не меньше pc.min_px пикселей; допуск — выше и местного шума в окне (яркая середина галактики шумит сильнее
    неба). У ненасыщенной звезды и ядра галактики уже соседний пиксель на 15–20% ниже
    пика — плато из одного пикселя. Если плато найдены и задан pc.saturation, дырой становятся и связные области (не меньше
    pc.min_px) выше этой доли уровня самого низкого плато над медианой — следы растекания заряда у ярких звёзд. Кайма
    ниже неба, связная с найденным, — тоже дыра. Кадр меняется на месте (копия из load_sci). Возвращает кадр и
    {plateaus, pixels, largest_px[, saturation_level, bleed_px, rim_px]}."""
    from scipy import ndimage

    ok = np.isfinite(a)
    if not ok.any():
        return a, {"plateaus": 0, "pixels": 0, "largest_px": 0}
    v = a[ok]
    med = float(np.median(v[:: max(1, v.size // 2_000_000)]))
    d = np.diff(np.where(ok, a, np.nan), axis=1)
    d = d[np.isfinite(d)]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size else 0.0
    z = np.where(ok, a, -np.inf)
    floor = med + max(pc.min_snr, PLATEAU_NOISE_K / pc.tol) * noise
    peaks = np.argwhere((z == ndimage.maximum_filter(z, 3)) & (z > floor))
    peaks = peaks[np.argsort(-a[peaks[:, 0], peaks[:, 1]])]
    H, W = a.shape
    out = np.zeros(a.shape, dtype=bool)
    sizes, levels = [], []
    half = 40
    for y, x in peaks:
        if out[y, x]:
            continue
        y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, H), max(x - half, 0), min(x + half + 1, W)
        win = z[y0:y1, x0:x1]
        band = pc.tol * (a[y, x] - med)
        # местный шум (яркая середина галактики шумит сильнее неба): допуск плато должен быть выше него
        dw = np.diff(np.where(np.isfinite(win), win, np.nan), axis=1)
        dw = dw[np.isfinite(dw)]
        if dw.size and band < PLATEAU_NOISE_K * 1.4826 * float(np.median(np.abs(dw - np.median(dw)))) / np.sqrt(2):
            continue
        lab, _ = ndimage.label(win >= a[y, x] - band)
        reg = lab == lab[y - y0, x - x0]
        n = int(reg.sum())
        if n >= pc.min_px:
            out[y0:y1, x0:x1] |= reg
            sizes.append(n)
            levels.append(float(a[y, x]))
    info: dict[str, Any] = {}
    if pc.saturation is not None and levels:
        level = med + pc.saturation * (min(levels) - med)
        lab, n = ndimage.label(z >= level)
        big = np.bincount(lab.ravel(), minlength=n + 1) >= pc.min_px
        big[0] = False
        bleed = big[lab] & ~out
        out |= big[lab]
        info = {"saturation_level": float(level), "bleed_px": int(bleed.sum())}
    if pc.grow_px and out.any():
        out = ndimage.binary_dilation(out, iterations=pc.grow_px)
    if out.any():
        # кайма ниже неба вокруг насыщения (у следов растекания — столбец через пиксель ниже порога, шире grow_px):
        # связные (8 соседей) с расширенной дырой пиксели ниже медианы на PLATEAU_NOISE_K σ — тоже дыра; иначе при bin N
        # они остаются в блоке и дают ноль в одном фильтре
        low = ok & ~out & (a < med - PLATEAU_NOISE_K * noise)
        lab, _ = ndimage.label(low | out, structure=np.ones((3, 3)))
        touch = np.zeros(lab.max() + 1, dtype=bool)
        touch[np.unique(lab[out])] = True
        rim = touch[lab] & low
        out |= rim
        info["rim_px"] = int(rim.sum())
    a[out] = np.nan
    return a, {"plateaus": len(sizes), "pixels": int(out.sum()), "largest_px": int(max(sizes) if sizes else 0), **info}
