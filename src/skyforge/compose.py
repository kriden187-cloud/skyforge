"""Раскраска: каждый фильтр получает оттенок по длине волны, слои смешиваются в RGB.

Короткие волны — синий конец hue_range, длинные — красный. Оттенки раздаются
по порядку фильтров, а не по абсолютной длине волны, как в пресс-релизах:
так соседние фильтры не сливаются в один цвет.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skyforge.config import ColorConfig, Config
from skyforge.filters import sort_by_wavelength, wavelength_um
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, save_png, shrink
from skyforge.stretch import sample, stretch_key, stretched_dir

# Яркость по Rec. 709 — для насыщенности и нейтрального фона.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def assign_colors(filters: list[str], mission: str, instrument: str, c: ColorConfig) -> dict[str, tuple[float, float, float]]:
    ordered = sort_by_wavelength(filters, mission, instrument)
    n = len(ordered)
    colors = {}
    for i, f in enumerate(ordered):
        if f in c.overrides:
            colors[f] = hex_to_rgb(c.overrides[f])
        elif n == 1:
            colors[f] = (1.0, 1.0, 1.0)
        else:
            h0, h1 = c.hue_range
            hue = (h0 + (h1 - h0) * i / (n - 1)) % 360
            colors[f] = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    return colors


def blend(layers: dict[str, np.ndarray], colors: dict[str, tuple[float, float, float]], c: ColorConfig) -> np.ndarray:
    first = next(iter(layers.values()))
    h, w = first.shape
    if c.blend == "screen":
        # 1 − Π(1 − слой·цвет): одиночный слой сохраняет яркость, совпадения светлеют к белому.
        inv = np.ones((h, w, 3), dtype=np.float32)
        for f, L in layers.items():
            col = np.asarray(colors[f], dtype=np.float32) * np.float32(c.weights.get(f, 1.0))
            Lf = np.nan_to_num(np.asarray(L, dtype=np.float32), nan=0.0)
            inv *= 1 - np.clip(Lf[..., None] * col, 0, 1)
        rgb = 1 - inv
    else:
        # Взвешенное среднее по каналам: все слои по 1 дают белый. Знаменатель — по фильтрам, которые
        # есть в этом пикселе: на краю покрытия (align.crop: union) цвет собирается из имеющихся, край не темнеет.
        num = np.zeros((h, w, 3), dtype=np.float32)
        den = np.zeros((h, w, 3), dtype=np.float32)
        for f, L in layers.items():
            col = np.asarray(colors[f], dtype=np.float32) * np.float32(c.weights.get(f, 1.0))
            La = np.asarray(L, dtype=np.float32)
            num += np.nan_to_num(La, nan=0.0)[..., None] * col
            den += np.isfinite(La)[..., None] * col
        rgb = num / np.where(den > 0, den, 1)
    # Где нет ни одного фильтра — честный чёрный, а не случайный цвет.
    nodata = np.all([~np.isfinite(np.asarray(L)) for L in layers.values()], axis=0)
    rgb[nodata] = 0
    return rgb


def neutralize(rgb: np.ndarray, pct: float, field: Any = None, keep_level: bool = False) -> tuple[np.ndarray, list[float]]:
    """Фон каждого канала в ноль: по самым тёмным пикселям, чтобы небо было серым, а не зелёным.
    field — поле статистики (skyforge.field): пустые и неполные края кадра в статистику не идут."""
    luma = rgb @ LUMA
    if field is not None:
        luma = field.values(luma)
    dark = sample(luma) <= np.nanpercentile(sample(luma), pct)
    bg = []
    for k in range(3):
        ch = sample(field.values(rgb[..., k]) if field is not None else rgb[..., k])
        bg.append(float(np.nanmedian(ch[dark])))
    bgv = np.asarray(bg, dtype=np.float32)
    if keep_level:  # только цветовой сдвиг фона между каналами; уровень неба (пьедестал) остаётся
        return np.clip(rgb - (bgv - bgv.mean()), 0, 1), bg
    out = (rgb - bgv) / np.where(bgv < 1, 1 - bgv, 1)
    return np.clip(out, 0, 1), bg


def shadows(rgb: np.ndarray, k: float, by: str = "luma", level: float = 0.0) -> np.ndarray:
    """Подъём теней после смешивания: asinh(k·яркость)/asinh(k), оттенок не меняется.

    average делит каждый канал на число вкладов, и деталь, яркая лишь в паре фильтров
    из шести (внешняя оболочка), темнеет втрое. Общая растяжка подняла бы её вместе с
    фоном и стыками мозаики; здесь поднимаются средние тона уже собранной картинки.
    k — сила: больше — сильнее, 0 — выкл.
    by: max — множитель по самому яркому каналу: тот поднимается кривой asinh и остаётся ≤ 1, остальные
    каналы — тем же множителем; деление пикселя не нужно, яркие цветные области не упираются в 1.
    """
    if k <= 0:
        return rgb
    if level > 0:  # 3k: поднимается сигнал над небом; кривая нечётная — шум ниже неба опускается так же, как выше поднят
        d = rgb - np.float32(level)
        span = 1 - level
        m = d.max(axis=-1) if by == "max" else d @ LUMA
        y = np.arcsinh(k * m / span) / np.arcsinh(k) * span
        gain = np.where(np.abs(m) > 1e-9, y / np.where(np.abs(m) > 1e-9, m, 1), k / np.arcsinh(k))
        out = d * gain[..., None]
        if by != "max":
            out = out / np.maximum(out.max(axis=-1, keepdims=True) / span, 1)
        return np.clip(level + out, 0, 1).astype(np.float32)
    if by == "max":
        m = rgb.max(axis=-1)
        y = np.arcsinh(k * m) / np.arcsinh(k)
        gain = np.where(m > 0, y / np.maximum(m, np.finfo(np.float32).tiny), 0)
        return np.clip(rgb * gain[..., None], 0, 1).astype(np.float32)
    lum = rgb @ LUMA
    y = np.arcsinh(k * lum) / np.arcsinh(k)
    gain = np.where(lum > 0, y / np.maximum(lum, np.finfo(np.float32).tiny), 0)
    out = rgb * gain[..., None]
    # Если какой-то канал ушёл за 1, уменьшаем весь пиксель, чтобы не сдвинуть оттенок.
    return np.clip(out / np.maximum(out.max(axis=-1, keepdims=True), 1), 0, 1)


def local_background(lum: np.ndarray, window_px: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Медиана яркости в окне шире звезды: свет туманности и неба под звездой без самой звезды.
    Считается на уменьшенной копии (окно в десятки пикселей на полном кадре — минуты); сетка блоков
    идёт от origin — начала поля статистики."""
    from skyforge.field import block_median_map

    return block_median_map(lum, window_px, origin)


def _star_fraction(x: np.ndarray, window_px: int, origin: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    tiny = np.finfo(np.float32).tiny
    local = local_background(x, window_px, origin)
    return np.clip((x - local) / np.maximum(x, tiny), 0, 1), local


def desaturate_stars(rgb: np.ndarray, mask: Any, s: float, det_layer: np.ndarray | None = None,
                     nebula_level: float = 0.05) -> np.ndarray:
    """Цвет звёзд к серому той же яркости.

    Обесцвечивается только доля света, которую добавила звезда сверх местного фона.
    На пустом небе доля считается по итоговой яркости — сереют и звезда, и её ореол из
    длинных волн. На туманности по итоговой яркости волокна неотличимы от звёзд, поэтому
    там доля берётся из фильтра поиска звёзд (det_layer, обычно F090W): звезда в нём яркая,
    а внешняя оболочка почти не видна. Между режимами — плавный переход по яркости
    местного фона (nebula_level).
    mask — stars.StarMask (вес 0–1, окно местного фона, фильтр поиска).
    """
    if mask is None or s >= 1:
        return rgb
    lum = rgb @ LUMA
    frac, local = _star_fraction(lum, mask.window_px, mask.origin)
    if det_layer is not None:
        # Свет звезды в фильтре поиска относительно всего света пикселя: на волокне, которого
        # в этом фильтре почти нет, доля около нуля, а не «весь свет фильтра — звезда».
        d = np.nan_to_num(np.asarray(det_layer, dtype=np.float32))
        star_light = np.clip(d - local_background(d, mask.window_px, mask.origin), 0, None)
        frac_det = np.clip(star_light / np.maximum(lum, np.finfo(np.float32).tiny), 0, 1)
        on_nebula = np.clip(local / nebula_level, 0, 1)
        frac = (1 - on_nebula) * frac + on_nebula * frac_det
    w = (mask.weight * frac)[..., None]
    grey = lum[..., None] + s * (rgb - lum[..., None])
    return np.clip(rgb + w * (grey - rgb), 0, 1)


def saturate(rgb: np.ndarray, s: float, gamut: str = "clip", level: float = 0.0, ramp: float = 0.0) -> np.ndarray:
    """Насыщенность вокруг яркости пикселя. gamut limit — самый яркий канал пикселя остаётся прежним, остальные
    расходятся от него (пиксель масштабируется обратно к исходному максимуму): канал не выходит за 1 и не
    прижимается к общему потолку — в яркой цветной области рисунок сохраняется (обрезка его стирает).
    ramp > 0 (3б) — насыщенность только над уровнем неба: от 1 при яркости level до s при level + ramp (0–1);
    шум неба не раздвигается за 0 (синий канал Hubble: 1,24% нулей неба от насыщенности 1,2)."""
    if s == 1.0:
        return rgb
    luma = (rgb @ LUMA)[..., None]
    if ramp > 0:
        s = 1 + (s - 1) * np.clip((luma - level) / ramp, 0, 1)
    out = luma + s * (rgb - luma)
    if gamut == "limit":
        m0 = rgb.max(axis=-1, keepdims=True)
        m1 = out.max(axis=-1, keepdims=True)
        out = out * np.where(m1 > 0, m0 / np.maximum(m1, np.float32(1e-6)), 0)
        return np.clip(out, 0, 1).astype(np.float32)
    return np.clip(out, 0, 1)


def halo_weight(rgb: np.ndarray, nebula_sd: np.ndarray, sc: Any, star_w: np.ndarray | None = None) -> np.ndarray:
    """Вес подъёма за маской туманности от сигнала: сглаженная яркость (звёзды заменены уровнем неба) над
    уровнем неба, делённая на halo_k·σ сглаженного неба, в 0–1. Небо — самые тёмные 30% сглаженной
    яркости за маской; σ — разброс там же (ступеньки и плавный ход фона), не шум пикселя."""
    from scipy.ndimage import gaussian_filter

    base = rgb.max(axis=-1) if sc.by == "max" else rgb @ LUMA
    out = nebula_sd < -sc.feather_px
    stars = (star_w > 0.01) if star_w is not None else np.zeros(base.shape, bool)
    level0 = float(np.nanmedian(base[out & ~stars]))
    s = gaussian_filter(np.where(stars, level0, base).astype(np.float32), sc.halo_smooth_px)
    v = s[out & ~stars]
    dark = v[v < np.percentile(v, 30)]
    level = float(np.median(dark))
    sigma = float(1.4826 * np.median(np.abs(dark - level))) or 1e-6
    return np.clip((s - level) / (sc.halo_k * sigma), 0, 1).astype(np.float32)


def apply_shadows(rgb: np.ndarray, sc: Any, nebula_sd: np.ndarray | None, star_w: np.ndarray | None = None,
                  halo_w: np.ndarray | None = None, level: float = 0.0) -> np.ndarray:
    """Подъём теней по всему кадру или только в маске туманности с мягким краем feather_px.
    level — уровень неба (above_sky): поднимается только сигнал над ним."""
    if sc.value <= 0:
        return rgb
    lifted = shadows(rgb, sc.value, sc.by, level)
    if sc.scope == "all":
        return lifted
    if nebula_sd is None:
        raise SkyforgeError("color.shadows.scope: nebula — нет маски туманности: запусти skyforge clean")
    w = np.clip(0.5 + nebula_sd / sc.feather_px, 0, 1).astype(np.float32)[..., None]
    out = rgb + w * (lifted - rgb)
    if sc.outside > 0:  # за маской — своя сила (внешний ореол), с весом от сигнала при halo_k > 0
        if sc.halo_k > 0:
            hw = halo_w if halo_w is not None else halo_weight(rgb, nebula_sd, sc, star_w)
            w_out = (1 - w) * hw[..., None]
        else:
            w_out = 1 - w
        out = out + w_out * (shadows(rgb, sc.outside, sc.by, level) - rgb)
    return out


def sky_weight(nebula_sd: np.ndarray | None, feather_px: float, shape: tuple[int, ...]) -> np.ndarray:
    """1 вне маски туманности, 0 внутри, линейный мягкий край feather_px по обе стороны границы."""
    if nebula_sd is None:
        return np.ones(shape[:2], dtype=np.float32)
    return np.clip(0.5 - nebula_sd / feather_px, 0, 1).astype(np.float32)


def equalize_sky(rgb: np.ndarray, sky: np.ndarray,
                 weight: np.ndarray | None = None) -> tuple[np.ndarray, list[float], list[float]]:
    """Медианы R, G, B по пустому небу → их среднее. Поправка одна на канал, но применяется с весом
    weight: на небе целиком, в туманности — нисколько (туманность не фон, её цвет не трогаем)."""
    if sky.sum() == 0:
        return rgb, [], []
    med = [float(np.median(rgb[..., k][sky])) for k in range(3)]
    target = float(np.mean(med))
    delta = (target - np.asarray(med, dtype=np.float32))
    w = np.ones(rgb.shape[:2], dtype=np.float32) if weight is None else weight
    out = rgb + w[..., None] * delta
    return np.clip(out, 0, 1).astype(np.float32), med, [target] * 3


def sky_color_map(rgb: np.ndarray, sky: np.ndarray, block_px: int, min_sky: float,
                  field: Any = None) -> tuple[np.ndarray, int]:
    """Цветовая поправка неба по блокам: гладкие карты медиан R, G, B по пустому небу (продолжены под маску
    от соседних блоков) → поправка каждого канала к их среднему. Сумма поправок по каналам в каждом
    пикселе — ноль: яркость (сумма каналов) не меняется, меняется только цвет.
    Карта считается по всему небу кадра (и по краям union, где набор фильтров неполный — у них свой баланс
    цвета); field — только начало сетки блоков (начало общего поля)."""
    from skyforge.background import block_map

    origin = field.origin if field is not None else None
    maps, measured = [], 0
    for k in range(3):
        m, measured = block_map(rgb[..., k], sky & np.isfinite(rgb[..., k]), block_px, min_sky, origin)
        maps.append(m)
    med = np.stack(maps, axis=-1)
    return (med.mean(axis=-1, keepdims=True) - med).astype(np.float32), measured


def base_color(cfg: Config, layers: dict[str, np.ndarray], sky: np.ndarray | None = None,
               nebula_sd: np.ndarray | None = None, mode: str | None = None,
               field: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Цвета фильтров, смешивание, нейтральный фон — до подъёма теней. mode — вместо color.neutral_background.
    field — поле статистики (skyforge.field); None — весь кадр, пиксели со всеми фильтрами."""
    from skyforge.field import make_field

    colors = assign_colors(list(layers), cfg.mission, cfg.instrument, cfg.color)
    rgb = blend(layers, colors, cfg.color)
    bg = None
    info: dict[str, Any] = {}
    mode = mode or cfg.color.neutral_background
    if field is None:
        field = make_field(list(layers.values()), None)
    sky_all = sky
    if sky is not None:
        # одна поправка на кадр (constant) — по небу поля статистики, где есть все фильтры
        sky_f = np.zeros(sky.shape, dtype=bool)
        sky_f[field.rows, field.cols] = field.mask(sky)
        sky = sky_f
    if mode != "off":
        rgb, bg = neutralize(rgb, cfg.color.background_percentile, field, cfg.color.background_keep_level)
        if sky is not None and mode == "constant":
            w = sky_weight(nebula_sd, cfg.color.sky_feather_px, rgb.shape)
            rgb, before, after = equalize_sky(rgb, sky, w)
            info.update(sky_medians_before=[round(v, 5) for v in before], sky_level=round(after[0], 5) if after else None)
        elif sky is not None and mode == "map" and sky.any():
            w = sky_weight(nebula_sd, cfg.color.sky_feather_px, rgb.shape)
            # карта цвета — по всему небу кадра: края union с неполным набором фильтров тоже приводятся к серому
            delta, measured = sky_color_map(rgb, sky_all, cfg.color.background_map_px,
                                            cfg.color.background_map_min_sky, field)
            # без обрезки в 0–1: обрезка поменяла бы сумму каналов; итог обрезается на следующих шагах
            rgb = rgb + w[..., None] * delta
            info.update(color_map_blocks=measured, color_map_max=round(float(np.abs(delta).max()), 5),
                        sky_medians_after=[round(float(np.median(rgb[..., k][sky])), 5) for k in range(3)])
    info.update({"colors": {f: "#%02x%02x%02x" % tuple(round(v * 255) for v in colors[f]) for f in colors},
                 "background_removed": bg})
    return rgb, info


def star_chroma(rgb: np.ndarray, stars: list, c: Any) -> np.ndarray:
    """Цвет вокруг звёзд с насыщенным ядром: яркость (самый яркий канал) та же, доли каналов — сглаженные
    по соседям с весом яркости; обрезанные белые пиксели в расчёт цвета не идут, ядро берёт цвет кольца."""
    from scipy.ndimage import gaussian_filter

    out = rgb.copy()
    h, w = rgb.shape[:2]
    R = int(np.ceil(c.r_out_px + 3 * c.core_sigma_px))
    for row, col, _, _ in stars:
        y0, y1, x0, x1 = max(row - R, 0), min(row + R + 1, h), max(col - R, 0), min(col + R + 1, w)
        sub = rgb[y0:y1, x0:x1]
        m = sub.max(axis=-1)
        frac = sub / np.maximum(m, 1e-6)[..., None]
        wgt = np.where(sub.min(axis=-1) < 0.99, m, 0).astype(np.float32)          # белое обрезанное — без цвета
        num = np.stack([gaussian_filter(frac[..., k] * wgt, c.sigma_px) for k in range(3)], -1)
        den = gaussian_filter(wgt, c.sigma_px)[..., None]
        num2 = np.stack([gaussian_filter(frac[..., k] * wgt, c.core_sigma_px) for k in range(3)], -1)
        den2 = gaussian_filter(wgt, c.core_sigma_px)[..., None]
        small = den / np.maximum(gaussian_filter(np.ones_like(wgt), c.sigma_px)[..., None], 1e-6)
        mix = np.clip(small * 4, 0, 1)                                            # мало цвета рядом — берём шире
        col_s = mix * num / np.maximum(den, 1e-6) + (1 - mix) * num2 / np.maximum(den2, 1e-6)
        col_s = col_s / np.maximum(col_s.max(axis=-1, keepdims=True), 1e-6)       # самый яркий канал = 1
        yy, xx = np.mgrid[y0:y1, x0:x1]
        r = np.hypot(yy - row, xx - col)
        t = np.where(r < c.r_in_px, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((r - c.r_in_px) / (c.r_out_px - c.r_in_px), 0, 1)))
        t = np.where(r >= c.r_out_px, 0.0, t)[..., None]
        new = m[..., None] * col_s
        out[y0:y1, x0:x1] = (sub + t * (new - sub)).astype(np.float32)
    return np.clip(out, 0, 1)


def scope_to_sky(mask: Any, nebula_sd: np.ndarray) -> Any:
    """Маска звёзд только со звёздами неба (3б, color.star_saturation_scope: sky). Решение — по каждой звезде: ядро
    веса (> 0,5), центр масс ядра вне маски туманности (nebula_sd < 0); пиксель ореола относится к ближайшему ядру —
    звезда у края маски не режется пополам, а слившиеся ореолы звезды на туманности и звезды неба решаются порознь
    (ревью шага 4: по всей слившейся области решал один пиксель плато, исход зависел от порядка строк)."""
    from scipy import ndimage

    from skyforge.stars import StarMask

    w = mask.weight
    cores, n = ndimage.label(w > 0.5, structure=np.ones((3, 3)))
    if n == 0:
        return StarMask(np.zeros_like(w), mask.window_px, mask.detect, mask.origin)
    com = ndimage.center_of_mass(np.ones_like(w), cores, np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = [nebula_sd[int(round(y)), int(round(x))] < 0 for y, x in com]
    iy, ix = ndimage.distance_transform_edt(cores == 0, return_distances=False, return_indices=True)
    nearest = cores[iy, ix]
    return StarMask(np.where(keep[nearest] & (w > 0.01), w, 0).astype(np.float32), mask.window_px, mask.detect,
                    mask.origin)


def tail_color(cfg: Config, rgb: np.ndarray, layers: dict[str, np.ndarray], star_mask: Any = None,
               nebula_sd: np.ndarray | None = None, sat_stars: list | None = None,
               star_w: np.ndarray | None = None) -> np.ndarray:
    """Тени, насыщенность, обесцвечивание звёзд. star_w — вес звёзд для подъёма за маской, если star_mask нет
    (обесцвечивание звёзд выключено)."""
    # Цветовая карта фона не обрезает значения (сумма каналов должна сохраниться точно); отрицательный канал
    # здесь — шум неба ниже нуля. Без обрезки shadows обнулил бы весь пиксель с отрицательной яркостью.
    rgb = np.clip(rgb, 0, 1)
    level = 0.0
    if cfg.color.shadows.above_sky:
        if cfg.stretch.sky_level is None:
            raise SkyforgeError("color.shadows.above_sky требует stretch.sky_level — уровень неба, от которого поднимать")
        level = cfg.stretch.sky_level / 255
    rgb = apply_shadows(rgb, cfg.color.shadows, nebula_sd, star_mask.weight if star_mask is not None else star_w,
                        level=level)
    ramp = cfg.color.saturation_above_sky / 255
    if ramp > 0 and cfg.stretch.sky_level is None:
        raise SkyforgeError("color.saturation_above_sky требует stretch.sky_level — уровень неба, от которого насыщать")
    rgb = saturate(rgb, cfg.color.saturation, cfg.color.saturation_gamut,
                   level=(cfg.stretch.sky_level or 0) / 255, ramp=ramp)
    det_layer = layers.get(star_mask.detect) if star_mask is not None else None
    if cfg.color.star_saturation_scope == "sky" and star_mask is not None and cfg.color.star_saturation < 1:
        if nebula_sd is None:
            raise SkyforgeError("color.star_saturation_scope: sky требует маску туманности: запусти skyforge clean")
        star_mask, det_layer = scope_to_sky(star_mask, nebula_sd), None   # на небе доля звезды — по итоговой яркости
    rgb = desaturate_stars(rgb, star_mask, cfg.color.star_saturation, det_layer, cfg.stars.nebula_level).astype(np.float32)
    if cfg.color.star_chroma.enabled and sat_stars:
        sc = cfg.color.star_chroma
        if sc.outside_nebula_only and nebula_sd is not None:
            sat_stars = [s for s in sat_stars if nebula_sd[s[0], s[1]] < 0]
        rgb = star_chroma(rgb, sat_stars, sc)
    return rgb


def ratio_map(dst: np.ndarray, src: np.ndarray, block: int) -> np.ndarray:
    """Гладкая карта dst/src: медианы обоих по блокам block×block, где есть оба (≥ 25% блока); пустые блоки —
    значение ближайшего, медиана 3×3 блока, билинейная интерполяция по центрам блоков."""
    from scipy import ndimage

    both = np.isfinite(dst) & np.isfinite(src)
    h, w = dst.shape
    nb_y, nb_x = -(-h // block), -(-w // block)
    small = np.full((nb_y, nb_x), np.nan, dtype=np.float32)
    for i in range(nb_y):
        for j in range(nb_x):
            sl = (slice(i * block, (i + 1) * block), slice(j * block, (j + 1) * block))
            m = both[sl]
            if m.mean() >= 0.25:
                ms = float(np.median(src[sl][m]))
                if ms > 0:
                    small[i, j] = float(np.median(dst[sl][m])) / ms
    bad = ~np.isfinite(small)
    if bad.all():
        raise SkyforgeError("color.fill_gaps: нет блоков, где есть оба фильтра пары")
    if bad.any():
        small = small[tuple(ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True))]
    small = ndimage.median_filter(small, size=3, mode="nearest")
    yy = (np.arange(h) + 0.5) / block - 0.5
    xx = (np.arange(w) + 0.5) / block - 0.5
    return ndimage.map_coordinates(small, np.meshgrid(yy, xx, indexing="ij"), order=1, mode="nearest").astype(np.float32)


def fill_gaps(cfg: Config, layers: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """color.fill_gaps: в щели одного фильтра пары — другой × гладкая карта отношения (фаза 6, W3-2).
    Возвращает слои и маски заполненного по фильтрам; пиксели вне щелей не меняются."""
    fg = cfg.color.fill_gaps
    if fg is None:
        return layers, {}
    out, filled = dict(layers), {}
    block = max(4, fg.block_px // cfg.bin)
    for group in fg.groups:
        for dst in group:
            if dst not in layers:
                continue
            D = np.asarray(layers[dst], dtype=np.float32)
            res, done = D.copy(), np.zeros(D.shape, dtype=bool)
            for src in (f for f in group if f != dst and f in layers):
                S = np.asarray(layers[src], dtype=np.float32)   # источник — исходный слой, не заполненный
                gap = ~np.isfinite(res) & np.isfinite(S)
                if gap.any():
                    Ss = S
                    if fg.smooth_px > 0:                             # без NaN-краёв: нормированная свёртка
                        from scipy import ndimage

                        sig = fg.smooth_px / cfg.bin
                        okS = np.isfinite(S)
                        num = ndimage.gaussian_filter(np.where(okS, S, 0), sig)
                        den = ndimage.gaussian_filter(okS.astype(np.float32), sig)
                        Ss = np.where(den > 1e-3, num / np.maximum(den, 1e-3), np.nan).astype(np.float32)
                    res[gap] = np.clip(Ss * ratio_map(D, S, block), 0, 1)[gap]
                    done |= gap
            if done.any():
                out[dst], filled[dst] = res, done
    return out, filled


def compose_layers(cfg: Config, layers: dict[str, np.ndarray], star_mask: Any = None,
                   nebula_sd: np.ndarray | None = None, sky: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    layers, _ = fill_gaps(cfg, layers)
    rgb, info = base_color(cfg, layers, sky, nebula_sd)
    return tail_color(cfg, rgb, layers, star_mask, nebula_sd), info


def composite_path(cfg: Config, paths: Paths) -> Path:
    return paths.style_work(cfg.slug, cfg.bin, cfg.style) / "composite.npy"


def compose_key(cfg: Config, in_key: dict[str, Any]) -> dict[str, Any]:
    skip = ({"star_saturation_scope"} if cfg.color.star_saturation_scope == "all" else set()) | (
        {"saturation_above_sky"} if cfg.color.saturation_above_sky == 0 else set())   # 3б: ключ JWST прежний
    skip |= {"fill_gaps"} if cfg.color.fill_gaps is None else set()          # фаза 6: ключ прежних объектов тот же
    color = cfg.color.model_dump(mode="json", exclude=skip)
    return {**stretch_key(cfg, in_key), "filters": cfg.filters, "color": color}


def load_stretched(cfg: Config, paths: Paths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from skyforge.clean import input_key  # clean импортирует compose для превью

    in_key = input_key(cfg, paths)
    d = stretched_dir(cfg, paths)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise SkyforgeError("нет растянутых кадров: сначала skyforge stretch")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["key"] != stretch_key(cfg, in_key):
        raise SkyforgeError("растяжка сделана с другими параметрами или кадрами: перезапусти skyforge stretch")
    want = [f for f in meta["filters"] if not cfg.filters or f in cfg.filters]
    return {f: np.load(d / f"{f}.npy") for f in want}, in_key


def compose(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    from skyforge.align import stats_field, with_resolved_crops
    from skyforge.clean import load_nebula_sd, load_sky_mask, load_star_weight

    layers, in_key = load_stretched(cfg, paths)
    layers, filled = fill_gaps(cfg, layers)
    if filled:
        composite_path(cfg, paths).parent.mkdir(parents=True, exist_ok=True)
        np.save(composite_path(cfg, paths).with_name("filled.npy"), np.logical_or.reduce(list(filled.values())))
        console.print("  заполнены щели: " + ", ".join(f"{f} {m.mean() * 100:.1f}% кадра" for f, m in filled.items()))
    view = with_resolved_crops(cfg, paths)  # кропы в пикселях текущей сетки (только для превью)
    field = stats_field(cfg, paths, list(layers.values()))
    stars = load_star_weight(cfg, paths)
    sd = load_nebula_sd(cfg, paths, required=cfg.color.shadows.scope == "nebula" and cfg.color.shadows.value > 0)
    sky = load_sky_mask(cfg, paths)
    base, info = base_color(cfg, layers, sky, sd if sd is not None else load_nebula_sd(cfg, paths), field=field)
    from skyforge.clean import load_saturated_stars

    sc = cfg.color.shadows
    star_w = None
    if stars is None and sc.outside > 0 and sc.halo_k > 0:  # вес ореола — без звёзд и при выкл. обесцвечивании
        from skyforge.clean import load_star_weight_map

        star_w = load_star_weight_map(cfg, paths)
    rgb = tail_color(cfg, base, layers, stars, sd, load_saturated_stars(cfg, paths), star_w)
    composite_path(cfg, paths).parent.mkdir(parents=True, exist_ok=True)
    np.save(composite_path(cfg, paths), rgb)
    composite_path(cfg, paths).with_suffix(".json").write_text(
        json.dumps({"key": compose_key(cfg, in_key), **info}, ensure_ascii=False, indent=2), encoding="utf-8")
    for f, col in info["colors"].items():
        console.print(f"  {f} ({wavelength_um(f, cfg.mission, cfg.instrument):.2f} мкм) → {col}")

    pv = cfg.preview.max_size
    save_png(rgb, out_dir / "compose.png", pv)
    colored = []
    for f, L in layers.items():
        col = np.asarray(hex_to_rgb(info["colors"][f]), dtype=np.float32)
        colored.append(np.nan_to_num(shrink(np.asarray(L), pv // 3))[..., None] * col)
    contact_sheet(colored, [f"{f} · {info['colors'][f]}" for f in layers], out_dir / "compose_layers.png",
                  title="Слои с назначенными цветами")
    _legend(info["colors"], cfg, out_dir / "compose_legend.png")
    info["previews"] = ["compose.png", "compose_layers.png", "compose_legend.png"]
    if cfg.color.shadows.value > 0 and sd is not None:
        _shadows_scope_sheet(cfg, base, layers, stars, sd, out_dir / "shadows_scope.png")
        info["previews"].append("shadows_scope.png")
    mcrop = next((c for c in view.report.crops if c.name == "sky3"), None)
    if mcrop is not None and sky is not None:
        if _nebula_mask_check(cfg, paths, layers, stars, mcrop, out_dir / "nebula_mask_check.png", field):
            info["previews"].append("nebula_mask_check.png")
    crops = [c for c in view.report.crops if c.name.startswith("sky")]
    if crops and sky is not None and cfg.color.neutral_background == "map":
        info["sky_check"] = _color_map_check(cfg, layers, stars, sd if sd is not None else load_nebula_sd(cfg, paths),
                                             sky, crops, out_dir / "background_check.png", field)
        info["previews"].append("background_check.png")
    return info


def _nebula_mask_check(cfg: Config, paths: Paths, layers: dict, stars: Any, crop: Any, path: Path,
                       field: Any = None) -> bool:
    """Кроп (sky3) 100% и ×5 с прежней маской туманности (квадрат, со звёздами) и с новой (3h)."""
    from skyforge.clean import clean_dir, load_sky_mask
    from skyforge.preview import display_crop

    d = clean_dir(cfg, paths)
    if not (d / "nebula_sd_prev.npy").exists():
        return False
    images, labels = [], []
    for name, fn in (("прежняя маска", "nebula_sd_prev.npy"), ("новая маска", "nebula_sd.npy")):
        sd = np.load(d / fn)
        sl = display_crop(sd.shape, crop.x, crop.y, crop.size, cfg.bin)
        base, _ = base_color(cfg, layers, load_sky_mask(cfg, paths, fn), sd, field=field)
        out = tail_color(cfg, base[sl], {f: np.asarray(a)[sl] for f, a in layers.items()},
                         stars.crop(sl) if stars is not None else None, sd[sl])
        images += [out, np.clip(out * 5, 0, 1)]
        labels += [f"{name}, 100%", f"{name}, ×5"]
    contact_sheet(images, labels, path, cols=2, tile_inches=4.5, interpolation="nearest",
                  title=f"Кроп {crop.name}, стиль {cfg.style or '—'}: с новой маской при ×5 плашек вокруг звёзд нет")
    return True


def _color_map_check(cfg: Config, layers: dict, stars: Any, sd: Any, sky: np.ndarray, crops: list,
                     path: Path, field: Any = None) -> dict[str, Any]:
    """Карта цветовой поправки (R, G, B) и кропы неба 100% и ×5: до (одна поправка на кадр, как в 3g)
    и после (карта). Медианы каналов по пустому небу каждого кропа."""
    from skyforge.preview import display_crop

    from skyforge.field import make_field

    field = field if field is not None else make_field(list(layers.values()), None)
    before, _ = base_color(cfg, layers, sky, sd, mode="constant", field=field)
    after, _ = base_color(cfg, layers, sky, sd, mode="map", field=field)
    rgb0, _ = neutralize(blend(layers, assign_colors(list(layers), cfg.mission, cfg.instrument, cfg.color), cfg.color),
                         cfg.color.background_percentile, field)
    delta, _ = sky_color_map(rgb0, sky, cfg.color.background_map_px, cfg.color.background_map_min_sky, field)
    lim = float(np.abs(delta).max()) or 1.0
    images = [np.clip(0.5 + delta / (2 * lim), 0, 1)]
    labels = [f"карта поправки R, G, B (серое — 0, ±{lim:.3g})"]
    meds: dict[str, Any] = {}
    for crop in crops:
        sl = display_crop(sky.shape, crop.x, crop.y, crop.size, cfg.bin)
        cl = {f: np.asarray(a)[sl] for f, a in layers.items()}
        cs = stars.crop(sl) if stars is not None else None
        csd = sd[sl] if sd is not None else None
        for name, base in (("до", before), ("после", after)):
            out = tail_color(cfg, base[sl], cl, cs, csd)
            m = sky[sl]
            med = [float(np.median(out[..., k][m])) if m.any() else float("nan") for k in range(3)]
            meds[f"{crop.name} {name}"] = [round(v, 4) for v in med]
            images += [out, np.clip(out * 5, 0, 1)]
            labels += [f"{crop.name} {name}, 100%", f"{crop.name} {name}, ×5: R {med[0]:.3f} G {med[1]:.3f} B {med[2]:.3f}"]
    contact_sheet(images, labels, path, cols=4, tile_inches=3.6, interpolation="nearest",
                  title=f"Цвет неба, стиль {cfg.style or '—'}: до — одна поправка на кадр, после — карта по блокам "
                        f"{cfg.color.background_map_px} px; при ×5 после — серое без пятен")
    return meds


def _background_check(cfg: Config, layers: dict, stars: Any, sd: Any, sky: np.ndarray, crop: Any,
                      path: Path) -> dict[str, Any]:
    """Кроп sky 100% до и после выравнивания каналов по пустому небу; медианы R, G, B по маске неба в кропе."""
    from skyforge.preview import display_crop

    sl = display_crop(sky.shape, crop.x, crop.y, crop.size, cfg.bin)
    off = cfg.model_copy(update={"color": cfg.color.model_copy(update={"neutral_background": "off"})})
    rgb_no, _ = base_color(off, layers)
    rgb_no, _ = neutralize(rgb_no, cfg.color.background_percentile)  # прежний шаг: только чёрная точка
    rgb_eq, _ = base_color(cfg, layers, sky, sd)
    crop_layers = {f: np.asarray(a)[sl] for f, a in layers.items()}
    cs = stars.crop(sl) if stars is not None else None
    csd = sd[sl] if sd is not None else None
    images, labels, meds = [], [], {}
    for name, base in (("до", rgb_no), ("после", rgb_eq)):
        out = tail_color(cfg, base[sl], crop_layers, cs, csd)
        m = sky[sl]
        med = [float(np.median(out[..., k][m])) if m.any() else float("nan") for k in range(3)]
        meds[name] = [round(v, 4) for v in med]
        images.append(out)
        labels.append(f"{name}: медианы R {med[0]:.3f} · G {med[1]:.3f} · B {med[2]:.3f}")
    contact_sheet(images, labels, path, cols=2, tile_inches=5, interpolation="nearest",
                  title=f"Пустое небо (кроп sky), 100%, стиль {cfg.style or '—'}: после — медианы каналов равны")
    return meds


def _shadows_scope_sheet(cfg: Config, base: np.ndarray, layers: dict, stars: Any, sd: np.ndarray, path: Path) -> None:
    """Кроп 100% на краю туманности (самая левая точка края): тени по всему кадру и только в маске."""
    edge = np.argwhere(np.abs(sd) <= 1)  # на краю знаковое расстояние ±1, ноля нет
    if edge.size == 0:
        return
    cy, cx = edge[np.argmin(edge[:, 1])]
    z = cfg.preview.zoom
    h, w = sd.shape
    y0, x0 = int(np.clip(cy - z // 2, 0, max(h - z, 0))), int(np.clip(cx - z // 2, 0, max(w - z, 0)))
    sl = (slice(y0, y0 + z), slice(x0, x0 + z))
    crop_layers = {f: np.asarray(a)[sl] for f, a in layers.items()}
    crop_stars = stars.crop(sl) if stars is not None else None
    images, labels = [], []
    for scope in ("all", "nebula"):
        c = cfg.model_copy(update={"color": cfg.color.model_copy(
            update={"shadows": cfg.color.shadows.model_copy(update={"scope": scope})})})
        images.append(tail_color(c, base[sl], crop_layers, crop_stars, sd[sl]))
        labels.append(f"shadows {cfg.color.shadows.value:g}, scope: {scope}")
    images.append(np.clip(0.5 + sd[sl] / cfg.color.shadows.feather_px, 0, 1))
    labels.append(f"маска туманности, край {cfg.color.shadows.feather_px:g} px")
    contact_sheet(images, labels, path, cols=3, tile_inches=4.5,
                  title="Край туманности, 100%: при scope nebula фон снаружи остаётся чёрным, ступеньки на краю нет")


def _legend(colors: dict[str, str], cfg: Config, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(colors) + 1, 1.4), facecolor="#111")
    for i, (f, col) in enumerate(colors.items()):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.6, color=col))
        ax.text(i + 0.45, 0.75, f, ha="center", color="#eee", fontsize=10)
        ax.text(i + 0.45, -0.25, f"{wavelength_um(f, cfg.mission, cfg.instrument):.2f} мкм", ha="center",
                color="#aaa", fontsize=8)
    ax.set_xlim(-0.1, len(colors))
    ax.set_ylim(-0.5, 1.0)
    ax.axis("off")
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
