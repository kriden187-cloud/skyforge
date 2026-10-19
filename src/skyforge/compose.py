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
        # Взвешенное среднее по каналам: все слои по 1 дают белый.
        num = np.zeros((h, w, 3), dtype=np.float32)
        den = np.zeros(3, dtype=np.float32)
        for f, L in layers.items():
            col = np.asarray(colors[f], dtype=np.float32) * np.float32(c.weights.get(f, 1.0))
            num += np.nan_to_num(np.asarray(L, dtype=np.float32), nan=0.0)[..., None] * col
            den += col
        rgb = num / np.where(den > 0, den, 1)
    # Где нет ни одного фильтра — честный чёрный, а не случайный цвет.
    nodata = np.all([~np.isfinite(np.asarray(L)) for L in layers.values()], axis=0)
    rgb[nodata] = 0
    return rgb


def neutralize(rgb: np.ndarray, pct: float) -> tuple[np.ndarray, list[float]]:
    """Фон каждого канала в ноль: по самым тёмным пикселям, чтобы небо было серым, а не зелёным."""
    luma = rgb @ LUMA
    dark = sample(luma) <= np.nanpercentile(sample(luma), pct)
    bg = []
    for k in range(3):
        ch = sample(rgb[..., k])
        bg.append(float(np.nanmedian(ch[dark])))
    bgv = np.asarray(bg, dtype=np.float32)
    out = (rgb - bgv) / np.where(bgv < 1, 1 - bgv, 1)
    return np.clip(out, 0, 1), bg


def shadows(rgb: np.ndarray, k: float) -> np.ndarray:
    """Подъём теней после смешивания: asinh(k·яркость)/asinh(k), оттенок не меняется.

    average делит каждый канал на число вкладов, и деталь, яркая лишь в паре фильтров
    из шести (внешняя оболочка), темнеет втрое. Общая растяжка подняла бы её вместе с
    фоном и стыками мозаики; здесь поднимаются средние тона уже собранной картинки.
    k — сила: больше — сильнее, 0 — выкл.
    """
    if k <= 0:
        return rgb
    lum = rgb @ LUMA
    y = np.arcsinh(k * lum) / np.arcsinh(k)
    gain = np.where(lum > 0, y / np.maximum(lum, np.finfo(np.float32).tiny), 0)
    out = rgb * gain[..., None]
    # Если какой-то канал ушёл за 1, уменьшаем весь пиксель, чтобы не сдвинуть оттенок.
    return np.clip(out / np.maximum(out.max(axis=-1, keepdims=True), 1), 0, 1)


def local_background(lum: np.ndarray, window_px: int) -> np.ndarray:
    """Медиана яркости в окне шире звезды: свет туманности и неба под звездой без самой звезды."""
    from scipy import ndimage

    from skyforge.preview import nan_block_mean

    # Медиана считается на уменьшенной копии: окно в десятки пикселей на полном кадре — минуты.
    factor = max(1, window_px // 16)
    small = nan_block_mean(lum, factor)
    size = max(3, window_px // factor) | 1
    med = ndimage.median_filter(small, size=size, mode="nearest")
    zoom = (lum.shape[0] / med.shape[0], lum.shape[1] / med.shape[1])
    out = ndimage.zoom(med, zoom, order=1, mode="nearest")
    return out[: lum.shape[0], : lum.shape[1]]


def _star_fraction(x: np.ndarray, window_px: int) -> tuple[np.ndarray, np.ndarray]:
    tiny = np.finfo(np.float32).tiny
    local = local_background(x, window_px)
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
    frac, local = _star_fraction(lum, mask.window_px)
    if det_layer is not None:
        # Свет звезды в фильтре поиска относительно всего света пикселя: на волокне, которого
        # в этом фильтре почти нет, доля около нуля, а не «весь свет фильтра — звезда».
        d = np.nan_to_num(np.asarray(det_layer, dtype=np.float32))
        star_light = np.clip(d - local_background(d, mask.window_px), 0, None)
        frac_det = np.clip(star_light / np.maximum(lum, np.finfo(np.float32).tiny), 0, 1)
        on_nebula = np.clip(local / nebula_level, 0, 1)
        frac = (1 - on_nebula) * frac + on_nebula * frac_det
    w = (mask.weight * frac)[..., None]
    grey = lum[..., None] + s * (rgb - lum[..., None])
    return np.clip(rgb + w * (grey - rgb), 0, 1)


def saturate(rgb: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return rgb
    luma = (rgb @ LUMA)[..., None]
    return np.clip(luma + s * (rgb - luma), 0, 1)


def apply_shadows(rgb: np.ndarray, sc: Any, nebula_sd: np.ndarray | None) -> np.ndarray:
    """Подъём теней по всему кадру или только в маске туманности с мягким краем feather_px."""
    if sc.value <= 0:
        return rgb
    lifted = shadows(rgb, sc.value)
    if sc.scope == "all":
        return lifted
    if nebula_sd is None:
        raise SkyforgeError("color.shadows.scope: nebula — нет маски туманности: запусти skyforge clean")
    w = np.clip(0.5 + nebula_sd / sc.feather_px, 0, 1).astype(np.float32)[..., None]
    return rgb + w * (lifted - rgb)


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


def sky_color_map(rgb: np.ndarray, sky: np.ndarray, block_px: int, min_sky: float) -> tuple[np.ndarray, int]:
    """Цветовая поправка неба по блокам: гладкие карты медиан R, G, B по пустому небу (продолжены под маску
    от соседних блоков) → поправка каждого канала к их среднему. Сумма поправок по каналам в каждом
    пикселе — ноль: яркость (сумма каналов) не меняется, меняется только цвет."""
    from skyforge.background import block_map

    maps, measured = [], 0
    for k in range(3):
        m, measured = block_map(rgb[..., k], sky & np.isfinite(rgb[..., k]), block_px, min_sky)
        maps.append(m)
    med = np.stack(maps, axis=-1)
    return (med.mean(axis=-1, keepdims=True) - med).astype(np.float32), measured


def base_color(cfg: Config, layers: dict[str, np.ndarray], sky: np.ndarray | None = None,
               nebula_sd: np.ndarray | None = None, mode: str | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Цвета фильтров, смешивание, нейтральный фон — до подъёма теней. mode — вместо color.neutral_background."""
    colors = assign_colors(list(layers), cfg.mission, cfg.instrument, cfg.color)
    rgb = blend(layers, colors, cfg.color)
    bg = None
    info: dict[str, Any] = {}
    mode = mode or cfg.color.neutral_background
    if mode != "off":
        rgb, bg = neutralize(rgb, cfg.color.background_percentile)
        if sky is not None and mode == "constant":
            w = sky_weight(nebula_sd, cfg.color.sky_feather_px, rgb.shape)
            rgb, before, after = equalize_sky(rgb, sky, w)
            info.update(sky_medians_before=[round(v, 5) for v in before], sky_level=round(after[0], 5) if after else None)
        elif sky is not None and mode == "map" and sky.any():
            w = sky_weight(nebula_sd, cfg.color.sky_feather_px, rgb.shape)
            delta, measured = sky_color_map(rgb, sky, cfg.color.background_map_px, cfg.color.background_map_min_sky)
            # без обрезки в 0–1: обрезка поменяла бы сумму каналов; итог обрезается на следующих шагах
            rgb = rgb + w[..., None] * delta
            info.update(color_map_blocks=measured, color_map_max=round(float(np.abs(delta).max()), 5),
                        sky_medians_after=[round(float(np.median(rgb[..., k][sky])), 5) for k in range(3)])
    info.update({"colors": {f: "#%02x%02x%02x" % tuple(round(v * 255) for v in colors[f]) for f in colors},
                 "background_removed": bg})
    return rgb, info


def tail_color(cfg: Config, rgb: np.ndarray, layers: dict[str, np.ndarray], star_mask: Any = None,
               nebula_sd: np.ndarray | None = None) -> np.ndarray:
    """Тени, насыщенность, обесцвечивание звёзд."""
    # Цветовая карта фона не обрезает значения (сумма каналов должна сохраниться точно); отрицательный канал
    # здесь — шум неба ниже нуля. Без обрезки shadows обнулил бы весь пиксель с отрицательной яркостью.
    rgb = np.clip(rgb, 0, 1)
    rgb = apply_shadows(rgb, cfg.color.shadows, nebula_sd)
    rgb = saturate(rgb, cfg.color.saturation)
    det_layer = layers.get(star_mask.detect) if star_mask is not None else None
    return desaturate_stars(rgb, star_mask, cfg.color.star_saturation, det_layer,
                            cfg.stars.nebula_level).astype(np.float32)


def compose_layers(cfg: Config, layers: dict[str, np.ndarray], star_mask: Any = None,
                   nebula_sd: np.ndarray | None = None, sky: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    rgb, info = base_color(cfg, layers, sky, nebula_sd)
    return tail_color(cfg, rgb, layers, star_mask, nebula_sd), info


def composite_path(cfg: Config, paths: Paths) -> Path:
    return paths.style_work(cfg.slug, cfg.bin, cfg.style) / "composite.npy"


def compose_key(cfg: Config, in_key: dict[str, Any]) -> dict[str, Any]:
    return {**stretch_key(cfg, in_key), "filters": cfg.filters, "color": cfg.color.model_dump(mode="json")}


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
    from skyforge.clean import load_nebula_sd, load_sky_mask, load_star_weight

    layers, in_key = load_stretched(cfg, paths)
    stars = load_star_weight(cfg, paths)
    sd = load_nebula_sd(cfg, paths, required=cfg.color.shadows.scope == "nebula" and cfg.color.shadows.value > 0)
    sky = load_sky_mask(cfg, paths)
    base, info = base_color(cfg, layers, sky, sd if sd is not None else load_nebula_sd(cfg, paths))
    rgb = tail_color(cfg, base, layers, stars, sd)
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
    mcrop = next((c for c in cfg.report.crops if c.name == "sky3"), None)
    if mcrop is not None and sky is not None:
        if _nebula_mask_check(cfg, paths, layers, stars, mcrop, out_dir / "nebula_mask_check.png"):
            info["previews"].append("nebula_mask_check.png")
    crops = [c for c in cfg.report.crops if c.name.startswith("sky")]
    if crops and sky is not None and cfg.color.neutral_background == "map":
        info["sky_check"] = _color_map_check(cfg, layers, stars, sd if sd is not None else load_nebula_sd(cfg, paths),
                                             sky, crops, out_dir / "background_check.png")
        info["previews"].append("background_check.png")
    return info


def _nebula_mask_check(cfg: Config, paths: Paths, layers: dict, stars: Any, crop: Any, path: Path) -> bool:
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
        base, _ = base_color(cfg, layers, load_sky_mask(cfg, paths, fn), sd)
        out = tail_color(cfg, base[sl], {f: np.asarray(a)[sl] for f, a in layers.items()},
                         stars.crop(sl) if stars is not None else None, sd[sl])
        images += [out, np.clip(out * 5, 0, 1)]
        labels += [f"{name}, 100%", f"{name}, ×5"]
    contact_sheet(images, labels, path, cols=2, tile_inches=4.5, interpolation="nearest",
                  title=f"Кроп {crop.name}, стиль {cfg.style or '—'}: с новой маской при ×5 плашек вокруг звёзд нет")
    return True


def _color_map_check(cfg: Config, layers: dict, stars: Any, sd: Any, sky: np.ndarray, crops: list,
                     path: Path) -> dict[str, Any]:
    """Карта цветовой поправки (R, G, B) и кропы неба 100% и ×5: до (одна поправка на кадр, как в 3g)
    и после (карта). Медианы каналов по пустому небу каждого кропа."""
    from skyforge.preview import display_crop

    before, _ = base_color(cfg, layers, sky, sd, mode="constant")
    after, _ = base_color(cfg, layers, sky, sd, mode="map")
    rgb0, _ = neutralize(blend(layers, assign_colors(list(layers), cfg.mission, cfg.instrument, cfg.color), cfg.color),
                         cfg.color.background_percentile)
    delta, _ = sky_color_map(rgb0, sky, cfg.color.background_map_px, cfg.color.background_map_min_sky)
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
