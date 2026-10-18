"""Этап clean: между совмещением и растяжкой чинит то, что видно на картинке, но не в коде.

Сначала заполняются насыщенные ядра звёзд (stars.py), затем вычитается
крупномасштабная модель фона (background.py). Результат кешируется в
data/work/<slug>/bin<N>/clean/. Если оба механизма выключены, этап ничего не пишет,
и растяжка берёт совмещённые кадры как есть.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned, load_tiles, read_aligned_meta
from skyforge.background import background_model, tile_pedestal
from skyforge.compose import compose_layers
from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import cache_previews, contact_sheet, quick_norm, shrink
from skyforge.stars import StarMask, core_area_px, fill_rings, find_core_holes, star_mask, star_weight
from skyforge.stretch import stretch_array

# Превью фона тянется жёстче рабочей растяжки: стыки плиток — это проценты от шума,
# при обычной растяжке их не видно, а на итоге с подъёмом теней (shadows) они вылезают.
REVEAL_BLACK, REVEAL_WHITE, REVEAL_A = 1.0, 90.0, 0.02


CLEAN_VERSION = 7  # поднять, если меняется алгоритм очистки или масок: старый кеш станет недействительным


def enabled(cfg: Config) -> bool:
    return (cfg.stars.fill_cores or cfg.background.subtract or cfg.color.star_saturation < 1
            or cfg.color.shadows.scope == "nebula")


def load_nebula_sd(cfg: Config, paths: Paths, required: bool = False) -> np.ndarray | None:
    """Знаковое расстояние до края маски туманности, px сетки: > 0 внутри, < 0 снаружи."""
    p = clean_dir(cfg, paths) / "nebula_sd.npy"
    if not p.exists():
        if required:
            raise SkyforgeError("для color.shadows.scope: nebula нужна маска туманности: запусти skyforge clean")
        return None
    return np.load(p)


def load_star_weight(cfg: Config, paths: Paths) -> StarMask | None:
    """Маска звёзд с этапа clean (нужна для color.star_saturation)."""
    if cfg.color.star_saturation >= 1:
        return None
    d = clean_dir(cfg, paths)
    p, meta = d / "star_weight.npy", d / "meta.json"
    if not p.exists() or not meta.exists():
        raise SkyforgeError("для color.star_saturation нужна маска звёзд: запусти skyforge clean")
    window = json.loads(meta.read_text(encoding="utf-8"))["star_window_px"]
    meta_data = json.loads(meta.read_text(encoding="utf-8"))
    return StarMask(np.load(p), window, meta_data["star_detect_filter"])


def load_sky_mask(cfg: Config, paths: Paths) -> np.ndarray | None:
    """Пустое небо: не туманность и не звёзды, с отступом color.sky_grow_px. None, если масок нет."""
    d = clean_dir(cfg, paths)
    if not (d / "nebula_sd.npy").exists() or not (d / "star_weight.npy").exists():
        return None
    g = cfg.color.sky_grow_px
    stars = np.load(d / "star_weight.npy") > 0.01
    if g > 0:
        stars = ndimage.binary_dilation(stars, iterations=g)
    return (np.load(d / "nebula_sd.npy") < -g) & ~stars


def clean_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work(cfg.slug, cfg.bin) / "clean"


def input_key(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Ключ данных, которые идут в растяжку: совмещённые кадры плюс параметры clean."""
    key: dict[str, Any] = {"align": read_aligned_meta(cfg, paths)["key"]}
    if enabled(cfg):
        key["version"] = CLEAN_VERSION
        key["stars"] = cfg.stars.model_dump(mode="json")
        key["background"] = cfg.background.model_dump(mode="json")
    return key


def load_input(cfg: Config, paths: Paths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, _, _ = load_aligned(cfg, paths)
    key = input_key(cfg, paths)
    if not enabled(cfg):
        return arrays, key
    d = clean_dir(cfg, paths)
    meta_path = d / "meta.json"
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8"))["key"] != key:
        raise SkyforgeError("очистка не сделана или сделана с другими параметрами: сначала skyforge clean")
    return {f: np.load(d / f"{f}.npy") for f in arrays}, key


CLEAN_PRODUCTS = ("star_weight.npy", "nebula_sd.npy")


def clean(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    if not enabled(cfg):
        console.print("clean: stars.fill_cores и background.subtract выключены — растяжка возьмёт совмещённые кадры")
        return {"enabled": False}
    d = clean_dir(cfg, paths)
    key = input_key(cfg, paths)
    meta_path = d / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        files = [f"{f}.npy" for f in meta.get("filters", {})] + list(CLEAN_PRODUCTS)
        if meta.get("key") == key and all((d / n).exists() for n in files):
            cache_previews(meta.get("previews", []), d / "previews", out_dir)
            console.print(f"clean: кеш актуален ({paths.rel(d)}), пересчёт не нужен")
            info = {k: v for k, v in meta.items() if k != "key"}
            return {**info, "cached": True}
    meta_path.unlink(missing_ok=True)  # упавший пересчёт не должен оставить «актуальный» кеш со старым ключом
    info = _compute(cfg, paths, d, out_dir)
    cache_previews(info["previews"], out_dir, d / "previews")
    meta_path.write_text(json.dumps({"key": key, **info}, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**info, "cached": False}


def _compute(cfg: Config, paths: Paths, d: Path, out_dir: Path) -> dict[str, Any]:
    arrays, _, meta = load_aligned(cfg, paths)
    scale = meta["pixel_scale_arcsec"]
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    before = {f: np.asarray(a, dtype=np.float32) for f, a in arrays.items()}
    after: dict[str, np.ndarray] = {}
    holes_all = np.zeros(next(iter(before.values())).shape, dtype=bool)
    info: dict[str, Any] = {"enabled": True, "filters": {}}
    models, masks = {}, {}
    star_masks = {}
    for f, a in before.items():
        out = a
        fi: dict[str, Any] = {}
        if cfg.stars.fill_cores:
            holes = find_core_holes(out, core_area_px(cfg.stars.max_core_arcsec, scale))
            out = fill_rings(out, holes)
            holes_all |= holes
            fi["core_pixels"] = int(holes.sum())
            fi["cores"] = int(ndimage.label(holes, structure=np.ones((3, 3)))[1])
        if cfg.background.subtract:
            bc = cfg.background
            model = np.zeros_like(out)
            tiles = load_tiles(cfg, paths, f) if bc.tiles else None
            if bc.tiles and tiles is None:
                console.print(f"  [yellow]{f}: в файле нет слоя CON/CTX — плитки не выравниваю[/]")
            if tiles is not None:
                ped, ti = tile_pedestal(out, tiles, bc, scale)
                out = out - np.nan_to_num(ped)
                model += np.nan_to_num(ped)
                fi.update(tiles=ti["tiles"], tiles_measured=ti["measured"], tiles_matched=ti["matched"])
            mask = np.zeros(out.shape, dtype=bool)
            if bc.smooth:
                smooth, mask, box = background_model(out, bc, scale)
                out = out - smooth
                model += np.nan_to_num(smooth)
                fi["box_px"] = box
                fi["masked_percent"] = round(float(mask.mean() * 100), 1)
            models[f], masks[f] = model, mask
        after[f] = out.astype(np.float32)
        np.save(d / f"{f}.npy", after[f])
        star_masks[f] = star_mask(after[f], cfg.stars, scale)
        info["filters"][f] = fi
        parts = []
        if "cores" in fi:
            parts.append(f"ядер {fi['cores']} ({fi['core_pixels']} px)")
        if "tiles" in fi:
            parts.append(f"плиток {fi['tiles']}, измерено {fi['tiles_measured']}")
        if "box_px" in fi:
            parts.append(f"гладкий фон: блок {fi['box_px']} px, маска {fi['masked_percent']}%")
        console.print(f"  {f}: " + ", ".join(parts))

    detect = _detect_filter(cfg, list(after))
    smask = star_weight(star_masks, detect, cfg.stars, scale)
    weight = smask.weight
    info["star_detect_filter"] = detect
    info["star_window_px"] = smask.window_px
    np.save(d / "star_weight.npy", weight)
    info["star_mask_percent"] = round(float((weight > 0.5).mean() * 100), 2)
    console.print(f"  маска звёзд (по {detect}): {info['star_mask_percent']}% кадра")
    sd = nebula_signed_distance(after, cfg, scale)
    np.save(d / "nebula_sd.npy", sd)
    info["nebula_percent"] = round(float((sd > 0).mean() * 100), 1)
    console.print(f"  маска туманности: {info['nebula_percent']}% кадра")
    sky = load_sky_mask(cfg, paths)
    if sky is not None:
        info["blocks"] = {}
        for f in after:
            b0, n0 = block_spread(before[f], sky, cfg.background.block_px)
            b1, n1 = block_spread(after[f], sky, cfg.background.block_px)
            info["blocks"][f] = {"before": b0, "after": b1, "noise": n1, "ratio": round(b1 / n1, 3) if n1 else None}
            console.print(f"  {f}: разброс медиан блоков {cfg.background.block_px}² по небу {b0:.3g} → {b1:.3g} "
                          f"(шум фона {n1:.3g}, отношение {info['blocks'][f]['ratio']})")
    previews = []
    if cfg.background.subtract:
        _background_sheet(cfg, before, after, models, masks, out_dir / "background_model.png")
        previews.append("background_model.png")
    _star_mask_preview(cfg, after, weight, out_dir / "star_mask.png")
    previews.append("star_mask.png")
    if cfg.stars.fill_cores and holes_all.any():
        _core_sheet(cfg, before, after, holes_all, out_dir / "star_cores.png")
        previews.append("star_cores.png")
    info["previews"] = previews
    return info


def block_spread(a: np.ndarray, sky: np.ndarray, block: int) -> tuple[float, float]:
    """Разброс медиан блоков block×block по пустому небу (блоки, где неба не меньше половины)
    и шум фона (σ пикселя неба с отсечкой). Ступеньки плиток видны как разброс блоков."""
    from astropy.stats import sigma_clipped_stats

    h, w = a.shape
    meds = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = sky[y:y + block, x:x + block]
            if m.mean() >= 0.5:
                v = a[y:y + block, x:x + block][m]
                v = v[np.isfinite(v)]
                if v.size:
                    meds.append(float(np.median(v)))
    v = a[sky & np.isfinite(a)]
    noise = float(sigma_clipped_stats(v[:: max(1, v.size // 2_000_000)], sigma=3.0)[2]) if v.size else 0.0
    return (round(float(np.std(meds)), 6) if len(meds) > 1 else 0.0), round(noise, 6)


def nebula_signed_distance(layers: dict[str, np.ndarray], cfg: Config, scale: float) -> np.ndarray:
    """Маска туманности: крупные области чувствительной маски фона (сглаживание, порог, расширение
    из background.*) по всем фильтрам; звёзды с ореолом в неё не попадают по площади.
    Возвращается знаковое расстояние до её края в пикселях: из него мягкий край любой ширины."""
    from skyforge.background import source_mask

    bc = cfg.background
    max_star = cfg.stars.max_star_area_arcsec2 / scale**2
    grow = (cfg.stars.halo_arcsec + bc.mask_dilate_arcsec) / scale
    star_with_halo = np.pi * (np.sqrt(max_star / np.pi) + grow) ** 2
    nebula = np.zeros(next(iter(layers.values())).shape, dtype=bool)
    for a in layers.values():
        m = source_mask(a, bc, scale)
        labels, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            continue
        area = ndimage.sum_labels(m, labels, index=np.arange(1, n + 1))
        big = np.zeros(n + 1, dtype=bool)
        big[1:] = area > star_with_halo
        nebula |= big[labels]
    # Маска фона расширяется квадратом — скругляем углы, чтобы край подъёма теней не шёл ступеньками.
    rounding = max(1.0, bc.mask_dilate_arcsec / scale / 2)
    nebula = ndimage.gaussian_filter(nebula.astype(np.float32), rounding) > 0.5
    inside = ndimage.distance_transform_edt(nebula)
    outside = ndimage.distance_transform_edt(~nebula)
    return (inside - outside).astype(np.float32)


def _background_sheet(cfg: Config, before: dict, after: dict, models: dict, masks: dict, path: Path) -> None:
    size = cfg.preview.max_size // 4
    images, labels = [], []
    for f in models:
        b = shrink(before[f], size)
        a = shrink(after[f], size)
        m = shrink(models[f], size)
        lo, hi = np.nanpercentile(m, [1, 99])
        images += [quick_norm(b, REVEAL_BLACK, REVEAL_WHITE, REVEAL_A),
                   shrink(masks[f].astype(np.float32), size),
                   np.clip((m - lo) / (hi - lo if hi > lo else 1), 0, 1),
                   quick_norm(a, REVEAL_BLACK, REVEAL_WHITE, REVEAL_A)]
        labels += [f"{f}: до", "маска источников", "модель фона", "после вычитания"]
    contact_sheet(images, labels, path, cols=4, tile_inches=3.0,
                  title="Фон: стыки плиток и градиенты должны уйти в «модель», туманность — остаться")


def _detect_filter(cfg: Config, filters: list[str]) -> str:
    from skyforge.filters import sort_by_wavelength

    if cfg.stars.detect_filter:
        if cfg.stars.detect_filter.upper() not in filters:
            raise SkyforgeError(f"stars.detect_filter {cfg.stars.detect_filter} нет среди фильтров {filters}")
        return cfg.stars.detect_filter.upper()
    return sort_by_wavelength(filters, cfg.mission, cfg.instrument)[0]


def _star_mask_preview(cfg: Config, after: dict, weight: np.ndarray, path: Path) -> None:
    """Самый коротковолновый фильтр серым, маска звёзд поверх красным: туманность должна остаться серой."""
    first = _detect_filter(cfg, list(after))
    size = cfg.preview.max_size
    grey = quick_norm(shrink(after[first], size), REVEAL_BLACK, REVEAL_WHITE, REVEAL_A)
    w = np.clip(shrink(weight, size), 0, 1)
    rgb = np.stack([np.clip(grey + 0.6 * w, 0, 1), grey * (1 - 0.5 * w), grey * (1 - 0.5 * w)], axis=-1)
    contact_sheet([rgb], [f"маска звёзд поверх {first}: красное обесцвечивается, туманность — нет"], path, cols=1,
                  tile_inches=9)


def _core_sheet(cfg: Config, before: dict, after: dict, holes: np.ndarray, path: Path) -> None:
    """Кроп 100% вокруг самой большой дыры: цветной предпросмотр до и после заполнения."""
    labels, n = ndimage.label(holes, structure=np.ones((3, 3)))
    sizes = ndimage.sum_labels(holes, labels, index=np.arange(1, n + 1))
    cy, cx = (int(round(v)) for v in ndimage.center_of_mass(holes, labels, int(np.argmax(sizes)) + 1))
    z = cfg.preview.core_zoom
    h, w = holes.shape
    y0, x0 = int(np.clip(cy - z // 2, 0, max(h - z, 0))), int(np.clip(cx - z // 2, 0, max(w - z, 0)))
    sl = (slice(y0, y0 + z), slice(x0, x0 + z))
    # Предпросмотр без нейтрализации фона: её статистика по крошечному кропу была бы случайной.
    # Без подъёма теней: маски туманности ещё может не быть, а ядру звезды он не нужен.
    c = cfg.model_copy(update={"color": cfg.color.model_copy(update={
        "neutral_background": False, "shadows": cfg.color.shadows.model_copy(update={"value": 0.0})})})
    renders = []
    for src in (before, after):
        layers = {f: stretch_array(a[sl], cfg.stretch.for_filter(f).model_copy(update={"anchor": "frame"}),
                                   stats_from=a)[0] for f, a in src.items()}
        renders.append(compose_layers(c, layers)[0])
    contact_sheet(renders, ["до: дыры насыщения пустые", "после: ядро заполнено кольцами"], path, cols=2,
                  title="Ядро самой насыщенной звезды, 100%: должно быть белым, без цветной точки",
                  interpolation="nearest")
