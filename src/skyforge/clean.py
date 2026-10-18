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
from skyforge.background import (background_model, block_model, boundary_jumps, nexp_levels, nexp_regions,
                                 tile_pedestal, tile_planes)
from skyforge.compose import compose_layers
from skyforge.cosmics import fill_mean, find_cosmics, find_gaps, remove_cosmics
from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import cache_previews, contact_sheet, quick_norm, shrink
from skyforge.stars import StarMask, core_area_px, fill_rings, find_core_holes, star_mask, star_weight
from skyforge.stretch import stretch_array

# Превью фона тянется жёстче рабочей растяжки: стыки плиток — это проценты от шума,
# при обычной растяжке их не видно, а на итоге с подъёмом теней (shadows) они вылезают.
REVEAL_BLACK, REVEAL_WHITE, REVEAL_A = 1.0, 90.0, 0.02


CLEAN_VERSION = 13  # поднять, если меняется алгоритм очистки или масок: старый кеш станет недействительным


def enabled(cfg: Config) -> bool:
    return (cfg.stars.fill_cores or cfg.background.subtract or cfg.color.star_saturation < 1
            or cfg.color.shadows.scope == "nebula" or cfg.clean.cosmics.clean or cfg.clean.gaps.fill)


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
    rect = read_aligned_meta(cfg, paths).get("intersection")
    return StarMask(np.load(p), window, meta_data["star_detect_filter"], (rect[0], rect[2]) if rect else (0, 0))


def load_star_weight_map(cfg: Config, paths: Paths) -> np.ndarray | None:
    """Вес звёзд с этапа clean без условия на star_saturation — для веса подъёма за маской (halo_k): звёзды в нём
    заменяются уровнем неба, иначе вокруг каждой звезды вес поднимается. None — нет файла."""
    p = clean_dir(cfg, paths) / "star_weight.npy"
    return np.load(p) if p.exists() else None


def load_sky_mask(cfg: Config, paths: Paths, nebula: str = "nebula_sd.npy") -> np.ndarray | None:
    """Пустое небо: не туманность и не звёзды, с отступом color.sky_grow_px. None, если масок нет.
    nebula — файл маски туманности (nebula_sd_prev.npy — прежняя, для сравнения)."""
    d = clean_dir(cfg, paths)
    if not (d / nebula).exists() or not (d / "star_weight.npy").exists():
        return None
    g = cfg.color.sky_grow_px
    stars = np.load(d / "star_weight.npy") > 0.01
    if g > 0:
        stars = ndimage.binary_dilation(stars, iterations=g)
    return (np.load(d / nebula) < -g) & ~stars


def saturated_stars(holes_by_f: dict[str, np.ndarray], min_core_px: int, min_filters: int) -> list[list[int]]:
    """Звёзды с насыщенным ядром: компоненты объединения дыр насыщения площадью ≥ min_core_px, которые есть
    хотя бы в min_filters фильтрах. [строка, столбец центра; площадь; число фильтров] на сетке."""
    if not holes_by_f:
        return []
    union = np.logical_or.reduce(list(holes_by_f.values()))
    labels, n = ndimage.label(union, structure=np.ones((3, 3)))
    out = []
    for i in range(1, n + 1):
        comp = labels == i
        area = int(comp.sum())
        nf = int(sum(h[comp].any() for h in holes_by_f.values()))
        if area >= min_core_px and nf >= min_filters:
            r, c = ndimage.center_of_mass(comp)
            out.append([int(round(r)), int(round(c)), area, nf])
    return out


def load_saturated_stars(cfg: Config, paths: Paths) -> list[list[int]]:
    meta = clean_dir(cfg, paths) / "meta.json"
    if not meta.exists():
        return []
    return json.loads(meta.read_text(encoding="utf-8")).get("saturated_stars", [])


def star_exclusion(cfg: Config, paths: Paths, shape: tuple[int, int]) -> np.ndarray:
    """Пиксели звёзд для выборки процентилей: маска звёзд clean и диски вокруг звёзд с насыщенным ядром."""
    out = np.zeros(shape, dtype=bool)
    p = clean_dir(cfg, paths) / "star_weight.npy"
    if p.exists():
        out |= np.load(p) > 0.01
    r = cfg.stretch.exclude_radius_px / cfg.bin
    if r > 0:
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        for row, col, _, _ in load_saturated_stars(cfg, paths):
            out |= (yy - row) ** 2 + (xx - col) ** 2 <= r * r
    return out


def clean_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work(cfg.slug, cfg.bin) / "clean"


def input_key(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Ключ данных, которые идут в растяжку: совмещённые кадры плюс параметры clean."""
    key: dict[str, Any] = {"align": read_aligned_meta(cfg, paths)["key"]}
    if enabled(cfg):
        key["version"] = CLEAN_VERSION
        key["stars"] = cfg.stars.model_dump(mode="json")
        key["background"] = cfg.background.model_dump(mode="json")
        # 3b: в ключе только если включены — кеш clean у JWST не сбрасывается
        if cfg.clean.cosmics.clean:
            key["cosmics"] = cfg.clean.cosmics.model_dump(mode="json")
        if cfg.clean.gaps.fill:
            key["gaps"] = cfg.clean.gaps.model_dump(mode="json")
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
    from skyforge.align import with_resolved_crops
    from skyforge.field import make_field

    arrays, grid_wcs, meta = load_aligned(cfg, paths)
    cfg = with_resolved_crops(cfg, paths)
    scale = meta["pixel_scale_arcsec"]
    field = make_field(list(arrays.values()), meta.get("intersection"))
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    before = {f: np.asarray(a, dtype=np.float32) for f, a in arrays.items()}
    cosmic_masks: dict[str, np.ndarray] = {}
    if cfg.clean.cosmics.clean:
        cosmic_masks = find_cosmics(before, cfg.clean.cosmics, scale)
    after: dict[str, np.ndarray] = {}
    holes_all = np.zeros(next(iter(before.values())).shape, dtype=bool)
    holes_by_f: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"enabled": True, "filters": {}}
    models, masks = {}, {}
    star_masks = {}
    tile_maps: dict[str, np.ndarray] = {}
    nexp_maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for f, a in before.items():
        out = a
        fi: dict[str, Any] = {}
        if f in cosmic_masks:
            out = remove_cosmics(out, cosmic_masks[f])
            fi["cosmic_pixels"] = int(cosmic_masks[f].sum())
        if cfg.stars.fill_cores:
            holes = find_core_holes(out, core_area_px(cfg.stars.max_core_arcsec, scale))
            out = fill_rings(out, holes)
            holes_all |= holes
            holes_by_f[f] = holes
            fi["core_pixels"] = int(holes.sum())
            fi["cores"] = int(ndimage.label(holes, structure=np.ones((3, 3)))[1])
        if cfg.clean.gaps.fill:  # после ядер: иначе мелкие дыры ядер заполнились бы средним, а не кольцами
            gaps = find_gaps(out, int(round(cfg.clean.gaps.max_width_arcsec / scale / 2)))
            out = fill_mean(out, gaps)
            fi["gap_pixels"] = int(gaps.sum())
        if cfg.background.subtract:
            bc = cfg.background
            model = np.zeros_like(out)
            tiles = load_tiles(cfg, paths, f) if bc.tiles else None
            if bc.tiles and tiles is None:
                console.print(f"  [yellow]{f}: в файле нет слоя CON/CTX — плитки не выравниваю[/]")
            if tiles is not None:
                fit = tile_planes if bc.tile_model == "plane" else tile_pedestal
                ped, ti = fit(out, tiles, bc, scale)
                out = out - np.nan_to_num(ped)
                model += np.nan_to_num(ped)
                fi.update(tiles=ti["tiles"], tiles_measured=ti["measured"], tiles_matched=ti["matched"],
                          tiles_continued=ti.get("continued", 0))
                tile_maps[f] = np.nan_to_num(ped)
            if bc.nexp_levels:
                ne = nexp_on_grid(cfg, paths, f, grid_wcs, out.shape)
                if ne is None:
                    console.print(f"  [yellow]{f}: нет слоя NEXP (не своя сборка) — уровни областей не свожу[/]")
                else:
                    ped, ni_ = nexp_levels(out, nexp_regions(ne, np.isfinite(out), bc.nexp_smooth_px, bc.nexp_min_area_px), bc, scale)
                    out = out - np.nan_to_num(ped)
                    model += np.nan_to_num(ped)
                    fi["nexp"] = ni_
                    nexp_maps[f] = (ne, np.nan_to_num(ped))
            mask = np.zeros(out.shape, dtype=bool)
            if bc.smooth:
                smooth, mask, box = (block_model if bc.model == "blocks" else background_model)(out, bc, scale)
                out = out - smooth
                model += np.nan_to_num(smooth)
                fi["box_px"] = box
                fi["masked_percent"] = round(float(mask.mean() * 100), 1)
            models[f], masks[f] = model, mask
        after[f] = out.astype(np.float32)
        np.save(d / f"{f}.npy", after[f])
        star_masks[f] = star_mask(after[f], cfg.stars, scale, field)
        info["filters"][f] = fi
        parts = []
        if "cosmic_pixels" in fi:
            parts.append(f"следы лучей {fi['cosmic_pixels']} px")
        if "gap_pixels" in fi:
            parts.append(f"зазоры {fi['gap_pixels']} px")
        if "cores" in fi:
            parts.append(f"ядер {fi['cores']} ({fi['core_pixels']} px)")
        if "tiles" in fi:
            parts.append(f"плиток {fi['tiles']}, измерено {fi['tiles_measured']}")
        if "nexp" in fi:
            parts.append(f"области NEXP {fi['nexp']['regions']}, сведено {fi['nexp']['matched']}, "
                         f"наибольшая поправка {fi['nexp']['max_abs']:.3g}")
        if "box_px" in fi:
            parts.append(f"гладкий фон: блок {fi['box_px']} px, маска {fi['masked_percent']}%")
        console.print(f"  {f}: " + ", ".join(parts))

    detect = _detect_filter(cfg, list(after))
    smask = star_weight(star_masks, detect, cfg.stars, scale)
    weight = smask.weight
    info["star_detect_filter"] = detect
    info["saturated_stars"] = saturated_stars(holes_by_f, cfg.stars.saturated_min_core_px,
                                              cfg.stars.saturated_min_filters)
    if info["saturated_stars"]:
        console.print("  звёзды с насыщенным ядром (строка, столбец сетки; площадь; фильтров): "
                      + ", ".join(f"({r}, {c}; {a}; {k})" for r, c, a, k in info["saturated_stars"]))
    info["star_window_px"] = smask.window_px
    np.save(d / "star_weight.npy", weight)
    info["star_mask_percent"] = round(float((weight > 0.5).mean() * 100), 2)
    console.print(f"  маска звёзд (по {detect}): {info['star_mask_percent']}% кадра")
    nebula, ni = nebula_mask(after, cfg, scale, star_masks[detect][0], holes_all, smask.window_px, field)
    sd = signed_distance(nebula)
    np.save(d / "nebula_sd.npy", sd)
    # прежняя маска (квадрат, со звёздами) — только для сравнения в отчёте и превью
    old = nebula_mask_square(after, cfg, scale, field)
    np.save(d / "nebula_sd_prev.npy", signed_distance(old))
    info["nebula_percent"] = round(float(nebula.mean() * 100), 1)
    info["nebula_percent_prev"] = round(float(old.mean() * 100), 1)
    info["nebula"] = ni
    console.print(f"  маска туманности: {info['nebula_percent']}% кадра (прежняя {info['nebula_percent_prev']}%); "
                  f"исключено компонент: звёздных {ni['dropped_star_components']}, "
                  f"отдельных с насыщенным ядром {ni['dropped_saturated']}")
    sky = load_sky_mask(cfg, paths)
    if sky is not None:
        info["blocks"] = {}
        for f in after:
            b0, n0 = block_spread(before[f], sky, cfg.background.block_px)
            b1, n1 = block_spread(after[f], sky, cfg.background.block_px)
            info["blocks"][f] = {"before": b0, "after": b1, "noise": n1, "ratio": round(b1 / n1, 3) if n1 else None}
            tiles = load_tiles(cfg, paths, f)
            if tiles is not None:
                j0, nb = boundary_jumps(before[f], tiles, sky, cfg.background.tile_band_px)
                j1, _ = boundary_jumps(after[f], tiles, sky, cfg.background.tile_band_px)
                info["blocks"][f].update(jumps_before=j0, jumps_after=j1, boundaries=nb)
            bf = info["blocks"][f]
            console.print(f"  {f}: скачки на границах CON по небу {bf.get('jumps_before', '—')}σ → {bf.get('jumps_after', '—')}σ; "
                          f"разброс медиан блоков {b0:.3g} → {b1:.3g} (шум {n1:.3g})")
        if tile_maps:
            _tile_planes_sheet(cfg, before, after, tile_maps, out_dir / "tile_planes.png")
            info["extra_previews"] = ["tile_planes.png"]
    previews = []
    if nexp_maps:
        _nexp_levels_sheet(nexp_maps, out_dir / "nexp_levels.png")
        previews.append("nexp_levels.png")
    if cfg.background.subtract:
        _background_sheet(cfg, before, after, models, masks, out_dir / "background_model.png")
        previews.append("background_model.png")
    _star_mask_preview(cfg, after, weight, out_dir / "star_mask.png")
    if cosmic_masks:
        _cosmics_sheet(cfg, before, after, cosmic_masks, out_dir / "cosmics.png")
        previews.append("cosmics.png")
    previews.append("star_mask.png")
    _nebula_mask_preview(cfg, after, old, nebula, out_dir / "nebula_mask.png")
    previews.append("nebula_mask.png")
    if cfg.stars.fill_cores and holes_all.any():
        _core_sheet(cfg, before, after, holes_all, out_dir / "star_cores.png")
        previews.append("star_cores.png")
    info["previews"] = previews + info.pop("extra_previews", [])
    return info


def nexp_on_grid(cfg: Config, paths: Paths, f: str, grid_wcs: Any, shape: tuple[int, int]) -> np.ndarray | None:
    """Слой NEXP своей сборки фильтра f на сетке кадра (ближайший сосед); None — у продукта слоя нет."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from reproject import reproject_interp

    from skyforge.fetch import require_products

    path = paths.abs(require_products(cfg, paths)[f].path)
    with fits.open(path, memmap=True, mode="readonly") as h:
        if "NEXP" not in h:
            return None
        ne = np.asarray(h["NEXP"].data, dtype=np.float32)
        w = WCS(h["SCI"].header)
    if cfg.bin > 1:
        from skyforge.align import bin_image

        ne, w = bin_image(ne, w, cfg.bin)
    g, _ = reproject_interp((ne, w), grid_wcs, shape_out=shape, order="nearest-neighbor")
    return np.rint(np.nan_to_num(g)).astype(np.int16)


def _nexp_levels_sheet(maps: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> None:
    imgs, labels = [], []
    for f, (ne, ped) in maps.items():
        imgs.append(shrink(ne.astype(np.float32) / max(1, int(ne.max())), 500))
        labels.append(f"{f}: число экспозиций")
        lim = float(np.percentile(np.abs(ped), 99.5)) or 1.0
        imgs.append(shrink(np.clip(0.5 + ped / (2 * lim), 0, 1), 500))
        labels.append(f"{f}: поправка уровня по областям NEXP (серое 0, ±{lim:.3g})")
    contact_sheet(imgs, labels, path, cols=4, tile_inches=3.6, title="3k: уровни областей числа экспозиций")


def _tile_planes_sheet(cfg: Config, before: dict, after: dict, maps: dict, path: Path) -> None:
    """По каждому фильтру: что вычли (карта поправки плиток) и пустое небо (кроп sky) до и после."""
    from skyforge.preview import display_crop

    crop = next((c for c in cfg.report.crops if c.name == "sky"), None)
    size = cfg.preview.max_size // 4
    images, labels = [], []
    for f, m in maps.items():
        lim = float(np.nanpercentile(np.abs(m), 99)) or 1.0
        images.append(np.clip(0.5 + shrink(m, size) / (2 * lim), 0, 1))
        labels.append(f"{f}: поправка плиток (±{lim:.3g})")
        if crop is not None:
            sl = display_crop(m.shape, crop.x, crop.y, crop.size, cfg.bin)
            for name, src in (("до", before[f]), ("после", after[f])):
                images.append(quick_norm(src[sl], REVEAL_BLACK, REVEAL_WHITE, REVEAL_A))
                labels.append(f"{f}: небо {name}")
    contact_sheet(images, labels, path, cols=3, tile_inches=3.2, interpolation="nearest",
                  title=f"Поправка плиток ({cfg.background.tile_model}): на «после» прямых границ между участками неба быть не должно")


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


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """Знаковое расстояние до края маски в пикселях (внутри +, снаружи −): из него мягкий край любой ширины."""
    return (ndimage.distance_transform_edt(mask) - ndimage.distance_transform_edt(~mask)).astype(np.float32)


def fill_stars(a: np.ndarray, stars: np.ndarray, window_px: int, origin: tuple[int, int] = (0, 0)) -> np.ndarray:
    """Пиксели звёзд → местная медиана фона без звёзд (окно window_px). Остальные пиксели не меняются.
    origin — начало поля статистики: от него идёт сетка блоков."""
    from skyforge.field import block_median_map

    x = np.where(stars, np.nan, a).astype(np.float32)
    if not np.isfinite(x).any():
        return np.array(a, dtype=np.float32, copy=True)
    bg = block_median_map(x, window_px, origin, fill_empty=True)
    out = np.array(a, dtype=np.float32, copy=True)
    out[stars] = bg[stars]
    return out


def _star_with_halo_px(cfg: Config, scale: float) -> float:
    """Площадь самой крупной звезды с ореолом и расширением маски: компоненты меньше — не туманность."""
    max_star = cfg.stars.max_star_area_arcsec2 / scale**2
    grow = (cfg.stars.halo_arcsec + cfg.background.mask_dilate_arcsec) / scale
    return float(np.pi * (np.sqrt(max_star / np.pi) + grow) ** 2)


def nebula_mask(layers: dict[str, np.ndarray], cfg: Config, scale: float, stars: np.ndarray | None = None,
                cores: np.ndarray | None = None, window_px: int = 95,
                field: Any = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Маска туманности (3h): по каждому фильтру звёзды (stars) заменяются местной медианой фона, на этом
    кадре — чувствительная маска фона (background.*), расширенная диском nebula_dilate_arcsec. Крупные
    компоненты — туманность, кроме тех, где звёзд больше nebula_star_fraction. В объединении по фильтрам
    отдельные от главной компоненты с насыщенным ядром (cores) — яркие звёзды со спайками — исключаются."""
    from skyforge.background import source_mask

    bc = cfg.background
    min_area = _star_with_halo_px(cfg, scale)
    r = (bc.mask_dilate_arcsec if bc.nebula_dilate_arcsec is None else bc.nebula_dilate_arcsec) / scale
    plain = bc.model_copy(update={"mask_dilate_arcsec": 0.0})
    nebula = np.zeros(next(iter(layers.values())).shape, dtype=bool)
    dropped_stars = 0
    for a in layers.values():
        finite = np.isfinite(a)
        origin = field.origin if field is not None else (0, 0)
        src = fill_stars(a, stars & finite, window_px, origin) if stars is not None else a
        m = (ndimage.distance_transform_edt(~source_mask(src, plain, scale, field=field)) <= r) & finite
        labels, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum_labels(m, labels, index=idx)
        star_frac = (ndimage.sum_labels(stars, labels, index=idx) / np.maximum(area, 1)
                     if stars is not None else np.zeros(n))
        big = area > min_area
        starry = big & (star_frac > bc.nebula_star_fraction)
        dropped_stars += int(starry.sum())
        keep = np.zeros(n + 1, dtype=bool)
        keep[1:] = big & ~starry
        nebula |= keep[labels]
    dropped_sat = 0
    if bc.nebula_drop_saturated and cores is not None and cores.any() and nebula.any():
        labels, n = ndimage.label(nebula, structure=np.ones((3, 3)))
        idx = np.arange(1, n + 1)
        main = int(np.argmax(ndimage.sum_labels(nebula, labels, index=idx))) + 1
        # с запасом 3 px: незаполненное ядро (NaN) само в маску не входит, только его край
        sat = ndimage.sum_labels(ndimage.binary_dilation(cores, iterations=3), labels, index=idx) > 0
        drop = np.zeros(n + 1, dtype=bool)
        drop[1:] = sat & (idx != main)
        dropped_sat = int(drop.sum())
        nebula &= ~drop[labels]
    return nebula, {"dilate_px": round(float(r), 1), "dropped_star_components": dropped_stars,
                    "dropped_saturated": dropped_sat}


def nebula_mask_square(layers: dict[str, np.ndarray], cfg: Config, scale: float, field: Any = None) -> np.ndarray:
    """Прежняя маска (до 3h): квадратное расширение, звёзды не вычитаются. Только для сравнения."""
    from skyforge.background import source_mask

    bc = cfg.background
    star_with_halo = _star_with_halo_px(cfg, scale)
    nebula = np.zeros(next(iter(layers.values())).shape, dtype=bool)
    for a in layers.values():
        m = source_mask(a, bc, scale, field=field)
        labels, n = ndimage.label(m, structure=np.ones((3, 3)))
        if n == 0:
            continue
        area = ndimage.sum_labels(m, labels, index=np.arange(1, n + 1))
        big = np.zeros(n + 1, dtype=bool)
        big[1:] = area > star_with_halo
        nebula |= big[labels]
    # Маска фона расширяется квадратом — скругляем углы, чтобы край подъёма теней не шёл ступеньками.
    rounding = max(1.0, bc.mask_dilate_arcsec / scale / 2)
    return ndimage.gaussian_filter(nebula.astype(np.float32), rounding) > 0.5


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
        # кропы пустого неба 100% до и после (sky, sky2 из report.crops), растяжка общая для «до» и «после»
        from skyforge.preview import display_crop

        for cr in [c for c in cfg.report.crops if c.name.startswith("sky")][:2]:
            sl = display_crop(before[f].shape, cr.x, cr.y, cr.size, cfg.bin)
            bb = before[f][sl]
            lo_, hi_ = np.nanpercentile(bb, [REVEAL_BLACK, REVEAL_WHITE])
            norm = lambda z: np.clip((np.nan_to_num(z, nan=lo_) - lo_) / (hi_ - lo_ if hi_ > lo_ else 1), 0, 1)  # noqa: E731
            images += [norm(bb), norm(after[f][sl] + (np.nanmedian(bb) - np.nanmedian(after[f][sl])))]
            labels += [f"{cr.name} до", f"{cr.name} после"]
    n_crops = len([c for c in cfg.report.crops if c.name.startswith("sky")][:2])
    contact_sheet(images, labels, path, cols=4 + 2 * n_crops, tile_inches=3.0,
                  title="Фон: стыки плиток и градиенты должны уйти в «модель», туманность — остаться")


def _cosmics_sheet(cfg: Config, before: dict, after: dict, masks: dict, path: Path) -> None:
    """Кроп 100% там, где следов больше всего: до и после по каждому фильтру."""
    total = sum(m.astype(np.float32) for m in masks.values())
    z = cfg.preview.zoom
    dens = ndimage.uniform_filter(total, z)
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    h, w = total.shape
    y0, x0 = int(np.clip(cy - z // 2, 0, max(h - z, 0))), int(np.clip(cx - z // 2, 0, max(w - z, 0)))
    sl = (slice(y0, y0 + z), slice(x0, x0 + z))
    images, labels = [], []
    for f in masks:
        s = cfg.stretch.for_filter(f)
        lo_hi = (s.black, s.white, s.a)
        images += [quick_norm(before[f][sl], *lo_hi), quick_norm(after[f][sl], *lo_hi)]
        labels += [f"{f}: до", f"после ({int(masks[f].sum())} px следов во всём кадре)"]
    contact_sheet(images, labels, path, cols=2, tile_inches=4, interpolation="nearest",
                  title="Следы космических лучей: звёзды и туманность должны остаться")


def _detect_filter(cfg: Config, filters: list[str]) -> str:
    from skyforge.filters import sort_by_wavelength

    if cfg.stars.detect_filter:
        if cfg.stars.detect_filter.upper() not in filters:
            raise SkyforgeError(f"stars.detect_filter {cfg.stars.detect_filter} нет среди фильтров {filters}")
        return cfg.stars.detect_filter.upper()
    return sort_by_wavelength(filters, cfg.mission, cfg.instrument)[0]


def _nebula_mask_preview(cfg: Config, after: dict, old: np.ndarray, new: np.ndarray, path: Path) -> None:
    """Прежняя и новая маска туманности поверх самого коротковолнового фильтра: красное — была только прежняя,
    зелёное — только новая, жёлтый контур — край новой."""
    first = _detect_filter(cfg, list(after))
    size = cfg.preview.max_size
    grey = quick_norm(shrink(after[first], size), REVEAL_BLACK, REVEAL_WHITE, REVEAL_A)
    o = shrink(old.astype(np.float32), size) > 0.5
    n = shrink(new.astype(np.float32), size) > 0.5
    rgb = np.stack([grey, grey, grey], axis=-1) * 0.7
    rgb[o & ~n] = rgb[o & ~n] * 0.4 + np.array([0.6, 0, 0])
    rgb[n & ~o] = rgb[n & ~o] * 0.4 + np.array([0, 0.6, 0])
    rgb[n ^ ndimage.binary_erosion(n, iterations=1)] = (1, 1, 0)
    contact_sheet([rgb], [f"маска туманности: прежняя {old.mean() * 100:.1f}% → новая {new.mean() * 100:.1f}% кадра; "
                          "красное — убрано, зелёное — добавлено"], path, cols=1, tile_inches=9)


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
        "neutral_background": "off", "shadows": cfg.color.shadows.model_copy(update={"value": 0.0})})})
    renders = []
    for src in (before, after):
        layers = {f: stretch_array(a[sl], cfg.stretch.for_filter(f).model_copy(update={"anchor": "frame"}),
                                   stats_from=a)[0] for f, a in src.items()}
        renders.append(compose_layers(c, layers)[0])
    contact_sheet(renders, ["до: дыры насыщения пустые", "после: ядро заполнено кольцами"], path, cols=2,
                  title="Ядро самой насыщенной звезды, 100%: должно быть белым, без цветной точки",
                  interpolation="nearest")
