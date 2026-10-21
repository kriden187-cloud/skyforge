"""Финиш: денойз, резкость, экспорт 16-битного TIFF и PNG с метаданными.

В файлы не пишется время: тот же конфиг даёт байт-в-байт тот же файл.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle

from skyforge import __version__
from skyforge.clean import input_key
from skyforge.compose import composite_path, compose_key
from skyforge.config import Config, FinishConfig
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, save_png, to_display, to_uint8

# denoise: 1.0 соответствует весу TV-регуляризации 0.1 на шкале яркости 0–1 — это уже сильное сглаживание.
DENOISE_WEIGHT_AT_MAX = 0.1


def unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Нерезкая маска: картинка + amount·(картинка − размытая). Своя, потому что
    skimage.filters.unsharp_mask 0.26 на RGB float32 возвращал тёмный кадр (максимум 0.49 вместо 1)."""
    blurred = np.stack([gaussian_filter(rgb[..., k], sigma=radius) for k in range(rgb.shape[-1])], axis=-1)
    return (rgb + np.float32(amount) * (rgb - blurred)).astype(np.float32)


def split_stars(rgb: np.ndarray, weight: np.ndarray, feather_px: float, window_px: int,
                origin: tuple[int, int] = (0, 0), core_sigma: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Звёзды отдельно от туманности.

    Под маской звёзд clean (вес > 0,5: звезда с ореолом) — местная медиана соседей вне маски (фон под звездой).
    Звезда — только то, что выше этого фона заметнее разброса внутри масок (core_sigma·σ по яркости): ореол
    маски — десятки пикселей, и туманность под ним иначе гасла бы вместе со звездой. Слой звёзд —
    избыток над фоном сверх tau с мягким краем feather_px вокруг ядра; без звёзд = кадр − звёзды."""
    from skyforge.compose import LUMA
    from skyforge.field import block_median_map

    mask = np.asarray(weight) > 0.5
    if not mask.any():
        return rgb, np.zeros_like(rgb)
    fill = np.stack([block_median_map(np.where(mask, np.nan, rgb[..., k]), window_px, origin, fill_empty=True)
                     for k in range(3)], axis=-1)
    excess = rgb - fill
    e = excess @ LUMA
    v = e[mask]
    tau = core_sigma * 1.4826 * float(np.median(np.abs(v - np.median(v))))
    core = mask & (e > tau)
    if not core.any():
        return rgb, np.zeros_like(rgb)
    from scipy.ndimage import distance_transform_edt

    alpha = np.clip(1 - distance_transform_edt(~core) / feather_px, 0, 1) * mask
    # в слой звёзд — только свет выше разброса туманности (tau): светлые детали туманности рядом со звездой не гаснут
    stars = (alpha[..., None] * np.clip(excess - np.float32(tau), 0, None)).astype(np.float32)
    return (rgb - stars).astype(np.float32), stars


def shrink_stars(stars: np.ndarray, radius: float) -> np.ndarray:
    """Уменьшение звёзд: минимум по диску радиуса radius (морфологическое сжатие) по каждому каналу."""
    from scipy.ndimage import grey_erosion

    if radius <= 0:
        return stars
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (yy**2 + xx**2) <= radius**2
    return np.stack([grey_erosion(stars[..., k], footprint=disk) for k in range(3)], axis=-1).astype(np.float32)


def local_contrast(rgb: np.ndarray, nebula_sd: np.ndarray, lc: Any, px: float = 1.0) -> np.ndarray:
    """Локальный контраст внутри маски туманности по самому яркому каналу пикселя, один множитель на три канала
    (оттенок не меняется). Подъём деталей плавно сходится к 1: m' = 1 − (1 − m)·exp(−подъём/(1 − m)) — канал
    за 1 не выходит и не прижимается к общему потолку (деление пикселя на максимум делало рельс, 3i)."""
    m = rgb.max(axis=-1)
    detail = m - gaussian_filter(m, sigma=lc.radius * px)
    w = np.clip(nebula_sd / (lc.feather_px * px), 0, 1)
    boost = np.float32(lc.amount) * detail * w
    room = np.maximum(1 - m, np.float32(1e-6))
    new = np.where(boost > 0, 1 - room * np.exp(-boost / room), np.maximum(m + boost, 0))
    gain = np.where(m > 1e-6, new / np.maximum(m, 1e-6), 1).astype(np.float32)
    return np.clip(rgb * gain[..., None], 0, 1).astype(np.float32)


def finish_array(rgb: np.ndarray, f: FinishConfig, star_weight: np.ndarray | None = None,
                 nebula_sd: np.ndarray | None = None, origin: tuple[int, int] = (0, 0), px: float = 1.0) -> np.ndarray:
    """Финиш: (3i) звёзды отдельно от туманности — локальный контраст без звёзд, звёзды × scale и сжатие;
    затем денойз и резкость. px — масштаб длин (для уменьшенных копий в sweep); nebula_sd — в пикселях этой сетки."""
    out = np.asarray(rgb, dtype=np.float32)
    st, lc = f.stars, f.local_contrast
    stars_on = star_weight is not None and (st.scale < 1 or st.shrink > 0)
    contrast_on = nebula_sd is not None and lc.amount > 0
    if stars_on or contrast_on:
        if star_weight is not None:
            starless, stars = split_stars(out, star_weight, st.feather_px * px,
                                          max(8, int(round(st.fill_window_px * px))), origin)
        else:
            starless, stars = out, np.zeros_like(out)
        if contrast_on:
            starless = local_contrast(starless, nebula_sd, lc, px)
        out = np.clip(starless + np.float32(st.scale) * shrink_stars(stars, st.shrink * px), 0, 1).astype(np.float32)
    if f.denoise > 0:
        out = denoise_tv_chambolle(out, weight=f.denoise * DENOISE_WEIGHT_AT_MAX, channel_axis=-1,
                                   max_num_iter=f.denoise_iters).astype(np.float32)
    if f.sharpen > 0:
        out = unsharp(out, f.sharpen_radius, f.sharpen)
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 1)


def export_metadata(cfg: Config, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": f"{cfg.target} — {cfg.mission.upper()} {cfg.instrument.upper()}",
        "credit": f"{cfg.credit}; обработка: {cfg.author}",
        "software": f"skyforge {__version__}",
        "filters": extra.get("colors"),
        "config_digest": cfg.digest(),
    }


def export(rgb: np.ndarray, base: Path, meta: dict[str, Any], nodata: np.ndarray | None = None) -> dict[str, str]:
    """nodata — пиксели без единого фильтра: в PNG прозрачные (альфа 0), в TIFF чёрные."""
    base.parent.mkdir(parents=True, exist_ok=True)
    tif = base.with_suffix(".tif")
    png = base.with_suffix(".png")
    u16 = to_display((np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16))
    desc = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    # Строки TIFF — только ASCII, поэтому кириллица там в виде \\uXXXX; в PNG — как есть.
    tifffile.imwrite(tif, u16, photometric="rgb", compression="zlib",
                     description=json.dumps(meta, ensure_ascii=True, sort_keys=True),
                     software=meta["software"], metadata=None)
    info = PngInfo()
    info.add_itxt("Title", meta["title"])
    info.add_itxt("Author", meta["credit"])
    info.add_itxt("Software", meta["software"])
    info.add_itxt("Description", desc)
    u8 = to_uint8(rgb)
    if nodata is not None and nodata.any():
        u8 = np.dstack([u8, np.where(nodata, 0, 255).astype(np.uint8)])
    Image.fromarray(to_display(u8)).save(png, pnginfo=info)
    return {"tif": tif.name, "png": png.name}


def nodata_mask(coverage: np.ndarray, fill_gaps_px: int, clean_filled: np.ndarray | None = None) -> np.ndarray:
    """Пиксели без единого фильтра. 3б: при заполнении швов в стиле (clean.fill_gaps_px) шов, пустой во всех фильтрах,
    заполнен и уже не nodata — то же правило (cosmics.fill_gaps_linear) по геометрии покрытия; иначе финиш обнулял
    заполненный шов, и в PNG он выходил прозрачной линией. clean_filled — есть данные хоть в одном фильтре после clean
    (фаза 4: ядра звёзд, пустые во всех фильтрах на этапе align, заполняет clean — иначе чёрная точка в звезде)."""
    nodata = coverage == 0
    if fill_gaps_px > 0 and nodata.any():
        from skyforge.cosmics import fill_gaps_linear

        _, filled = fill_gaps_linear(np.where(nodata, np.nan, 1.0).astype(np.float32), fill_gaps_px)
        nodata = nodata & ~filled
    if clean_filled is not None:             # фаза 4: ядра звёзд, заполненные clean (fill_cores), — данные
        nodata = nodata & ~clean_filled
    return nodata


def fill_nodata(rgb: np.ndarray, nodata: np.ndarray, mode: str, sky_level: float) -> np.ndarray:
    """Пиксели без единого фильтра: black — 0; sky — ровный уровень неба (0–1) во всех каналах, без шума (3б)."""
    out = rgb.copy()
    out[nodata] = sky_level if mode == "sky" else 0.0
    return out


def _clean_filled(cfg: Config, paths: Paths, shape: tuple[int, int]) -> np.ndarray | None:
    """Есть данные хоть в одном фильтре после clean — только если clean заполняет ядра (stars.fill_cores)."""
    if not cfg.stars.fill_cores:
        return None
    from skyforge.clean import clean_dir

    d = clean_dir(cfg, paths)
    out = None
    for f in cfg.filters:
        p = d / f"{f}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r")
            if a.shape == shape:
                out = np.isfinite(a) if out is None else out | np.isfinite(a)
    return out


def finish_masks(cfg: Config, paths: Paths) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int]]:
    """Маска звёзд и знаковое расстояние до края туманности (этап clean), начало поля статистики —
    только если финиш их использует (3i); иначе None."""
    from skyforge.align import read_aligned_meta
    from skyforge.clean import clean_dir, load_nebula_sd

    st, lc = cfg.finish.stars, cfg.finish.local_contrast
    if not (st.scale < 1 or st.shrink > 0 or lc.amount > 0):
        return None, None, (0, 0)
    p = clean_dir(cfg, paths) / "star_weight.npy"
    sw = np.load(p) if p.exists() else None
    rect = read_aligned_meta(cfg, paths).get("intersection")
    return sw, load_nebula_sd(cfg, paths), ((rect[0], rect[2]) if rect else (0, 0))


def finish_region(rgb: np.ndarray, f: FinishConfig, sw: np.ndarray | None, sd: np.ndarray | None,
                  origin: tuple[int, int], sl: tuple[slice, slice]) -> np.ndarray:
    """Финиш кропа с запасом вокруг (размытие локального контраста и окно медианы видят соседей)."""
    pad = int(3 * f.local_contrast.radius + f.stars.fill_window_px + f.stars.feather_px)
    h, w = rgb.shape[:2]
    r0, r1 = max(0, sl[0].start - pad), min(h, sl[0].stop + pad)
    c0, c1 = max(0, sl[1].start - pad), min(w, sl[1].stop + pad)
    big = (slice(r0, r1), slice(c0, c1))
    out = finish_array(rgb[big], f, sw[big] if sw is not None else None, sd[big] if sd is not None else None,
                       (origin[0] - r0, origin[1] - c0))
    return out[sl[0].start - r0:sl[0].stop - r0, sl[1].start - c0:sl[1].stop - c0]


def _finish_sheets(cfg: Config, rgb: np.ndarray, sw: Any, sd: Any, origin: tuple[int, int],
                   crops: list[Any], out_dir: Path) -> list[str]:
    """Превью 3i: поле звёзд при scale 1 / 0,7 / 0,5 и core, inner при amount 0 / 0,3 / 0,6 (кропы 100%)."""
    made = []
    by = {c.name: c for c in crops}
    f = cfg.finish
    if sw is not None and "sky3" in by:
        c = by["sky3"]
        sl = display_crop(rgb.shape, c.x, c.y, c.size, cfg.bin)
        imgs, labels = [], []
        for s in (1.0, 0.7, 0.5):
            fs = f.model_copy(update={"stars": f.stars.model_copy(update={"scale": s})})
            imgs.append(finish_region(rgb, fs, sw, sd, origin, sl))
            labels.append(f"звёзды × {s:g}, shrink {f.stars.shrink:g}")
        contact_sheet(imgs, labels, out_dir / "finish_stars.png", cols=3, tile_inches=4.5, interpolation="nearest",
                      title="Поле звёзд 100%: меньше и тише, без тёмных колец и без «дырок» в туманности")
        made.append("finish_stars.png")
    if sd is not None:
        imgs, labels = [], []
        for name in ("core", "inner"):
            if name not in by:
                continue
            c = by[name]
            sl = display_crop(rgb.shape, c.x, c.y, c.size, cfg.bin)
            for amt in (0.0, 0.3, 0.6):
                fa = f.model_copy(update={"local_contrast": f.local_contrast.model_copy(update={"amount": amt})})
                imgs.append(finish_region(rgb, fa, sw, sd, origin, sl))
                labels.append(f"{name}: local_contrast {amt:g}, radius {f.local_contrast.radius:g}")
        if imgs:
            contact_sheet(imgs, labels, out_dir / "finish_contrast.png", cols=3, tile_inches=4.5,
                          interpolation="nearest",
                          title="Локальный контраст 100%: структура внутри кольца, волокна без «пластилина» и ореолов")
            made.append("finish_contrast.png")
    return made


def output_base(cfg: Config, out_dir: Path) -> Path:
    suffix = f"_bin{cfg.bin}" if cfg.bin > 1 else ""
    style = f"_{cfg.style}" if cfg.style else ""
    return out_dir / f"{cfg.slug}{style}{suffix}"


def finish(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    cp = composite_path(cfg, paths)
    meta_path = cp.with_suffix(".json")
    if not cp.exists() or not meta_path.exists():
        raise SkyforgeError("нет цветной композиции: сначала skyforge compose")
    cmeta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cmeta["key"] != compose_key(cfg, input_key(cfg, paths)):
        raise SkyforgeError("композиция сделана с другими параметрами: перезапусти skyforge stretch и compose")

    from skyforge.align import load_coverage, read_aligned_meta, with_resolved_crops

    rgb = np.load(cp)
    sw, sd, origin = finish_masks(cfg, paths)
    done = finish_array(rgb, cfg.finish, sw, sd, origin)
    coverage = load_coverage(cfg, paths)
    nodata = (nodata_mask(coverage, cfg.clean.fill_gaps_px, _clean_filled(cfg, paths, done.shape[:2]))
              if coverage is not None and coverage.shape == done.shape[:2] else None)
    if nodata is not None:  # без единого фильтра — чёрный или уровень неба (в PNG прозрачный)
        if cfg.finish.nodata_fill == "sky" and cfg.stretch.sky_level is None:
            raise SkyforgeError("finish.nodata_fill: sky требует stretch.sky_level")
        done = fill_nodata(done, nodata, cfg.finish.nodata_fill, (cfg.stretch.sky_level or 0) / 255)
    files = export(done, output_base(cfg, out_dir), export_metadata(cfg, cmeta), nodata)
    console.print(f"  итог: {files['png']}, {files['tif']} ({done.shape[1]}×{done.shape[0]})")

    pv = cfg.preview
    save_png(done, out_dir / "finish_preview.png", pv.max_size)
    # Кроп 100% из центра до и после финиша: видно, что сделали денойз и резкость.
    z = min(pv.zoom, *done.shape[:2])
    cy, cx = done.shape[0] // 2, done.shape[1] // 2
    sl = (slice(cy - z // 2, cy - z // 2 + z), slice(cx - z // 2, cx - z // 2 + z))
    contact_sheet([np.clip(np.nan_to_num(rgb[sl]), 0, 1), done[sl]],
                  ["до финиша, 100%", f"denoise {cfg.finish.denoise}, sharpen {cfg.finish.sharpen}, 100%"],
                  out_dir / "finish_zoom.png", cols=2)
    crops = []
    for c in with_resolved_crops(cfg, paths).report.crops:
        sl = display_crop(done.shape, c.x, c.y, c.size, cfg.bin)
        name = f"crop_{c.name}.png"
        Image.fromarray(to_display(to_uint8(done[sl]))).save(out_dir / name)
        crops.append({"name": c.name, "file": name})
    previews = ["finish_preview.png", "finish_zoom.png"]
    if sw is not None or sd is not None:
        previews += _finish_sheets(cfg, rgb, sw, sd, origin, with_resolved_crops(cfg, paths).report.crops, out_dir)
    am = read_aligned_meta(cfg, paths)
    if am.get("mode") == "union" and am.get("intersection"):
        _frame_preview(done, am["intersection"], out_dir / "finish_frame.png", pv.max_size)
        previews.append("finish_frame.png")
    return {"files": files, "shape": list(done.shape[:2]), "crops": crops, "previews": previews}


def _frame_preview(done: np.ndarray, inter: list[int], path: Path, max_size: int) -> None:
    """Итог по объединению покрытий; пунктир — прежний кадр (поле, где есть все фильтры)."""
    import matplotlib.pyplot as plt

    f = max(1, int(np.ceil(max(done.shape[:2]) / max_size)))
    img = to_display(to_uint8(done[::f, ::f]))
    h = img.shape[0]
    r0, r1, c0, c1 = (v / f for v in inter)
    fig, ax = plt.subplots(figsize=(9, 9 * img.shape[0] / img.shape[1]))
    ax.imshow(img, interpolation="lanczos")
    ax.plot([c0, c1, c1, c0, c0], [h - r0, h - r0, h - r1, h - r1, h - r0], "w--", lw=1.0)
    ax.set_title("Итог по объединению покрытий; пунктир — прежний кадр (все фильтры)")
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close(fig)
