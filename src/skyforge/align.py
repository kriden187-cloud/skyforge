"""Совмещение фильтров на общую сетку по WCS и обрезка до общего поля.

Сетка — опорный фильтр (уменьшенный в bin раз), при north_up повёрнутый севером вверх.
Все фильтры, включая опорный, переводятся на неё одной интерполяцией.
Результат кешируется в data/work/<slug>/bin<N>/aligned/ по ключу из файлов и параметров.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.wcs import WCS, Sip
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from reproject.mosaicking import find_optimal_celestial_wcs
from skimage.registration import phase_cross_correlation

from skyforge.config import Config
from skyforge.fetch import Product, require_products
from skyforge.filters import wavelength_um
from skyforge.fitsio import load_context, load_sci
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import (PREVIEW_A, PREVIEW_BLACK, PREVIEW_VERSION, PREVIEW_WHITE, cache_previews, checkerboard,
                              contact_sheet, quick_norm, save_png, shrink)

ALIGN_VERSION = 2  # поднять, если меняется алгоритм: старый кеш станет недействительным


# --- геометрия ---


def binned_wcs(wcs: WCS, factor: int, shape: tuple[int, int]) -> WCS:
    """WCS для картинки, усреднённой блоками factor×factor.

    Новый пиксель i' покрывает старые factor*i' … factor*i'+factor-1, его центр
    в старых координатах factor*i' + (factor-1)/2. В FITS (с единицы) это
    CRPIX' = (CRPIX - 0.5)/factor + 0.5, а масштаб растёт в factor раз.
    SIP-коэффициенты A_pq масштабируются как factor^(p+q-1).
    """
    w = wcs.deepcopy()
    if factor == 1:
        return w
    w.wcs.crpix = (np.asarray(w.wcs.crpix) - 0.5) / factor + 0.5
    if w.wcs.has_cd():
        w.wcs.cd = np.asarray(w.wcs.cd) * factor
    else:
        w.wcs.cdelt = np.asarray(w.wcs.cdelt) * factor
    if w.sip is not None:
        def scale(m: np.ndarray | None) -> np.ndarray | None:
            if m is None:
                return None
            p, q = np.indices(m.shape)
            return m * float(factor) ** (p + q - 1)

        s = w.sip
        w.sip = Sip(scale(s.a), scale(s.b), scale(s.ap), scale(s.bp), w.wcs.crpix)
    w.pixel_shape = (shape[1] // factor, shape[0] // factor)
    w.wcs.set()
    return w


def bin_image(data: np.ndarray, wcs: WCS, factor: int) -> tuple[np.ndarray, WCS]:
    if factor == 1:
        return data, wcs
    h, w = data.shape
    h2, w2 = h // factor * factor, w // factor * factor
    blocks = data[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(blocks, axis=(1, 3)).astype(np.float32)
    return out, binned_wcs(wcs, factor, data.shape)


def target_grid(ref_wcs: WCS, ref_shape: tuple[int, int], north_up: bool) -> tuple[WCS, tuple[int, int]]:
    if not north_up:
        return ref_wcs, ref_shape
    scale = float(np.mean(proj_plane_pixel_scales(ref_wcs))) * u.deg
    wcs, shape = find_optimal_celestial_wcs([(ref_shape, ref_wcs)], resolution=scale, frame="icrs")
    return wcs, tuple(int(s) for s in shape)


def inscribed_rect(mask: np.ndarray, min_coverage: float) -> tuple[int, int, int, int]:
    """Прямоугольник без пустых краёв: начинаем с рамки вокруг данных и срезаем худший край,
    пока на каждом краю не останется не меньше min_coverage заполненных пикселей.
    Возвращает (r0, r1, c0, c1), полуоткрытые интервалы."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise SkyforgeError("фильтры не пересекаются на небе: общего поля нет")
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    while r1 - r0 > 1 and c1 - c0 > 1:
        edges = {
            "top": mask[r0, c0:c1].mean(),
            "bottom": mask[r1 - 1, c0:c1].mean(),
            "left": mask[r0:r1, c0].mean(),
            "right": mask[r0:r1, c1 - 1].mean(),
        }
        worst = min(edges, key=edges.get)
        if edges[worst] >= min_coverage:
            break
        if worst == "top":
            r0 += 1
        elif worst == "bottom":
            r1 -= 1
        elif worst == "left":
            c0 += 1
        else:
            c1 -= 1
    return int(r0), int(r1), int(c0), int(c1)


def reproject_nan_safe(data: np.ndarray, wcs: WCS, grid_wcs: WCS, shape: tuple[int, int], order: str) -> np.ndarray:
    """Репроекция, в которой NaN не расползаются по всему кадру.

    Сплайны порядка 2–3 считают коэффициенты по всему массиву, и один NaN портит всё.
    Для них дыры временно заполняются медианой, а маска дыр переносится отдельно билинейно.
    """
    nan = ~np.isfinite(data)
    if order in ("nearest-neighbor", "bilinear") or not nan.any():
        out, _ = reproject_interp((data, wcs), grid_wcs, shape_out=shape, order=order)
        return out.astype(np.float32)
    filled = np.where(nan, np.float32(np.nanmedian(data[::4, ::4])), data)
    out, footprint = reproject_interp((filled, wcs), grid_wcs, shape_out=shape, order=order)
    holes, _ = reproject_interp((nan.astype(np.float32), wcs), grid_wcs, shape_out=shape, order="bilinear")
    out[(holes > 0) | (footprint == 0)] = np.nan
    return out.astype(np.float32)


def _reproject_labels(labels: np.ndarray, wcs_binned: WCS, factor: int, grid_wcs: WCS,
                      shape: tuple[int, int]) -> np.ndarray:
    """Карта областей CON на общую сетку: ближайший сосед, номера остаются целыми; вне поля −1."""
    if factor > 1:
        h, w = labels.shape
        labels = labels[factor // 2:h // factor * factor:factor, factor // 2:w // factor * factor:factor]
    out, _ = reproject_interp((labels.astype(np.float64), wcs_binned), grid_wcs, shape_out=shape,
                              order="nearest-neighbor")
    return np.where(np.isfinite(out), np.rint(out), -1).astype(np.int32)


def load_tiles(cfg: Config, paths: Paths, f: str) -> np.ndarray | None:
    p = aligned_dir(cfg, paths) / f"{f}.tiles.npy"
    return np.load(p) if p.exists() else None


def residual_shift(ref: np.ndarray, other: np.ndarray, max_side: int = 2048) -> tuple[float, float]:
    """Остаточный сдвиг между фильтрами (dy, dx) в пикселях сетки по взаимной корреляции центра кадра."""
    h, w = ref.shape
    s = min(max_side, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    a = np.nan_to_num(ref[y0:y0 + s, x0:x0 + s])
    b = np.nan_to_num(other[y0:y0 + s, x0:x0 + s])
    shift, _, _ = phase_cross_correlation(a, b, upsample_factor=10, normalization=None)
    return float(shift[0]), float(shift[1])


# --- кеш ---


def aligned_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work(cfg.slug, cfg.bin) / "aligned"


def _cache_key(cfg: Config, prods: dict[str, Product], reference: str) -> dict[str, Any]:
    return {
        "version": ALIGN_VERSION,
        "files": {f: [p.path, p.size] for f, p in prods.items()},
        "reference": reference,
        "bin": cfg.bin,
        "north_up": cfg.finish.north_up,
        "align": cfg.align.model_dump(),
    }


def read_aligned_meta(cfg: Config, paths: Paths) -> dict[str, Any]:
    meta_path = aligned_dir(cfg, paths) / "meta.json"
    if not meta_path.exists():
        raise SkyforgeError(f"нет совмещённых кадров для bin {cfg.bin}: сначала skyforge align")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_aligned(cfg: Config, paths: Paths) -> tuple[dict[str, np.ndarray], WCS, dict[str, Any]]:
    d = aligned_dir(cfg, paths)
    meta = read_aligned_meta(cfg, paths)
    from astropy.io import fits

    wcs = WCS(fits.Header.fromstring(meta["wcs"], sep="\n"))
    want = [f for f in meta["filters"] if not cfg.filters or f in cfg.filters]
    arrays = {f: np.load(d / f"{f}.npy") for f in want}
    return arrays, wcs, meta


def pick_reference(cfg: Config, prods: dict[str, Product]) -> str:
    if cfg.reference:
        if cfg.reference not in prods:
            raise SkyforgeError(f"опорного фильтра {cfg.reference} нет на диске (есть: {', '.join(prods)})")
        return cfg.reference
    # Без указания — самая длинная волна: у неё самый крупный пиксель, сетка не раздувается.
    return max(prods, key=lambda f: wavelength_um(f, cfg.mission, cfg.instrument))


# --- этап ---


def align(cfg: Config, paths: Paths, out_dir: Path, *, force: bool = False) -> dict[str, Any]:
    prods = require_products(cfg, paths)
    reference = pick_reference(cfg, prods)
    d = aligned_dir(cfg, paths)
    key = _cache_key(cfg, prods, reference)
    meta_path = d / "meta.json"

    cached = False
    if not force and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("key") == key and all((d / f"{f}.npy").exists() for f in prods):
            cached = True
            console.print(f"align: кеш актуален ({paths.rel(d)}), пересчёт не нужен")

    if not cached:
        meta = _compute(cfg, paths, prods, reference, d, key)

    # Превью от стиля не зависят: при актуальном кеше копируются, а не считаются заново.
    pv_key = {**cfg.preview.model_dump(), "version": PREVIEW_VERSION}
    pv_path = d / "previews" / "info.json"
    pv = json.loads(pv_path.read_text(encoding="utf-8")) if pv_path.exists() else None
    if cached and pv and pv.get("preview") == pv_key:
        cache_previews(pv["info"]["previews"], d / "previews", out_dir)
        info = dict(pv["info"])
        meta = read_aligned_meta(cfg, paths)
        for f, s in info["shifts_px"].items():
            console.print(f"  остаточный сдвиг {f} относительно {reference}: dy={s['dy']:+.2f} dx={s['dx']:+.2f} px")
    else:
        arrays, _, meta = load_aligned(cfg.model_copy(update={"filters": []}), paths)
        info = _previews(cfg, arrays, reference, meta, out_dir)
        cache_previews(info["previews"], out_dir, d / "previews")
        pv_path.write_text(json.dumps({"preview": pv_key, "info": info}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    info.update({"cached": cached, "reference": reference, "shape": meta["shape"],
                 "crop": meta["crop"], "pixel_scale_arcsec": meta["pixel_scale_arcsec"]})
    return info


def _compute(cfg: Config, paths: Paths, prods: dict[str, Product], reference: str,
             d: Path, key: dict[str, Any]) -> dict[str, Any]:
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    ref_data, ref_wcs = load_sci(paths.abs(prods[reference].path))
    ref_data, ref_wcs = bin_image(ref_data, ref_wcs, cfg.bin)
    grid_wcs, grid_shape = target_grid(ref_wcs, ref_data.shape, cfg.finish.north_up)
    del ref_data
    console.print(f"align: сетка {reference}, {grid_shape[1]}×{grid_shape[0]} px, bin {cfg.bin}, "
                  f"{'север вверх' if cfg.finish.north_up else 'ориентация опорного кадра'}")

    valid = np.ones(grid_shape, dtype=bool)
    for f, prod in prods.items():
        console.print(f"  {f}: репроекция…")
        data, wcs = load_sci(paths.abs(prod.path))
        data, wcs = bin_image(data, wcs, cfg.bin)
        out = reproject_nan_safe(data, wcs, grid_wcs, grid_shape, cfg.align.order)
        valid &= np.isfinite(out)
        np.save(d / f"{f}.npy", out)
        labels = load_context(paths.abs(prod.path))
        if labels is not None:
            np.save(d / f"{f}.tiles.npy", _reproject_labels(labels, wcs, cfg.bin, grid_wcs, grid_shape))
        del data, out

    r0, r1, c0, c1 = inscribed_rect(valid, cfg.align.crop_min_coverage)
    console.print(f"  обрезка до общего поля: {c1 - c0}×{r1 - r0} px из {grid_shape[1]}×{grid_shape[0]}")
    for f in prods:
        for name in (f"{f}.npy", f"{f}.tiles.npy"):
            if (d / name).exists():
                np.save(d / name, np.load(d / name)[r0:r1, c0:c1].copy())
    crop_wcs = grid_wcs.slice((slice(r0, r1), slice(c0, c1)))

    meta = {
        "key": key,
        "filters": list(prods),
        "reference": reference,
        "shape": [r1 - r0, c1 - c0],
        "crop": [r0, r1, c0, c1],
        "pixel_scale_arcsec": round(float(np.mean(proj_plane_pixel_scales(crop_wcs))) * 3600, 5),
        # Заголовок построчно, чтобы читать обратно через Header.fromstring(sep="\n").
        "wcs": "\n".join(card.image for card in crop_wcs.to_header(relax=True).cards),
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _previews(cfg: Config, arrays: dict[str, np.ndarray], reference: str, meta: dict[str, Any],
              out_dir: Path) -> dict[str, Any]:
    pv = cfg.preview
    norm = {}
    for f, a in arrays.items():
        norm[f] = quick_norm(np.asarray(a), PREVIEW_BLACK, PREVIEW_WHITE, PREVIEW_A)

    # Для шахматки берём фильтр, сильнее всего отличающийся по длине волны: там ошибка заметнее всего.
    lam = {f: wavelength_um(f, cfg.mission, cfg.instrument) for f in arrays}
    other = max((f for f in arrays if f != reference), key=lambda f: abs(lam[f] - lam[reference]), default=reference)

    cell = max(4, pv.checker_cell // cfg.bin)
    board = checkerboard(norm[reference], norm[other], cell)
    save_png(board, out_dir / "align_check.png", pv.max_size)

    # Кроп 100% вокруг самой яркой точки опорного кадра — обычно звезда.
    ref = norm[reference]
    z = min(pv.zoom, *ref.shape)
    sm = shrink(np.nan_to_num(ref), 8)
    iy, ix = np.unravel_index(np.argmax(sm), sm.shape)
    cy = int(np.clip(iy * 8 + 4, z // 2, ref.shape[0] - z // 2))
    cx = int(np.clip(ix * 8 + 4, z // 2, ref.shape[1] - z // 2))
    sl = (slice(cy - z // 2, cy - z // 2 + z), slice(cx - z // 2, cx - z // 2 + z))
    zoom_cell = max(4, z // 6)
    zoom = checkerboard(ref[sl], norm[other][sl], zoom_cell)
    overlay = np.stack([ref[sl], norm[other][sl], norm[other][sl]], axis=-1)
    contact_sheet([zoom, overlay], [f"шахматка 100%: {reference} / {other}", f"наложение: {reference} красный, {other} голубой"],
                  out_dir / "align_zoom.png", cols=2, title="Звёзды должны совпадать: без ступенек на границах клеток и без цветных ореолов")

    shifts = {}
    for f in arrays:
        if f == reference:
            continue
        dy, dx = residual_shift(norm[reference], norm[f])
        shifts[f] = {"dy": round(dy, 2), "dx": round(dx, 2)}
    worst = max((abs(s["dy"]) + abs(s["dx"]) for s in shifts.values()), default=0.0)
    for f, s in shifts.items():
        console.print(f"  остаточный сдвиг {f} относительно {reference}: dy={s['dy']:+.2f} dx={s['dx']:+.2f} px")
    if worst > 1.0:
        console.print("[yellow]  сдвиг больше пикселя: посмотри align_zoom.png — возможно, кадры не совмещены[/]")
    return {"checker_pair": [reference, other], "shifts_px": shifts,
            "previews": ["align_check.png", "align_zoom.png"]}
