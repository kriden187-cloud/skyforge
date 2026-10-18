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
from skyforge.fitsio import load_context, load_sci, target_coords
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import (PREVIEW_A, PREVIEW_BLACK, PREVIEW_VERSION, PREVIEW_WHITE, cache_previews, checkerboard,
                              contact_sheet, quick_norm, save_png, shrink)

ALIGN_VERSION = 9  # поднять, если меняется алгоритм: старый кеш станет недействительным


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


def footprint_angle(valid: np.ndarray, step_deg: float = 0.25) -> float:
    """Угол (градусы), при повороте на который рамка вокруг непустых пикселей наименьшая."""
    f = max(1, int(np.ceil(max(valid.shape) / 400)))
    pts = np.argwhere(valid[::f, ::f]).astype(np.float64) * f
    if len(pts) < 3:
        return 0.0
    pts -= pts.mean(axis=0)
    best, best_area = 0.0, np.inf
    for t in np.arange(0.0, 90.0, step_deg):
        c, s = np.cos(np.radians(t)), np.sin(np.radians(t))
        y = c * pts[:, 0] - s * pts[:, 1]
        x = s * pts[:, 0] + c * pts[:, 1]
        area = np.ptp(y) * np.ptp(x)
        if area < best_area:
            best, best_area = float(t), area
    return best


def rotated_grid(ref_wcs: WCS, valid: np.ndarray, angle_deg: float) -> tuple[WCS, tuple[int, int]]:
    """Сетка того же масштаба, повёрнутая на angle_deg, по размеру реальных данных.

    Старый пиксель p (с нуля) и новый p' связаны как p − c = R·(p' − c'); отсюда
    CD' = CD·R и CRPIX' − 1 = c' + Rᵀ·(CRPIX − 1 − c)."""
    t = np.radians(angle_deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])  # в осях (x, y)
    yy, xx = np.nonzero(valid)
    pts = np.stack([xx, yy]).astype(np.float64)  # (x, y)
    c = pts.mean(axis=1)
    q = rot.T @ (pts - c[:, None])
    lo, hi = q.min(axis=1), q.max(axis=1)
    shape = (int(np.ceil(hi[1] - lo[1])) + 2, int(np.ceil(hi[0] - lo[0])) + 2)
    c_new = -lo + 1
    w = WCS(naxis=2)
    w.wcs.ctype = list(ref_wcs.wcs.ctype)[:2] if "SIP" not in ref_wcs.wcs.ctype[0] else ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = ref_wcs.wcs.crval
    w.wcs.cd = ref_wcs.pixel_scale_matrix @ rot
    w.wcs.crpix = c_new + rot.T @ (np.asarray(ref_wcs.wcs.crpix) - 1 - c) + 1
    w.pixel_shape = (shape[1], shape[0])
    w.wcs.set()
    return w, shape


def target_grid(ref_wcs: WCS, ref_shape: tuple[int, int], north_up: bool,
                ref_valid: np.ndarray | None = None, auto_rotate: bool = False) -> tuple[WCS, tuple[int, int]]:
    if auto_rotate and ref_valid is not None:  # 3b: поле Hubble повёрнуто — сетка вдоль его формы
        angle = footprint_angle(ref_valid)
        console.print(f"  поворот сетки по форме поля: {angle:.2f}°")
        return rotated_grid(ref_wcs, ref_valid, angle)
    if not north_up:
        return ref_wcs, ref_shape
    scale = float(np.mean(proj_plane_pixel_scales(ref_wcs))) * u.deg
    wcs, shape = find_optimal_celestial_wcs([(ref_shape, ref_wcs)], resolution=scale, frame="icrs")
    return wcs, tuple(int(s) for s in shape)


def union_grid(grid_wcs: WCS, grid_shape: tuple[int, int], files: list[Path]) -> tuple[WCS, tuple[int, int]]:
    """Сетка опорного кадра, расширенная на целое число пикселей так, чтобы вместить данные всех файлов.
    Решётка та же (сдвиг CRPIX на целое): пиксели внутри прежней сетки репроецируются как раньше."""
    from astropy.io import fits

    from skyforge.fitsio import _sci_index, _wcs

    lo = np.array([0.0, 0.0])
    hi = np.array([grid_shape[1] - 1.0, grid_shape[0] - 1.0])
    for p in files:
        with fits.open(p, memmap=True, mode="readonly") as h:
            i = _sci_index(h)
            data = h[i].data
            wcs = _wcs(h[i].header)
            ok = np.isfinite(data[::8, ::8])
            rows, cols = np.flatnonzero(ok.any(axis=1)) * 8, np.flatnonzero(ok.any(axis=0)) * 8
        if rows.size == 0:
            continue
        ys = [rows[0] - 8, rows[-1] + 8]
        xs = [cols[0] - 8, cols[-1] + 8]
        cx, cy = np.meshgrid(xs, ys)
        gx, gy = grid_wcs.world_to_pixel(wcs.pixel_to_world(cx.ravel(), cy.ravel()))
        lo = np.minimum(lo, [np.min(gx), np.min(gy)])
        hi = np.maximum(hi, [np.max(gx), np.max(gy)])
    c_lo, r_lo = (int(np.floor(v)) for v in lo)
    c_hi, r_hi = (int(np.ceil(v)) + 1 for v in hi)
    w = grid_wcs.deepcopy()
    w.wcs.crpix = np.asarray(w.wcs.crpix) - [c_lo, r_lo]
    if w.sip is not None:
        w.sip = Sip(w.sip.a, w.sip.b, w.sip.ap, w.sip.bp, w.wcs.crpix)
    w.wcs.set()
    return w, (r_hi - r_lo, c_hi - c_lo)


def _nexp_on_grid(path: Path, grid_wcs: WCS, grid_shape: tuple[int, int]) -> np.ndarray | None:
    """Число экспозиций (слой NEXP своей сборки уровня 2) на сетке кадра; None, если слоя нет (i2d)."""
    from astropy.io import fits

    with fits.open(path, memmap=True, mode="readonly") as h:
        if "NEXP" not in h:
            return None
        ne = np.asarray(h["NEXP"].data, dtype=np.float32)
        wcs = WCS(h["SCI"].header)
    out, _ = reproject_interp((ne, wcs), grid_wcs, shape_out=grid_shape, order="nearest-neighbor")
    return np.nan_to_num(out, nan=0.0)


def _fill_small_holes(ok: np.ndarray, hole_px: int) -> np.ndarray:
    from scipy import ndimage

    labels, n = ndimage.label(~ok)
    if n:
        sizes = ndimage.sum_labels(~ok, labels, index=np.arange(1, n + 1))
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        small = np.zeros(n + 1, dtype=bool)
        small[1:] = sizes <= hole_px
        small[border] = False
        ok = ok | small[labels]
    return ok


def channel_rect(finite: dict[str, np.ndarray], groups: list[list[str]], hole_px: int = 2000,
                 inset_px: float = 0) -> tuple[int, int, int, int]:
    """Прямоугольник, внутри которого в каждой группе фильтров есть хотя бы один: края без канала (полосы,
    где есть только короткие или только длинные волны) срезаются. Мелкие дыры внутри (битые пиксели,
    ядра звёзд) не в счёт — по ним прямоугольник не сжимается. inset_px — отступ внутрь от границы
    покрытия каждой группы: крайние столбцы и строки с неполным набором не попадают в кадр."""
    from scipy import ndimage

    ok = np.ones(next(iter(finite.values())).shape, dtype=bool)
    for g in groups:
        have = [finite[f] for f in g if f in finite]
        if have:
            gok = _fill_small_holes(np.logical_or.reduce(have), hole_px)
            if inset_px > 0:
                # граница кадра сетки — тоже граница покрытия
                gok = ndimage.distance_transform_edt(np.pad(gok, 1))[1:-1, 1:-1] > inset_px
            ok &= gok
    return inscribed_rect(_fill_small_holes(ok, hole_px), 1.0)


def stats_field(cfg: Config, paths: Paths, arrays: list[np.ndarray]) -> Any:
    """Поле статистики (skyforge.field): общее поле всех фильтров из метаданных align."""
    from skyforge.field import make_field

    return make_field(arrays, read_aligned_meta(cfg, paths).get("intersection"))


def resolve_crops(crops: list[Any], wcs: WCS, height: int, bin_: int) -> list[Any]:
    """Кропы с центром на небе (ra, dec) → x, y на итоговой картинке при bin 1 для этой сетки."""
    out = []
    for c in crops:
        if c.ra is None:
            out.append(c)
            continue
        col, row = wcs.world_to_pixel_values(c.ra, c.dec)
        out.append(c.model_copy(update={"x": int(round(float(col))) * bin_,
                                        "y": int(round(height - 1 - float(row))) * bin_}))
    return out


def with_resolved_crops(cfg: Config, paths: Paths) -> Config:
    """Копия конфига, где report.crops пересчитаны в пиксели текущей сетки align."""
    if not any(c.ra is not None for c in cfg.report.crops):
        return cfg
    from astropy.io import fits

    meta = read_aligned_meta(cfg, paths)
    wcs = WCS(fits.Header.fromstring(meta["wcs"], sep="\n"))
    crops = resolve_crops(cfg.report.crops, wcs, meta["shape"][0], cfg.bin)
    return cfg.model_copy(update={"report": cfg.report.model_copy(update={"crops": crops})})


def load_coverage(cfg: Config, paths: Paths) -> np.ndarray | None:
    """Число фильтров на пиксель (этап align); None для старого кеша без карты."""
    p = aligned_dir(cfg, paths) / "coverage.npy"
    return np.load(p) if p.exists() else None


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


def largest_rect_with(valid: np.ndarray, ty: int, tx: int) -> tuple[int, int, int, int] | None:
    """Наибольший прямоугольник без пустот, содержащий пиксель (ty, tx). Ищется на уменьшенной
    маске, где блок считается заполненным, только если заполнен целиком, — поэтому на полной
    сетке прямоугольник гарантированно без NaN. (r0, r1, c0, c1), полуоткрытые."""
    f = max(1, int(np.ceil(max(valid.shape) / 300)))
    h, w = valid.shape[0] // f, valid.shape[1] // f
    small = valid[:h * f, :w * f].reshape(h, f, w, f).all(axis=(1, 3))
    sy, sx = ty // f, tx // f
    if not (0 <= sy < h and 0 <= sx < w) or not small[sy, sx]:
        return None
    bad = np.vstack([np.zeros((1, w), dtype=np.int32), np.cumsum(~small, axis=0, dtype=np.int32)])
    best, best_area = None, 0
    for r0 in range(sy, -1, -1):
        if bad[sy + 1, sx] - bad[r0, sx]:
            break
        for r1 in range(sy, h):
            ok = (bad[r1 + 1] - bad[r0]) == 0
            if not ok[sx]:
                break
            left = sx - int(np.argmax(~ok[sx::-1])) + 1 if (~ok[:sx + 1]).any() else 0
            right = sx + int(np.argmax(~ok[sx:])) if (~ok[sx:]).any() else w
            area = (r1 - r0 + 1) * (right - left)
            if area > best_area:
                best, best_area = (r0, r1 + 1, left, right), area
    r0, r1, c0, c1 = best
    return r0 * f, r1 * f, c0 * f, c1 * f


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
        # поля 3b (Hubble) — в ключе, только если отличны от умолчаний: кеш совмещения JWST не сбрасывается
        "align": cfg.align.model_dump(exclude={k for k, v in (("crop_mode", "trim"), ("auto_rotate", False))
                                               if getattr(cfg.align, k) == v}),
        "destripe": cfg.clean.destripe.model_dump(),
        "stars": cfg.stars.mask_key(),              # маски для destripe
        "background": cfg.background.model_dump(),
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
    if (d / "previews" / "destripe_check.png").exists() and "destripe_check.png" not in info["previews"]:
        cache_previews(["destripe_check.png"], d / "previews", out_dir)
        info["previews"] = [*info["previews"], "destripe_check.png"]
    if meta.get("destripe"):
        info["destripe"] = meta["destripe"]
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
    grid_wcs, grid_shape = target_grid(ref_wcs, ref_data.shape, cfg.finish.north_up,
                                       np.isfinite(ref_data), cfg.align.auto_rotate)
    del ref_data
    orient = ("по форме поля" if cfg.align.auto_rotate else
              "север вверх" if cfg.finish.north_up else "ориентация опорного кадра")
    console.print(f"align: сетка {reference}, {grid_shape[1]}×{grid_shape[0]} px, bin {cfg.bin}, {orient}")

    if cfg.align.crop == "union":
        grid_wcs, grid_shape = union_grid(grid_wcs, grid_shape, [paths.abs(p.path) for p in prods.values()])
        console.print(f"  объединение покрытий: сетка расширена до {grid_shape[1]}×{grid_shape[0]} px")
    valid = np.ones(grid_shape, dtype=bool)
    count = np.zeros(grid_shape, dtype=np.uint8)
    finite: dict[str, np.ndarray] = {}
    stripes: dict[str, Any] = {}
    keep: dict[str, Any] | None = None
    for f, prod in prods.items():
        data, wcs = load_sci(paths.abs(prod.path))
        if cfg.clean.destripe.enabled:
            data, stripes[f], kept = _destripe_native(cfg, data, wcs, load_context(paths.abs(prod.path)))
            pf = cfg.clean.destripe.preview_filter
            if (pf and f == pf.upper()) or (not pf and (keep is None or stripes[f]["before"] > keep["info"]["before"])):
                keep = {"filter": f, "info": stripes[f], **kept}  # для превью: заданный или самый полосатый
            del kept
        console.print(f"  {f}: репроекция…")
        data, wcs = bin_image(data, wcs, cfg.bin)
        out = reproject_nan_safe(data, wcs, grid_wcs, grid_shape, cfg.align.order)
        valid &= np.isfinite(out)
        count += np.isfinite(out)
        if cfg.align.crop == "union" and (cfg.align.channel_groups or cfg.align.require_all):
            finite[f] = np.isfinite(out)
            if cfg.align.min_exposures_sw and wavelength_um(f, cfg.mission, cfg.instrument) < 2.5:
                ne = _nexp_on_grid(paths.abs(prod.path), grid_wcs, grid_shape)
                if ne is not None:
                    finite[f"{f}:nexp"] = ne >= cfg.align.min_exposures_sw
        np.save(d / f"{f}.npy", out)
        labels = load_context(paths.abs(prod.path))
        if labels is not None:
            np.save(d / f"{f}.tiles.npy", _reproject_labels(labels, wcs, cfg.bin, grid_wcs, grid_shape))
        del data, out

    ir0, ir1, ic0, ic1 = inscribed_rect(valid, cfg.align.crop_min_coverage)
    if cfg.align.crop == "union" and (cfg.align.channel_groups or cfg.align.require_all):
        groups = [list(g) for g in cfg.align.channel_groups]
        if cfg.align.require_all:
            groups += [[f] for f in prods]
        groups += [[k] for k in finite if k.endswith(":nexp")]
        r0, r1, c0, c1 = channel_rect(finite, groups, inset_px=cfg.align.channel_inset_px / cfg.bin)
        del finite
        console.print(f"  кадр: на краях все группы каналов"
                      + (", все фильтры" if cfg.align.require_all else "")
                      + (f", коротких волн ≥ {cfg.align.min_exposures_sw} экспозиций" if cfg.align.min_exposures_sw else "")
                      + f" — {c1 - c0}×{r1 - r0} px (общее поле {ic1 - ic0}×{ir1 - ir0})")
    elif cfg.align.crop == "union":
        rows, cols = np.flatnonzero(count.any(axis=1)), np.flatnonzero(count.any(axis=0))
        r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        console.print(f"  кадр по объединению покрытий: {c1 - c0}×{r1 - r0} px (общее поле {ic1 - ic0}×{ir1 - ir0})")
    else:
        rect = _target_rect(cfg, paths, prods[reference], grid_wcs, valid) if cfg.align.crop_mode == "target" else None
        r0, r1, c0, c1 = rect or (ir0, ir1, ic0, ic1)
        console.print(f"  обрезка до общего поля: {c1 - c0}×{r1 - r0} px из {grid_shape[1]}×{grid_shape[0]}")
    np.save(d / "coverage.npy", count[r0:r1, c0:c1].copy())
    for f in prods:
        for name in (f"{f}.npy", f"{f}.tiles.npy"):
            if (d / name).exists():
                np.save(d / name, np.load(d / name)[r0:r1, c0:c1].copy())
    crop_wcs = grid_wcs.slice((slice(r0, r1), slice(c0, c1)))
    if keep is not None:
        _destripe_preview(cfg, keep, crop_wcs, (r1 - r0, c1 - c0), d / "previews" / "destripe_check.png")

    meta = {
        "key": key,
        "filters": list(prods),
        "reference": reference,
        "shape": [r1 - r0, c1 - c0],
        "crop": [r0, r1, c0, c1],
        "mode": cfg.align.crop,
        # общее поле (все фильтры) в координатах кадра: в union — граница прежнего кадра
        "intersection": [int(max(ir0, r0) - r0), int(min(ir1, r1) - r0), int(max(ic0, c0) - c0), int(min(ic1, c1) - c0)],
        "pixel_scale_arcsec": round(float(np.mean(proj_plane_pixel_scales(crop_wcs))) * 3600, 5),
        "destripe": stripes,
        # Заголовок построчно, чтобы читать обратно через Header.fromstring(sep="\n").
        "wcs": "\n".join(card.image for card in crop_wcs.to_header(relax=True).cards),
    }
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _destripe_native(cfg: Config, data: np.ndarray, wcs: WCS,
                     labels: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    from skyforge.destripe import destripe, destripe_mask, step_metric

    scale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600
    mask = destripe_mask(data, cfg, scale)
    out, info = destripe(data, labels, mask, cfg.clean.destripe)
    c = cfg.clean.destripe
    info.update(rows_before=round(step_metric(data, mask, labels, c, 1), 6),
                rows_after=round(step_metric(out, mask, labels, c, 1), 6),
                cols_before=round(step_metric(data, mask, labels, c, 0), 6),
                cols_after=round(step_metric(out, mask, labels, c, 0), 6))
    info["before"], info["after"] = info["rows_before"], info["rows_after"]
    console.print(f"  destripe: |Δ медиан соседних строк| {info['rows_before']:.3g} → {info['rows_after']:.3g}, "
                  f"столбцов {info['cols_before']:.3g} → {info['cols_after']:.3g}")
    return out, info, {"before": data, "after": out, "mask": mask, "wcs": wcs, "scale": scale}


def _destripe_preview(cfg: Config, keep: dict[str, Any], grid_wcs: WCS, grid_shape: tuple[int, int],
                      path: Path) -> None:
    """Пустое небо 100% до и после destripe в исходной сетке фильтра и профиль медиан строк."""
    import matplotlib.pyplot as plt

    sky = next((c for c in resolve_crops(cfg.report.crops, grid_wcs, grid_shape[0], cfg.bin) if c.name == "sky"), None)
    if sky is None:
        console.print("  [yellow]destripe_check.png: в report.crops нет кропа sky — превью пропущено[/]")
        return
    rs, cs = _display_to_grid(sky, grid_shape, cfg.bin)
    ra, dec = grid_wcs.pixel_to_world_values((cs.start + cs.stop) / 2, (rs.start + rs.stop) / 2)
    nx, ny = keep["wcs"].world_to_pixel_values(ra, dec)
    grid_scale = float(np.mean(proj_plane_pixel_scales(grid_wcs))) * 3600
    side = max(16, int(round((rs.stop - rs.start) * grid_scale / keep["scale"])))
    h, w = keep["before"].shape
    r0 = int(np.clip(round(float(ny)) - side // 2, 0, max(h - side, 0)))
    c0 = int(np.clip(round(float(nx)) - side // 2, 0, max(w - side, 0)))
    sl = (slice(r0, r0 + side), slice(c0, c0 + side))
    before, after, mask = keep["before"][sl], keep["after"][sl], keep["mask"][sl]
    lo, hi = np.nanpercentile(before[~mask], [1, 99]) if (~mask).any() else (0, 1)
    norm = lambda a: np.clip((a - lo) / (hi - lo if hi > lo else 1), 0, 1)  # noqa: E731
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # строки целиком под маской — ожидаемо
        prof_b = np.nanmedian(np.where(mask, np.nan, before), axis=1)
        prof_a = np.nanmedian(np.where(mask, np.nan, after), axis=1)
    fig = plt.figure(figsize=(11, 9), facecolor="white")
    for i, (img, title) in enumerate(((before, "до destripe"), (after, "после destripe"))):
        ax = fig.add_subplot(2, 2, i + 1)
        ax.imshow(norm(img), cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(f"{keep['filter']}, пустое небо 100% — {title}")
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    rows = np.arange(r0, r0 + len(prof_b))
    ax.plot(rows, prof_b, color="#c0392b", lw=1, label=f"до (|Δ соседних строк| {keep['info']['rows_before']:.3g})")
    ax.plot(rows, prof_a, color="#2471a3", lw=1, label=f"после ({keep['info']['rows_after']:.3g})")
    ax.set_xlabel("строка детектора (исходная сетка)")
    ax.set_ylabel("медиана строки по пустому небу")
    ax.set_title("плавный ход фона остаётся, полосы — быстрые скачки от строки к строке — уходят")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _display_to_grid(crop: Any, shape: tuple[int, int], bin_: int) -> tuple[slice, slice]:
    from skyforge.preview import display_crop

    return display_crop(shape, crop.x, crop.y, crop.size, bin_)


def _target_rect(cfg: Config, paths: Paths, ref: Product, grid_wcs: WCS,
                 valid: np.ndarray) -> tuple[int, int, int, int] | None:
    if cfg.coords:
        ra, dec = (float(v) for v in cfg.coords.replace(",", " ").split()[:2])
    else:
        tc = target_coords(paths.abs(ref.path))
        if tc is None:
            console.print("[yellow]  crop_mode: target — нет координат цели (coords или заголовок), режу по краям[/]")
            return None
        ra, dec = tc
    x, y = grid_wcs.world_to_pixel_values(ra, dec)
    rect = largest_rect_with(valid, int(round(float(y))), int(round(float(x))))
    if rect is None:
        console.print("[yellow]  crop_mode: target — цель вне общего поля, режу по краям[/]")
    else:
        console.print(f"  обрезка вокруг цели ({ra:.4f}, {dec:.4f})")
    return rect


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
    count = np.sum([np.isfinite(np.asarray(a)) for a in arrays.values()], axis=0)
    _coverage_preview(count, len(arrays), meta.get("intersection"), out_dir / "coverage.png", pv.max_size)
    return {"checker_pair": [reference, other], "shifts_px": shifts, "mode": meta.get("mode", "intersection"),
            "coverage_percent": {str(k): round(float((count == k).mean() * 100), 2) for k in range(len(arrays) + 1)},
            "previews": ["align_check.png", "align_zoom.png", "coverage.png"]}


def _coverage_preview(count: np.ndarray, n: int, inter: list[int] | None, path: Path, max_size: int) -> None:
    """Число фильтров на пиксель; пунктир — общее поле всех фильтров (граница кадра в режиме intersection)."""
    import matplotlib.pyplot as plt

    f = max(1, int(np.ceil(max(count.shape) / max_size)))
    small = count[::f, ::f]
    fig, ax = plt.subplots(figsize=(8, 8 * small.shape[0] / small.shape[1] + 0.6))
    im = ax.imshow(small[::-1], cmap=plt.get_cmap("viridis", n + 1), vmin=-0.5, vmax=n + 0.5, interpolation="nearest")
    if inter:
        r0, r1, c0, c1 = (v / f for v in inter)
        h = small.shape[0]
        ax.plot([c0, c1, c1, c0, c0], [h - r0, h - r0, h - r1, h - r1, h - r0], "w--", lw=1.2)
    fig.colorbar(im, ax=ax, ticks=range(n + 1), fraction=0.04, label="фильтров на пиксель")
    ax.set_title("Покрытие: число фильтров на пиксель; пунктир — поле, где есть все фильтры")
    ax.axis("off")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
