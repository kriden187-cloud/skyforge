"""Своя мозаика из экспозиций уровня 2 (_cal.fits) — фаза 3f.

Зачем: в мозаике уровня 3 рисунок полос 1/f отдельных экспозиций уже смешан, и ступеньки
между областями CON не снимаются ни константой, ни плоскостью (фазы 3d–3e). Здесь полосы
убираются там, где они возникают, — в системе детектора каждой экспозиции, — а уровни
экспозиций сводятся до сборки.

1. Маска источников (туманность и звёзды) строится по i2d того же фильтра и проецируется
   в кадр экспозиции по WCS; к ней — плохие пиксели из DQ и яркие пиксели (лучи).
2. destripe в кадре детектора: строки — по каждому усилителю (полосы по 512 столбцов),
   вычитается быстрая часть медиан (окно, как в 3d: ореол остаётся); сдвиг между усилителями —
   по ступеньке на их общей границе; затем столбцы.
3. Уровни: одна константа на экспозицию по свободному небу в перекрытиях (МНК по парам).
4. Сборка на сетке i2d уровня 3: взвешенное среднее (1/ERR²) с отбрасыванием выбросов по
   стеку (космические лучи, горячие пиксели); на выходе FITS той же формы, что i2d, с WHT,
   CON (биты экспозиций) и NEXP — всё после align не меняется.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_interp
from scipy import ndimage

from skyforge.config import Config, DestripeConfig, Level2Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import cache_previews, contact_sheet, quick_norm, shrink

BUILD_VERSION = 1
CHUNK_ROWS = 256           # строк сетки за один проход стека
PAIR_MIN_PIXELS = 500      # минимум общего свободного неба для пары перекрывающихся экспозиций
REVEAL = (1.0, 90.0, 0.02)  # жёсткая растяжка превью: полосы — доли σ


# ------------------------------------------------------------------ чтение


def _wcs(h: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(h).celestial


def load_exposure(path: Path, dq_bad: int) -> tuple[np.ndarray, np.ndarray, WCS, fits.Header]:
    """SCI и ERR как float32; плохие по DQ, нечисловые и с ERR ≤ 0 — NaN."""
    # без memmap: DQ хранится как uint32 со сдвигом BZERO, такой слой memmap не открывает
    with fits.open(path, memmap=False, mode="readonly") as h:
        sci = np.array(h["SCI"].data, dtype=np.float32)
        err = np.array(h["ERR"].data, dtype=np.float32)
        dq = np.asarray(h["DQ"].data) if "DQ" in h else None
        wcs = _wcs(h["SCI"].header)
        prim = h[0].header.copy()
    if dq is None:
        raise SkyforgeError(f"{path.name}: нет слоя DQ — без него сборка не отличит плохие пиксели")
    bad = ((dq & dq_bad) != 0) | ~np.isfinite(sci) | ~np.isfinite(err) | (err <= 0)
    sci[bad] = np.nan
    err[bad] = np.nan
    return sci, err, wcs, prim


def source_mask_grid(i2d: np.ndarray, scale: float, cfg: Config) -> np.ndarray:
    """Туманность и звёзды на сетке i2d — те же маски, что в clean."""
    from skyforge.background import source_mask
    from skyforge.stars import star_mask

    return source_mask(i2d, cfg.background, scale, for_tiles=True) | star_mask(i2d, cfg.stars, scale)[0]


def mask_to_frame(mask: np.ndarray, grid_wcs: WCS, frame_wcs: WCS, shape: tuple[int, int], grow: int) -> np.ndarray:
    m, _ = reproject_interp((mask.astype(np.float32), grid_wcs), frame_wcs, shape_out=shape, order="bilinear")
    m = np.nan_to_num(m) > 0
    return ndimage.binary_dilation(m, iterations=grow) if grow else m


# ------------------------------------------------------------------ destripe в кадре детектора


def _bright(a: np.ndarray, free: np.ndarray, k: float) -> np.ndarray:
    v = a[free]
    if v.size < 100:
        return np.zeros(a.shape, dtype=bool)
    med, _, std = sigma_clipped_stats(v[:: max(1, v.size // 1_000_000)], sigma=3.0)
    return ndimage.binary_dilation(np.nan_to_num(a, nan=-np.inf) > med + k * std, iterations=2)


def _lines(vals: np.ndarray, c: Level2Config, axis: int) -> np.ndarray | None:
    from skyforge.destripe import _line_offsets

    dc = DestripeConfig(min_pixels=c.min_pixels, min_free_fraction=c.min_free_fraction)
    extent = np.full(vals.shape[0] if axis == 1 else vals.shape[1], vals.shape[axis])  # длина каждой линии
    off = _line_offsets(vals, dc, axis, extent)
    if off is None:
        return None
    if c.window:
        off = off - ndimage.median_filter(off, size=min(c.window, off.size) | 1, mode="reflect")
    return off


def amp_steps(a: np.ndarray, free: np.ndarray, n_amp: int, band: int = 8) -> np.ndarray:
    """Сдвиги усилителей по ступенькам на их общих границах (накопительно, среднее — ноль).
    Медиана полосы целиком смешала бы сдвиг усилителя с ходом ореола по кадру."""
    w = a.shape[1] // n_amp
    lev = [0.0]
    for k in range(1, n_amp):
        b = k * w
        left = np.where(free[:, b - band:b], a[:, b - band:b], np.nan)
        right = np.where(free[:, b:b + band], a[:, b:b + band], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmedian(right, axis=1) - np.nanmedian(left, axis=1)
        d = d[np.isfinite(d)]
        lev.append(lev[-1] + (float(np.median(d)) if d.size > 20 else 0.0))
    lev = np.asarray(lev, dtype=np.float32)
    return lev - lev.mean()


def destripe_frame(a: np.ndarray, mask: np.ndarray, c: Level2Config) -> tuple[np.ndarray, dict[str, float]]:
    from skyforge.destripe import step_metric

    out = np.array(a, dtype=np.float32, copy=True)
    free = np.isfinite(out) & ~mask
    free &= ~_bright(out, free, c.pixel_sigma)
    h, w = out.shape
    n = max(1, c.amplifiers)
    aw = w // n
    amps = np.minimum(np.arange(w) // aw, n - 1)
    labels = np.broadcast_to(amps[None, :], out.shape).astype(np.int32)
    dc = DestripeConfig(min_pixels=c.min_pixels, min_free_fraction=c.min_free_fraction)
    before = step_metric(out, ~free, labels, dc, 1)
    # строки по усилителям
    for k in range(n):
        sl = slice(k * aw, w if k == n - 1 else (k + 1) * aw)
        off = _lines(np.where(free[:, sl], out[:, sl], np.nan), c, 1)
        if off is not None:
            out[:, sl] -= off[:, None]
    # сдвиги усилителей — по ступенькам на границах
    lev = amp_steps(out, free, n)
    for k in range(n):
        out[:, k * aw: w if k == n - 1 else (k + 1) * aw] -= lev[k]
    # столбцы
    if c.columns:
        off = _lines(np.where(free, out, np.nan), c, 0)
        if off is not None:
            out -= off[None, :]
    after = step_metric(out, ~free, labels, dc, 1)
    return out, {"rows_before": round(before, 6), "rows_after": round(after, 6),
                 "amp_offsets": [round(float(v), 5) for v in lev], "free_percent": round(float(free.mean() * 100), 1)}


# ------------------------------------------------------------------ проекция и стек


def _bbox_on_grid(frame_wcs: WCS, shape: tuple[int, int], grid_wcs: WCS, grid_shape: tuple[int, int],
                  pad: int = 4) -> tuple[int, int, int, int] | None:
    h, w = shape
    t = np.linspace(0, 1, 9)
    xs = np.concatenate([t * (w - 1), np.full(9, w - 1), t[::-1] * (w - 1), np.zeros(9)])
    ys = np.concatenate([np.zeros(9), t * (h - 1), np.full(9, h - 1), t[::-1] * (h - 1)])
    ra, dec = frame_wcs.all_pix2world(xs, ys, 0)
    gx, gy = grid_wcs.all_world2pix(ra, dec, 0)
    x0, x1 = int(max(0, math.floor(gx.min()) - pad)), int(min(grid_shape[1], math.ceil(gx.max()) + pad + 1))
    y0, y1 = int(max(0, math.floor(gy.min()) - pad)), int(min(grid_shape[0], math.ceil(gy.max()) + pad + 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return y0, y1, x0, x1


def project(a: np.ndarray, frame_wcs: WCS, grid_wcs: WCS, box: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = box
    sub = grid_wcs.slice((slice(y0, y1), slice(x0, x1)))
    out, _ = reproject_interp((a, frame_wcs), sub, shape_out=(y1 - y0, x1 - x0), order="bilinear")
    return out.astype(np.float32)


def match_levels(exps: list[dict[str, Any]], free_grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Константы экспозиций c_i: c_i − c_j = медиана(e_i − e_j) по общему свободному небу (МНК, Σc = 0)."""
    n = len(exps)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = exps[i]["box"], exps[j]["box"]
            y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
            if y1 <= y0 or x1 <= x0:
                continue
            ei = exps[i]["data"][y0 - a[0]:y1 - a[0], x0 - a[2]:x1 - a[2]]
            ej = exps[j]["data"][y0 - b[0]:y1 - b[0], x0 - b[2]:x1 - b[2]]
            ok = np.isfinite(ei) & np.isfinite(ej) & free_grid[y0:y1, x0:x1]
            if ok.sum() < PAIR_MIN_PIXELS:
                continue
            pairs.append({"i": i, "j": j, "d": float(np.median(ei[ok] - ej[ok])), "n": int(ok.sum()),
                          "box": (y0, y1, x0, x1)})
    c = np.zeros(n)
    if pairs:
        A = np.zeros((len(pairs) + 1, n))
        b = np.zeros(len(pairs) + 1)
        for k, p in enumerate(pairs):
            w = math.sqrt(p["n"])
            A[k, p["i"]], A[k, p["j"]], b[k] = w, -w, w * p["d"]
        A[-1, :] = max(math.sqrt(p["n"]) for p in pairs)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        for p in pairs:
            p["residual"] = p["d"] - (c[p["i"]] - c[p["j"]])
    return c, pairs


def stack(exps: list[dict[str, Any]], shape: tuple[int, int], c: Level2Config) -> dict[str, np.ndarray]:
    """Взвешенное среднее с отбрасыванием выбросов по стеку, кусками строк сетки."""
    H, W = shape
    sci = np.full(shape, np.nan, dtype=np.float32)
    wht = np.zeros(shape, dtype=np.float32)
    nexp = np.zeros(shape, dtype=np.int16)
    con = np.zeros(shape, dtype=np.int64)
    rejected = 0
    for r0 in range(0, H, CHUNK_ROWS):
        r1 = min(H, r0 + CHUNK_ROWS)
        idx = [k for k, e in enumerate(exps) if e["box"][0] < r1 and e["box"][1] > r0]
        if not idx:
            continue
        S = np.full((len(idx), r1 - r0, W), np.nan, dtype=np.float32)
        E = np.full_like(S, np.nan)
        for m, k in enumerate(idx):
            e = exps[k]
            y0, y1, x0, x1 = e["box"]
            ya, yb = max(y0, r0), min(y1, r1)
            S[m, ya - r0:yb - r0, x0:x1] = e["data"][ya - y0:yb - y0]
            E[m, ya - r0:yb - r0, x0:x1] = e["err"][ya - y0:yb - y0]
        S[~np.isfinite(E)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(S, axis=0)
        gy, gx = np.gradient(np.nan_to_num(med))
        grad = np.hypot(gy, gx)
        cnt = np.isfinite(S).sum(axis=0)
        dev = np.abs(S - med[None])
        bad = (dev > c.clip_sigma * E + c.clip_gradient * grad[None]) & (cnt[None] >= 3)
        # двое на пиксель: медиана — их среднее, отбрасываем бо́льшее, если расхождение не по шуму
        two = cnt == 2
        if two.any():
            hi = np.nanargmax(np.where(np.isfinite(S), S, -np.inf), axis=0)
            diff = np.nanmax(S, axis=0) - np.nanmin(S, axis=0)
            e2 = np.sqrt(np.nansum(E**2, axis=0))
            drop = two & (diff > c.clip_sigma * e2 + c.clip_gradient * grad)
            yy, xx = np.nonzero(drop)
            bad[hi[yy, xx], yy, xx] = True
        S[bad] = np.nan
        rejected += int(bad.sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            w = np.where(np.isfinite(S), 1.0 / (E * E), 0.0).astype(np.float32)
        ws = w.sum(axis=0)
        num = np.nansum(np.where(np.isfinite(S), S * w, 0.0), axis=0)
        ok = ws > 0
        sci[r0:r1][ok] = (num[ok] / ws[ok]).astype(np.float32)
        wht[r0:r1] = ws
        good = np.isfinite(S)
        nexp[r0:r1] = good.sum(axis=0)
        bits = np.zeros((r1 - r0, W), dtype=np.int64)
        for m, k in enumerate(idx):
            bits |= np.where(good[m], np.int64(1) << np.int64(k), np.int64(0))
        con[r0:r1] = bits
    return {"sci": sci, "wht": wht, "nexp": nexp, "con": con, "rejected": rejected}


# ------------------------------------------------------------------ этап build


def build_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work_root / cfg.slug / "level2"


def _key(cfg: Config, files: list[Path], i2d: Path) -> dict[str, Any]:
    return {"version": BUILD_VERSION, "files": [[p.name, p.stat().st_size] for p in files],
            "i2d": [i2d.name, i2d.stat().st_size], "level2": cfg.level2.model_dump(),
            # только то, от чего зависят маски источников (иначе любое переключение в background пересобирает мозаику)
            "masks": {k: getattr(cfg.background, k) for k in ("mask_sigma", "mask_smooth_arcsec", "tile_mask_dilate_arcsec")}
            | {"stars": cfg.stars.model_dump()}}


def built_products(cfg: Config, paths: Paths) -> dict[str, Path]:
    d = build_dir(cfg, paths)
    meta = d / "meta.json"
    if not meta.exists():
        return {}
    m = json.loads(meta.read_text(encoding="utf-8"))
    return {f: d / v["file"] for f, v in m.get("filters", {}).items() if (d / v["file"]).exists()}


def build(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    from skyforge.fetch import level2_files, require_products

    if cfg.source != "level2":
        return {"skipped": "source: level3"}
    l3 = require_products(cfg, paths, level3=True)
    l2 = level2_files(cfg, paths)
    missing = [f for f in l3 if f not in l2]
    if missing:
        raise SkyforgeError(f"нет экспозиций уровня 2 для {missing}: skyforge fetch (source: level2)")
    d = build_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"filters": {}}
    info: dict[str, Any] = {"filters": {}}
    previews: dict[str, list] = {"destripe": [], "overlaps": [], "stars": []}
    for f, prod in l3.items():
        i2d_path = paths.abs(prod.path)
        key = _key(cfg, l2[f], i2d_path)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
        old = meta["filters"].get(f)
        if old and old.get("digest") == digest and (d / old["file"]).exists():
            console.print(f"  {f}: сборка из кеша ({old['file']})")
            info["filters"][f] = {**old, "cached": True}
            continue
        if old and (d / old["file"]).exists():
            (d / old["file"]).unlink()
        fi, pv = _build_filter(cfg, f, i2d_path, l2[f], d / f"{f}_l2_{digest}.fits")
        fi.update(digest=digest, file=f"{f}_l2_{digest}.fits")
        meta["filters"][f] = fi
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        info["filters"][f] = fi
        for k in previews:
            previews[k].append(pv[k])
    names = []
    if previews["destripe"]:
        imgs, labels = [], []
        for item in previews["destripe"]:
            imgs += item[0]
            labels += item[1]
        contact_sheet(imgs, labels, d / "previews" / "l2_destripe.png", cols=2, tile_inches=4, interpolation="nearest",
                      title="Одна экспозиция на фильтр, пустое небо 100% в кадре детектора: до и после destripe")
        imgs, labels = [], []
        for item in previews["overlaps"]:
            imgs += item[0]
            labels += item[1]
        contact_sheet(imgs, labels, d / "previews" / "l2_overlaps.png", cols=2, tile_inches=4,
                      title="Число экспозиций на пиксель и остаточный скачок уровней в перекрытиях после сведения")
        imgs, labels = [], []
        for item in previews["stars"]:
            imgs += item[0]
            labels += item[1]
        if imgs:
            contact_sheet(imgs, labels, d / "previews" / "l2_stars.png", cols=5, tile_inches=2.0, interpolation="nearest",
                          title="Изолированные звёзды: i2d STScI (верх) и наша сборка (низ) — FWHM и поток")
    for n in ("l2_destripe.png", "l2_overlaps.png", "l2_stars.png"):
        if (d / "previews" / n).exists():
            names.append(n)
    cache_previews(names, d / "previews", out_dir)
    info["previews"] = names
    return info


def _build_filter(cfg: Config, f: str, i2d_path: Path, files: list[Path], dest: Path) -> tuple[dict[str, Any], dict]:
    c = cfg.level2
    with fits.open(i2d_path, memmap=True) as h:
        i2d = np.array(h["SCI"].data, dtype=np.float32)
        if "WHT" in h:
            i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
        grid_hdr, prim_hdr = h["SCI"].header.copy(), h[0].header.copy()
    grid_wcs = _wcs(grid_hdr)
    shape = i2d.shape
    scale = float(np.mean(proj_plane_pixel_scales(grid_wcs))) * 3600
    src_grid = source_mask_grid(i2d, scale, cfg)
    free_grid = np.isfinite(i2d) & ~src_grid
    console.print(f"  {f}: {len(files)} экспозиций → сетка i2d {shape[1]}×{shape[0]}")
    exps, stripes = [], []
    demo = None
    for k, p in enumerate(files):
        sci, err, fw, _ = load_exposure(p, c.dq_bad)
        mask = mask_to_frame(src_grid, grid_wcs, fw, sci.shape, c.mask_grow_px)
        clean, st = destripe_frame(sci, mask, c)
        stripes.append(st)
        if st["free_percent"] < 5:
            console.print(f"    [yellow]{p.name}: свободного неба {st['free_percent']}% — полосы почти не измерить[/]")
        if demo is None or st["free_percent"] > demo[2]["free_percent"]:
            demo = (sci, clean, st, mask, p.name)
        box = _bbox_on_grid(fw, sci.shape, grid_wcs, shape)
        if box is None:
            continue
        exps.append({"name": p.name, "box": box, "data": project(clean, fw, grid_wcs, box),
                     "err": project(err, fw, grid_wcs, box)})
    levels, pairs = (np.zeros(len(exps)), [])
    if c.match_levels and len(exps) > 1:
        levels, pairs = match_levels(exps, free_grid)
        for e, lv in zip(exps, levels):
            e["data"] -= np.float32(lv)
    res = stack(exps, shape, c)
    # FITS той же формы, что i2d: SCI с WCS i2d, WHT, CON (биты экспозиций), NEXP
    hdus = [fits.PrimaryHDU(header=prim_hdr), fits.ImageHDU(res["sci"], header=grid_hdr, name="SCI"),
            fits.ImageHDU(res["wht"], name="WHT"), fits.ImageHDU(res["con"][None], name="CON"),
            fits.ImageHDU(res["nexp"], name="NEXP")]
    hdus[0].header["SKYFORGE"] = ("level2", "mosaic built from level-2 _cal.fits")  # FITS: только ASCII
    dest.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList(hdus).writeto(dest, overwrite=True)

    resid = [abs(p["residual"]) for p in pairs if "residual" in p]
    sky_v = i2d[free_grid]
    noise = float(sigma_clipped_stats(sky_v[:: max(1, sky_v.size // 1_000_000)], sigma=3.0)[2]) if sky_v.size else 0.0
    stars = compare_stars(i2d, res["sci"], exps=exps)
    rb = float(np.median([s["rows_before"] for s in stripes]))
    ra = float(np.median([s["rows_after"] for s in stripes]))
    fi = {"exposures": len(exps), "rows_before": round(rb, 6), "rows_after": round(ra, 6),
          "pairs": len(pairs), "level_residual_sigma": round(float(np.median(resid)) / noise, 4) if resid and noise else None,
          "rejected_pixels": res["rejected"], "nexp_median": float(np.median(res["nexp"][res["nexp"] > 0])),
          "stars": stars["summary"]}
    console.print(f"    destripe |Δ строк| (медиана по экспозициям) {rb:.3g} → {ra:.3g}; пар перекрытий {len(pairs)}, "
                  f"остаток уровней {fi['level_residual_sigma']}σ; выбросов отброшено {res['rejected']}; "
                  f"звёзды: FWHM {stars['summary'].get('fwhm_ratio')}, поток {stars['summary'].get('flux_ratio')}")
    pv = {"destripe": _demo_images(f, demo), "overlaps": _overlap_images(f, res["nexp"], pairs, noise, shape),
          "stars": stars["images"]}
    return fi, pv


def _demo_images(f: str, demo: tuple) -> tuple[list, list]:
    sci, clean, st, mask, name = demo
    h, w = sci.shape
    s = 512
    free = (~mask & np.isfinite(sci)).astype(np.float32)
    dens = ndimage.uniform_filter(free, s)
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    y0, x0 = int(np.clip(cy - s // 2, 0, h - s)), int(np.clip(cx - s // 2, 0, w - s))
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    a = quick_norm(sci[sl], *REVEAL)
    b = quick_norm(clean[sl], *REVEAL)
    return [a, b], [f"{f} {name[:26]}: до", f"после (|Δ строк| {st['rows_before']:.3g} → {st['rows_after']:.3g})"]


def _overlap_images(f: str, nexp: np.ndarray, pairs: list, noise: float, shape) -> tuple[list, list]:
    heat = np.zeros(shape, dtype=np.float32)
    for p in pairs:
        if "residual" in p:
            y0, y1, x0, x1 = p["box"]
            heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], abs(p["residual"]) / noise if noise else 0)
    heat[nexp < 2] = 0
    n = shrink(nexp.astype(np.float32), 400)
    return ([np.clip(n / max(float(n.max()), 1), 0, 1), np.clip(shrink(heat, 400) / 0.2, 0, 1)],
            [f"{f}: экспозиций на пиксель (до {int(nexp.max())})", f"{f}: |остаток| уровней, белое = 0.2σ"])


# ------------------------------------------------------------------ звёзды: FWHM и поток


def _fwhm_flux(img: np.ndarray, y: int, x: int, r_ap: float) -> tuple[float, float]:
    from scipy.optimize import curve_fit

    s = 12
    cut = img[y - s:y + s + 1, x - s:x + s + 1].astype(np.float64)
    if cut.shape != (2 * s + 1, 2 * s + 1) or not np.isfinite(cut).all():
        return float("nan"), float("nan")
    yy, xx = np.indices(cut.shape)
    r = np.hypot(yy - s, xx - s)
    bg = float(np.median(cut[r > s - 3]))
    z = cut - bg

    def g(c, a, x0, y0, sg):
        return a * np.exp(-((c[1] - x0) ** 2 + (c[0] - y0) ** 2) / (2 * sg**2))

    try:
        p, _ = curve_fit(g, (yy.ravel(), xx.ravel()), z.ravel(), p0=(z.max(), s, s, 1.5), maxfev=4000)
    except Exception:
        return float("nan"), float("nan")
    return float(2.3548 * abs(p[3])), float(z[r <= r_ap].sum())


def _single_flux(exps: list[dict[str, Any]], y: int, x: int, r_ap: float) -> float:
    """Медиана потока звезды по отдельным экспозициям (уже на сетке) — опора, не зависящая ни от одной сборки."""
    vals = []
    for e in exps:
        y0, y1, x0, x1 = e["box"]
        if y0 + 13 <= y < y1 - 13 and x0 + 13 <= x < x1 - 13:
            v = _fwhm_flux(e["data"], y - y0, x - x0, r_ap)[1]
            if np.isfinite(v):
                vals.append(v)
    return float(np.median(vals)) if vals else float("nan")


def compare_stars(ref: np.ndarray, ours: np.ndarray, n: int = 5, exps: list | None = None) -> dict[str, Any]:
    """Пять изолированных ненасыщенных звёзд на пустом небе: FWHM и поток в i2d, в нашей сборке и
    (если даны exps) медианный поток по отдельным экспозициям."""
    a = np.nan_to_num(ref)
    sm = ndimage.gaussian_filter(a, 1.0)
    pk = np.argwhere((sm == ndimage.maximum_filter(sm, 41)) & (sm > np.percentile(sm, 99.9)))
    med, _, std = sigma_clipped_stats(a[:: 7, :: 7], sigma=3.0)
    cand = []
    for y, x in pk:
        if not (30 <= y < a.shape[0] - 30 and 30 <= x < a.shape[1] - 30):
            continue
        win = ref[y - 15:y + 16, x - 15:x + 16]
        if not np.isfinite(win).all() or not np.isfinite(ours[y - 15:y + 16, x - 15:x + 16]).all():
            continue  # насыщенное ядро или край поля
        peak = a[y, x] - med
        if peak < 50 * std:
            continue
        # изолированность: в радиусе 15 px нет другого пика ярче четверти этого
        others = ((sm[y - 15:y + 16, x - 15:x + 16] == ndimage.maximum_filter(sm, 5)[y - 15:y + 16, x - 15:x + 16])
                  & (sm[y - 15:y + 16, x - 15:x + 16] - med > 0.25 * peak)).sum()
        if others > 1:
            continue
        yy, xx = np.indices(win.shape)
        ring = np.hypot(yy - 15, xx - 15) > 10
        if float(np.median(win[ring])) - med > 5 * std:
            continue  # на туманности, не на пустом небе
        cand.append((peak, int(y), int(x)))
    cand.sort(reverse=True)
    rows, imgs, labels = [], [], []
    for _, y, x in cand[: n * 3]:
        f1, fl1 = _fwhm_flux(ref, y, x, 6)
        if not np.isfinite(f1):
            continue
        f2, fl2 = _fwhm_flux(ours, y, x, max(6, 2 * f1))
        fl1 = _fwhm_flux(ref, y, x, max(6, 2 * f1))[1]
        if not (np.isfinite(f2) and fl1 > 0):
            continue
        single = _single_flux(exps, y, x, max(6, 2 * f1)) if exps else float("nan")
        rows.append({"x": x, "y": y, "fwhm_i2d": round(f1, 3), "fwhm_ours": round(f2, 3),
                     "flux_i2d": fl1, "flux_ours": fl2, "flux_single": single})
        lo, hi = np.nanpercentile(ref[y - 12:y + 13, x - 12:x + 13], [5, 99.5])
        for img in (ref, ours):
            imgs.append(np.clip((np.nan_to_num(img[y - 12:y + 13, x - 12:x + 13]) - lo) / (hi - lo), 0, 1))
        labels += [f"i2d {f1:.2f}px", f"наша {f2:.2f}px, {fl2 / fl1 * 100 - 100:+.1f}%"]
        if len(rows) == n:
            break
    summ = {}
    if rows:
        summ = {"stars": len(rows),
                "fwhm_ratio": round(float(np.median([r["fwhm_ours"] / r["fwhm_i2d"] for r in rows])), 4),
                "flux_ratio": round(float(np.median([r["flux_ours"] / r["flux_i2d"] for r in rows])), 4),
                "fwhm_worst_percent": round(float(max(abs(r["fwhm_ours"] / r["fwhm_i2d"] - 1) for r in rows) * 100), 2),
                "flux_worst_percent": round(float(max(abs(r["flux_ours"] / r["flux_i2d"] - 1) for r in rows) * 100), 2)}
        sing = [r for r in rows if np.isfinite(r["flux_single"]) and r["flux_single"] > 0]
        if sing:
            summ["flux_ratio_vs_single"] = round(float(np.median([r["flux_ours"] / r["flux_single"] for r in sing])), 4)
            summ["i2d_ratio_vs_single"] = round(float(np.median([r["flux_i2d"] / r["flux_single"] for r in sing])), 4)
    # раскладка листа: n столбцов, i2d сверху, наша снизу (пустые клетки, если звёзд меньше)
    top, bottom = imgs[0::2], imgs[1::2]
    lt, lb = labels[0::2], labels[1::2]
    blank = np.zeros((25, 25), dtype=np.float32)
    while len(top) < n:
        top.append(blank); bottom.append(blank); lt.append(""); lb.append("")
    imgs, labels = top + bottom, lt + lb
    return {"summary": summ, "rows": rows, "images": (imgs, labels)}
