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
import re
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

BUILD_VERSION = 4
CHUNK_ROWS = 256           # строк сетки за один проход стека
PAIR_MIN_PIXELS = 500      # минимум общего свободного неба для пары перекрывающихся экспозиций
REVEAL = (1.0, 90.0, 0.02)  # жёсткая растяжка превью: полосы — доли σ


# ------------------------------------------------------------------ чтение


def _wcs(h: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        return WCS(h).celestial


def load_exposure(path: Path, dq_bad: int, weights: str = "err") -> tuple[np.ndarray, np.ndarray, WCS, fits.Header]:
    """SCI и «ошибка для весов» как float32; плохие по DQ, нечисловые и с ошибкой ≤ 0 — NaN.
    weights=rnoise — √VAR_RNOISE (шум считывания без пуассоновского шума источника), иначе ERR."""
    # без memmap: DQ хранится как uint32 со сдвигом BZERO, такой слой memmap не открывает
    with fits.open(path, memmap=False, mode="readonly") as h:
        sci = np.array(h["SCI"].data, dtype=np.float32)
        if weights == "rnoise" and "VAR_RNOISE" in h:
            err = np.sqrt(np.array(h["VAR_RNOISE"].data, dtype=np.float32))
        else:
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


# ------------------------------------------------------------------ астрометрия (аналог tweakreg)


def _gauss_centroid(img: np.ndarray, x: float, y: float, half: int = 5) -> tuple[float, float] | None:
    """Центроид звезды подгонкой 2D-Гаусса с фоном в окне (2·half+1)² вокруг (x, y). None — не вышло."""
    from scipy.optimize import curve_fit

    xi, yi = int(round(x)), int(round(y))
    if not (half <= xi < img.shape[1] - half and half <= yi < img.shape[0] - half):
        return None
    w = img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)
    if not np.isfinite(w).all():
        return None
    yy, xx = np.indices(w.shape)
    bg = float(np.median(np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]])))
    z = w - bg
    if z.max() <= 0:
        return None

    def g(c, a, x0, y0, sg, b):
        return a * np.exp(-((c[1] - x0) ** 2 + (c[0] - y0) ** 2) / (2 * sg**2)) + b

    try:
        p, _ = curve_fit(g, (yy.ravel(), xx.ravel()), z.ravel(), p0=(z.max(), half, half, 1.2, 0.0), maxfev=3000)
    except Exception:
        return None
    if not (0 <= p[1] <= 2 * half and 0 <= p[2] <= 2 * half and 0.3 < abs(p[3]) < half):
        return None
    return xi - half + float(p[1]), yi - half + float(p[2])


def reference_stars(i2d: np.ndarray, grid_wcs: WCS, limit: int = 400) -> np.ndarray:
    """Изолированные ненасыщенные звёзды на пустом небе i2d: (ra, dec) по центроидам на сетке i2d.
    Астрометрия i2d уровня 3 уже подстроена STScI — она и есть опора."""
    a = np.nan_to_num(i2d)
    sm = ndimage.gaussian_filter(a, 1.0)
    med, _, std = sigma_clipped_stats(a[::7, ::7], sigma=3.0)
    peaks = np.argwhere((sm == ndimage.maximum_filter(sm, 21)) & (sm - med > 20 * std))
    loc_max = sm == ndimage.maximum_filter(sm, 5)
    out = []
    order = np.argsort(-sm[peaks[:, 0], peaks[:, 1]])
    for y, x in peaks[order]:
        if not (15 <= y < a.shape[0] - 15 and 15 <= x < a.shape[1] - 15):
            continue
        win = i2d[y - 12:y + 13, x - 12:x + 13]
        if not np.isfinite(win).all():
            continue  # насыщенное ядро или край поля
        yy, xx = np.indices(win.shape)
        if float(np.median(win[np.hypot(yy - 12, xx - 12) > 9])) - med > 5 * std:
            continue  # на туманности
        peak = sm[y, x] - med
        neigh = loc_max[y - 12:y + 13, x - 12:x + 13] & (sm[y - 12:y + 13, x - 12:x + 13] - med > 0.1 * peak)
        if neigh.sum() > 1:
            continue  # соседка ярче 10% пика ближе 12 px
        c = _gauss_centroid(i2d, float(x), float(y), 5)
        if c is None:
            continue
        ra, dec = grid_wcs.all_pix2world([[c[0], c[1]]], 0)[0]
        out.append((ra, dec))
        if len(out) >= limit:
            break
    return np.asarray(out, dtype=np.float64).reshape(-1, 2)


def _rotated(wcs: WCS, shift: np.ndarray, theta: float) -> WCS:
    """WCS, у которой пиксель m = crpix + R(θ)(p − crpix) + shift смотрит туда, куда старая смотрела из p.
    CRPIX' = CRPIX + shift, CD' = CD·R(θ)⁻¹, SIP — с новым CRPIX (поворот ~0.01° меняет SIP пренебрежимо)."""
    from astropy.wcs import Sip

    w = wcs.deepcopy()
    crpix = np.asarray(w.wcs.crpix, dtype=np.float64) + shift
    ct, st = math.cos(theta), math.sin(theta)
    rinv = np.array([[ct, st], [-st, ct]])
    cd = w.wcs.cd if w.wcs.has_cd() else np.diag(w.wcs.cdelt) @ w.wcs.get_pc()
    w.wcs.crpix = crpix
    w.wcs.cd = np.asarray(cd) @ rinv
    if w.sip is not None:
        s = w.sip
        w.sip = Sip(s.a, s.b, s.ap, s.bp, crpix)
    w.wcs.set()
    return w


def fit_tweak(pred: np.ndarray, meas: np.ndarray, crpix: np.ndarray) -> dict[str, Any]:
    """Сдвиг и, отдельно, сдвиг + поворот вокруг CRPIX, переводящие предсказанные положения в измеренные.
    Робастно: точки дальше 3·MAD от сдвига отбрасываются (до 5 проходов). Оба решения — на одних звёздах."""
    d = meas - pred
    keep = np.ones(len(d), dtype=bool)
    shift = np.median(d, axis=0)
    for _ in range(5):
        shift = np.mean(d[keep], axis=0)
        r = np.hypot(*(d - shift).T)
        mad = 1.4826 * float(np.median(r[keep])) if keep.sum() else 0.0
        new = r <= max(3 * mad, 0.05)
        if new.sum() < 3 or np.array_equal(new, keep):
            break
        keep = new
    n = int(keep.sum())
    rms = lambda v: float(np.sqrt(np.mean(np.sum(v**2, axis=1))))  # noqa: E731
    out = {"used": n, "rms_before": rms(d[keep]), "shift": shift, "rms_shift": rms(d[keep] - shift),
           "rot_shift": shift, "theta": 0.0, "rms_rot": rms(d[keep] - shift)}
    if n >= 4:
        # m − p ≈ s + θ·[−(py − cy), (px − cx)] при малом θ
        q = pred[keep] - crpix
        A = np.zeros((2 * n, 3))
        A[0::2, 0], A[1::2, 1] = 1, 1
        A[0::2, 2], A[1::2, 2] = -q[:, 1], q[:, 0]
        sol, *_ = np.linalg.lstsq(A, d[keep].ravel(), rcond=None)
        out["rot_shift"], out["theta"] = sol[:2], float(sol[2])
        out["rms_rot"] = rms(d[keep] - (sol[:2] + sol[2] * np.column_stack([-q[:, 1], q[:, 0]])))
    return out


def measure_tweak(sci: np.ndarray, wcs: WCS, stars_world: np.ndarray, c: Level2Config) -> dict[str, Any]:
    """Центроиды опорных звёзд в экспозиции и подгонка; WCS не меняется (решение — после, по группам)."""
    info: dict[str, Any] = {"stars": 0}
    if stars_world.size == 0:
        return info
    h, w = sci.shape
    # фаза 8 (OMC3-N, мозаика 14502×6332): звёзды со всей мозаики далеко за детектором — обращение дисторсии SIP для них
    # расходится (NoConvergence). Сначала отбор линейной частью WCS с запасом 100 px, точная формула — только для них
    lx, ly = wcs.wcs_world2pix(stars_world[:, 0], stars_world[:, 1], 0)
    near = (lx > -100) & (lx < w + 100) & (ly > -100) & (ly < h + 100)
    stars_world = stars_world[near]
    if stars_world.size == 0:
        return info
    px, py = wcs.all_world2pix(stars_world[:, 0], stars_world[:, 1], 0)
    inside = (px > 8) & (px < w - 9) & (py > 8) & (py < h - 9)
    pred, meas = [], []
    for i in np.flatnonzero(inside):
        m = _gauss_centroid(sci, px[i], py[i], 5)
        if m is not None and math.hypot(m[0] - px[i], m[1] - py[i]) < 2.0:
            pred.append((px[i], py[i]))
            meas.append(m)
        if len(pred) >= c.tweak_max_stars:
            break
    info["stars"] = len(pred)
    if len(pred) < c.tweak_min_stars:
        return info
    pred_a, meas_a = np.asarray(pred), np.asarray(meas)
    fit = fit_tweak(pred_a, meas_a, np.asarray(wcs.wcs.crpix) - 1)
    use_rot = fit["used"] >= c.tweak_rot_min_stars and abs(math.degrees(fit["theta"])) >= c.tweak_rot_min_deg
    shift = fit["rot_shift"] if use_rot else fit["shift"]
    theta = fit["theta"] if use_rot else 0.0
    rms_after = fit["rms_rot"] if use_rot else fit["rms_shift"]
    info.update(used=fit["used"], dx=float(shift[0]), dy=float(shift[1]), theta=theta, rot_deg=math.degrees(theta),
                rms_before=fit["rms_before"], rms_after=rms_after,
                precision=rms_after / math.sqrt(fit["used"]), pred=pred_a, meas=meas_a)
    return info


def decide_tweaks(infos: list[dict[str, Any]], names: list[str], c: Level2Config) -> None:
    """Решение по правилам 3g: что исправлять, чем и что исключить. Пишет в infos поля
    status, apply (dx, dy, θ), exclude."""
    det = [re.search(r"_(nrc[ab]\w*?)_cal", n) for n in names]
    det = [m.group(1) if m else "" for m in det]
    group = [re.sub(r"_nrc[ab]\w*?_cal\.fits$", "", n) for n in names]
    good = [("precision" in i) and i["precision"] < c.tweak_max_precision for i in infos]
    for i, d, g_ok in zip(infos, det, good):
        i["exclude"] = False
        i["apply"] = None
        if d.endswith("long"):  # длинные волны: соседей нет
            if g_ok and i["rms_after"] <= c.tweak_lw_max_residual:
                i["apply"], i["status"] = (i["dx"], i["dy"], i["theta"]), "поворот" if i["theta"] else "сдвиг"
            else:
                i["status"] = "не исправлено"
                if i.get("rms_before", 0) > c.tweak_exclude_above:
                    i["exclude"], i["status"] = True, "исключена"
    sw = [k for k, d in enumerate(det) if not d.endswith("long")]
    for g in sorted({group[k] for k in sw}):
        members = [k for k in sw if group[k] == g]
        ok = [k for k in members if good[k]]
        if not ok:
            for k in members:
                infos[k]["status"] = "не исправлено (в группе нет точных)"
            continue
        med = np.median([[infos[k]["dx"], infos[k]["dy"]] for k in ok], axis=0)
        for k in members:
            i = infos[k]
            if good[k] and math.hypot(i["dx"] - med[0], i["dy"] - med[1]) <= c.tweak_group_tol:
                i["apply"], i["status"] = (i["dx"], i["dy"], i["theta"]), "поворот" if i["theta"] else "сдвиг"
            else:
                off = math.hypot(i["dx"] - med[0], i["dy"] - med[1]) if "dx" in i else None
                i["apply"] = (float(med[0]), float(med[1]), 0.0)
                i["status"] = (f"медиана группы (отличался на {off:.3f} px)" if off is not None and good[k]
                               else "медиана группы (своя подгонка неточна)")
    for i in infos:
        i.setdefault("status", "мало звёзд")
        i.setdefault("apply", None)
        i.setdefault("exclude", False)


def apply_tweak(wcs: WCS, info: dict[str, Any]) -> tuple[WCS, list[float], list[float]]:
    """Внести поправку; вернуть остатки по звёздам до и после (для гистограммы)."""
    if not info.get("apply"):
        return wcs, [], []
    dx, dy, th = info["apply"]
    new = _rotated(wcs, np.array([dx, dy]), th)
    before, after = [], []
    if "pred" in info:
        meas, pred = info["meas"], info["pred"]
        # предсказание новой WCS для тех же звёзд: p → crpix + R(p − crpix) + s
        crpix = np.asarray(wcs.wcs.crpix) - 1
        q = pred - crpix
        ct, st = math.cos(th), math.sin(th)
        newp = crpix + np.column_stack([ct * q[:, 0] - st * q[:, 1], st * q[:, 0] + ct * q[:, 1]]) + np.array([dx, dy])
        before = [round(float(v), 4) for v in np.hypot(*(meas - pred).T)]
        after = [round(float(v), 4) for v in np.hypot(*(meas - newp).T)]
    return new, before, after


def tweak_exposure(sci: np.ndarray, wcs: WCS, stars_world: np.ndarray, c: Level2Config) -> tuple[WCS, dict[str, Any]]:
    """Одна экспозиция без группы (тесты, отдельные кадры): правила как для длинных волн."""
    info = measure_tweak(sci, wcs, stars_world, c)
    decide_tweaks([info], ["single_nrcblong_cal.fits"], c)
    if info["status"] == "не исправлено" and info["stars"] < c.tweak_min_stars:
        info["status"] = "мало звёзд"
    new, b, a = apply_tweak(wcs, info)
    info["residuals_before"], info["residuals_after"] = b, a
    return new, info


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


def overlap_residual(exps: list[dict[str, Any]], sci: np.ndarray, nexp: np.ndarray, free_grid: np.ndarray,
                     block: int) -> tuple[float, np.ndarray, float]:
    """Остаток уровней в перекрытиях — одно определение и для числа, и для карты.

    Для каждой экспозиции: разность «экспозиция − стек» на свободном небе, медиана по блокам
    block×block (блок считается, если неба в нём не меньше половины); по пикселям, где
    экспозиций ≥ 2, — среднее |медиана блока| по всем экспозициям, в σ шума одной экспозиции.
    Возвращает (среднее по блокам, карта блоков в σ, σ одной экспозиции)."""
    H, W = sci.shape
    nb_y, nb_x = (H + block - 1) // block, (W + block - 1) // block
    tot = np.zeros((nb_y, nb_x))
    cnt = np.zeros((nb_y, nb_x))
    sig = []
    for e in exps:
        y0, y1, x0, x1 = e["box"]
        d = e["data"] - sci[y0:y1, x0:x1]
        ok = np.isfinite(d) & free_grid[y0:y1, x0:x1] & (nexp[y0:y1, x0:x1] >= 2)
        okk = free_grid[y0:y1, x0:x1] & np.isfinite(e["data"])
        if okk.sum() > 1000:
            from skyforge.background import pixel_noise

            sig.append(pixel_noise(e["data"], okk))  # шум пикселя одной экспозиции, без влияния градиентов
        for by in range(y0 // block, (y1 - 1) // block + 1):
            for bx in range(x0 // block, (x1 - 1) // block + 1):
                ys = slice(max(by * block, y0) - y0, min((by + 1) * block, y1) - y0)
                xs = slice(max(bx * block, x0) - x0, min((bx + 1) * block, x1) - x0)
                m = ok[ys, xs]
                if m.sum() >= 0.5 * block * block:
                    tot[by, bx] += abs(float(np.median(d[ys, xs][m])))
                    cnt[by, bx] += 1
    s1 = float(np.median(sig)) if sig else 0.0
    mp = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan) / (s1 if s1 else 1.0)
    metric = float(np.nanmean(mp)) if np.isfinite(mp).any() else 0.0
    return metric, mp, s1


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


def match_levels_per_amp(exps: list[dict[str, Any]], free_grid: np.ndarray) -> tuple[np.ndarray, list]:
    """Константа на экспозицию и усилитель: каждая экспозиция делится на полосы усилителей
    (карта номера усилителя спроецирована на сетку вместе с данными) и сводится как отдельная."""
    parts, owner = [], []
    for k, e in enumerate(exps):
        amp = e["amp"]
        for a in np.unique(amp[amp >= 0]):
            parts.append({"box": e["box"], "data": np.where(amp == a, e["data"], np.nan).astype(np.float32)})
            owner.append((k, int(a)))
    c, pairs = match_levels(parts, free_grid)
    for (k, a), lv in zip(owner, c):
        e = exps[k]
        e["data"] = np.where(e["amp"] == a, e["data"] - np.float32(lv), e["data"]).astype(np.float32)
    return np.asarray(c), pairs


def reproject_mask(path: Path, grid_wcs: WCS, shape: tuple[int, int]) -> np.ndarray:
    """Маска-файл (FITS с WCS, 1 — исключено) на сетке сборки; за пределами файла — не исключено."""
    with fits.open(path, memmap=False) as h:
        m = np.asarray(h[0].data, dtype=np.float32)
        w = _wcs(h[0].header)
    g, _ = reproject_interp((m, w), grid_wcs, shape_out=shape, order="nearest-neighbor")
    return np.nan_to_num(g) > 0.5


CURVE_GROUP_MIN = 200  # пикселей в полосе пары перекрытия для строки МНК
CURVE_RIDGE = 1e-6     # нижняя граница стягивания кривых — доля медианного веса данных
CURVE_MAX_PAIR_PX = 300_000  # пикселей перекрытия пары для медиан полос (детерминированное прореживание)


def _hat_index(y: np.ndarray, knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ломаная по узлам: значение в y = (1 − t)·g[k] + t·g[k + 1]. Возвращает (k, t)."""
    h = float(knots[1] - knots[0])
    yc = np.clip(y, knots[0], knots[-1])
    k = np.minimum((yc - knots[0]) // h, len(knots) - 2).astype(np.int64)
    return k, (yc - knots[k]) / h


def match_curves_y(exps: list[dict[str, Any]], free: np.ndarray, c: Level2Config,
                   sigma_pix: float, sigma_stack: float) -> tuple[list[np.ndarray], np.ndarray, dict[str, Any]]:
    """Кривая фона вдоль оси y детектора на экспозицию (ломаная, узлы через curve_knot_rows строк), один МНК без
    итераций вместе с константами на экспозицию и усилитель. Строки — медианы разности «экспозиция i − экспозиция j»
    в перекрытии по полосам curve_bin_px строк сетки и парам усилителей: c_ia − c_jb + g_i(y_i) − g_j(y_j) = медиана.
    Кривые стянуты к нулю (строки g/τ, τ = curve_prior·σ стека); средняя кривая экспозиции — ноль; средняя по фильтру
    Σ констант — ноль; решение перпендикулярно узорам, привязанным к небу (их данные не определяют, см. ниже):
    сведение убирает только расхождение экспозиций между собой и не вносит в мозаику своего узора.
    Нормальные уравнения копятся по строкам (в строке ≤ 6 ненулевых).
    Возвращает (узлы-значения кривой каждой экспозиции, узлы, сведения). Данные не меняет."""
    n = len(exps)
    na = max(int(max(int(e["amp"].max()) for e in exps)) + 1, 1)
    knots = [np.arange(0, e["ny"] - 1 + c.curve_knot_rows, c.curve_knot_rows, dtype=np.float64) for e in exps]
    off = np.cumsum([n * na] + [len(k) for k in knots])     # начало кривой экспозиции i в векторе неизвестных
    nu = int(off[-1])
    sig_d = math.sqrt(2) * max(sigma_pix, 1e-12)
    ATA = np.zeros((nu, nu))
    ATb = np.zeros(nu)
    rows = used = 0

    def add(idx: np.ndarray, val: np.ndarray, rhs: np.ndarray, w: np.ndarray) -> None:
        # idx, val: (r, m) — индексы и коэффициенты строк; накопление AᵀA и Aᵀb
        vw = val * w[:, None]
        for p in range(idx.shape[1]):
            for q in range(idx.shape[1]):
                np.add.at(ATA, (idx[:, p], idx[:, q]), vw[:, p] * vw[:, q])
            np.add.at(ATb, idx[:, p], vw[:, p] * rhs * w)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = exps[i]["box"], exps[j]["box"]
            y0, y1, x0, x1 = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
            if y1 <= y0 or x1 <= x0:
                continue
            si = (slice(y0 - a[0], y1 - a[0]), slice(x0 - a[2], x1 - a[2]))
            sj = (slice(y0 - b[0], y1 - b[0]), slice(x0 - b[2], x1 - b[2]))
            ei, ej = exps[i]["data"][si], exps[j]["data"][sj]
            ai, aj = exps[i]["amp"][si], exps[j]["amp"][sj]
            yi, yj = exps[i]["ydet"][si], exps[j]["ydet"][sj]
            ok = (np.isfinite(ei) & np.isfinite(ej) & free[y0:y1, x0:x1] & (ai >= 0) & (aj >= 0)
                  & np.isfinite(yi) & np.isfinite(yj))
            if ok.sum() < PAIR_MIN_PIXELS:
                continue
            ys, xs = np.nonzero(ok)
            step = max(1, ys.size // CURVE_MAX_PAIR_PX)
            ys, xs = ys[::step], xs[::step]
            key = (ai[ys, xs].astype(np.int64) * na + aj[ys, xs]) * 100000 + (ys + y0) // c.curve_bin_px
            d = (ei[ys, xs] - ej[ys, xs]).astype(np.float64)
            vi, vj = yi[ys, xs].astype(np.float64), yj[ys, xs].astype(np.float64)
            order = np.argsort(key, kind="stable")
            key, d, vi, vj = key[order], d[order], vi[order], vj[order]
            uniq, start, cnt = np.unique(key, return_index=True, return_counts=True)
            good = cnt >= CURVE_GROUP_MIN
            if not good.any():
                continue
            uniq, start, cnt = uniq[good], start[good], cnt[good]
            med = np.array([np.median(d[s0:s0 + m]) for s0, m in zip(start, cnt)])
            mi = np.add.reduceat(vi, start) / cnt if good.all() else np.array([vi[s0:s0 + m].mean() for s0, m in zip(start, cnt)])
            mj = np.add.reduceat(vj, start) / cnt if good.all() else np.array([vj[s0:s0 + m].mean() for s0, m in zip(start, cnt)])
            pa, pb = np.divmod(uniq // 100000, na)
            ki, ti = _hat_index(mi, knots[i])
            kj, tj = _hat_index(mj, knots[j])
            idx = np.column_stack([i * na + pa, j * na + pb, off[i] + ki, off[i] + ki + 1, off[j] + kj, off[j] + kj + 1])
            val = np.column_stack([np.ones(cnt.size), -np.ones(cnt.size), 1 - ti, ti, -(1 - tj), -tj])
            add(idx, val, med, np.sqrt(cnt) / (1.2533 * sig_d))
            rows += cnt.size
            used += int(cnt.sum()) * step
    info: dict[str, Any] = {"rows": rows, "pixels": used}
    if not rows:
        return [np.zeros(len(k)) for k in knots], knots[0], info
    big = 30 * float(np.sqrt(np.max(np.diag(ATA))))
    tau = max(c.curve_prior * sigma_stack, 1e-12)
    # Кривые стянуты к нулю. Узор, привязанный к небу (у каждой экспозиции — в её координатах детектора), в разностях
    # соседей сокращается: эти направления данные не определяют вовсе, их выбирает только стягивание (наименьшие
    # кривые — небо и ореол не трогаются). Поэтому стягивание не слабее CURVE_RIDGE доли веса данных: иначе система
    # вырождена численно и решение в этих направлениях произвольно.
    dc = np.diag(ATA)[n * na:].copy()
    ridge = max(1.0 / tau ** 2, CURVE_RIDGE * float(np.median(dc[dc > 0])) if (dc > 0).any() else 0.0)
    ATA[n * na:, n * na:] += np.eye(nu - n * na) * ridge
    ATA[:n * na, :n * na] += np.eye(n * na) * 1e-9 * float(np.mean(np.diag(ATA)))  # усилитель без перекрытий
    r = np.zeros(nu); r[:n * na] = big                               # Σ констант = 0
    ATA += np.outer(r, r)
    # Узор, привязанный к небу: p(Y) даёт экспозиции i кривую p(y + oy_i) (оси детектора вдоль сетки, oy_i — строка
    # сетки детекторной строки 0). Такие кривые в разностях соседей сокращаются — данные их не определяют. Решение
    # делается перпендикулярным им: для каждой «шапки» φ_m по строкам сетки (шаг узлов) Σ_i Σ_k φ_m(t_k + oy_i)·g_ik = 0.
    # Это выбор наименьших кривых в этих направлениях, но явно: со стягиванием 1e-6 численно не держалось, и кривые
    # вносили в мозаику полосы по строкам (F187N: размах профиля неба 0,95 → 3,45σ).
    H = free.shape[0]
    oy = []
    for e in exps:
        y0 = e["box"][0]
        okp = np.isfinite(e["ydet"])
        rr = np.nonzero(okp)[0]
        oy.append(float(np.median(rr + y0 - e["ydet"][okp])) if rr.size else 0.0)
    step_k = float(c.curve_knot_rows)
    centers = np.arange(-step_k, H + step_k, step_k)
    for m in centers:
        r = np.zeros(nu)
        for i in range(n):
            Y = knots[i] + oy[i]
            r[off[i]:off[i + 1]] = np.clip(1 - np.abs(Y - m) / step_k, 0, None)
        nr = float(np.linalg.norm(r))
        if nr > 0:
            r *= big / nr
            ATA += np.outer(r, r)
    info["sky_patterns"] = int(len(centers))
    sol = np.linalg.solve(ATA, ATb)
    curves = [sol[off[i]:off[i + 1]] for i in range(n)]
    info["ridge_to_data"] = float(ridge / float(np.median(dc[dc > 0]))) if (dc > 0).any() else None
    return curves, knots[0], info


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


# поля level2, которые только оценивают результат — сборку из-за них не пересобирать
# 3k: поля кривой по y; пока кривая выключена, в ключ не входят — сборка и её ключ как до 3k
CURVE_FIELDS = {"match_curve_y", "curve_knot_rows", "curve_prior", "curve_bin_px"}
CHECK_ONLY = {"star_fwhm_small_px", "star_fwhm_tol_small_percent", "star_fwhm_tol_percent",
              "workers"}  # не влияют на сборку: проверки и число процессов
# фаза 8: гибрид — шаг после стека, свой файл рядом со сборкой; ключ стека от него не зависит
BLEND_FIELDS = {"blend_large_scale", "blend_sigma_px"}
BLEND_VERSION = 2  # 2: вес заполненных из i2d пикселей — от i2d


def fwhm_check(rows: list[dict[str, Any]], c: Level2Config) -> dict[str, Any]:
    """Сколько звёзд в допуске FWHM к i2d. Допуск по фильтру: медиана FWHM i2d < star_fwhm_small_px — ±small, иначе ±обычный."""
    if not rows:
        return {}
    med = float(np.median([r["fwhm_i2d"] for r in rows]))
    tol = c.star_fwhm_tol_small_percent if med < c.star_fwhm_small_px else c.star_fwhm_tol_percent
    dev = [(r["fwhm_ours"] / r["fwhm_i2d"] - 1) * 100 for r in rows]
    return {"fwhm_i2d_median_px": round(med, 3), "tol_percent": tol, "deviation_percent": [round(d, 2) for d in dev],
            "passed": sum(abs(d) <= tol for d in dev), "total": len(dev)}


def build_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work_root / cfg.slug / "level2"


def _key(cfg: Config, files: list[Path], i2d: Path) -> dict[str, Any]:
    return {"version": BUILD_VERSION, "files": [[p.name, p.stat().st_size] for p in files],
            "i2d": [i2d.name, i2d.stat().st_size], "level2": cfg.level2.model_dump(exclude=CHECK_ONLY | BLEND_FIELDS | (
                set() if cfg.level2.match_curve_y else CURVE_FIELDS)),
            # только то, от чего зависят маски источников (иначе любое переключение в background пересобирает мозаику)
            "masks": {k: getattr(cfg.background, k) for k in ("mask_sigma", "mask_smooth_arcsec", "tile_mask_dilate_arcsec")}
            | {"stars": cfg.stars.mask_key()}}


def built_products(cfg: Config, paths: Paths) -> dict[str, Path]:
    d = build_dir(cfg, paths)
    meta = d / "meta.json"
    if not meta.exists():
        return {}
    m = json.loads(meta.read_text(encoding="utf-8"))
    blend = cfg.level2.blend_large_scale is not None
    out = {}
    for f, v in m.get("filters", {}).items():
        name = blend_name(v["file"], cfg) if blend else v["file"]
        if (d / name).exists():
            out[f] = d / name
    return out


def blend_name(base: str, cfg: Config) -> str:
    """Имя файла гибрида для собранного файла base: версия шага и сигма в имени — смена любой даёт новый файл."""
    s = cfg.level2.blend_sigma_px or 64.0
    return base.replace(".fits", f"_ls{cfg.level2.blend_large_scale}{s:g}v{BLEND_VERSION}.fits")


def blend_large_scale(base: Path, i2d_path: Path, sigma: float, dest: Path) -> dict[str, Any]:
    """Гибрид (фаза 8): своя − сглаженная(своя − i2d) на общей сетке (сборка — на сетке i2d); сглаживание гауссом с
    нормировкой по маске общих пикселей (пустые не тянут к нулю). Пустые пиксели своей сборки, где у i2d данные
    есть, — из i2d того же фильтра (слой FROMI2D). Остальные слои — как у сборки."""
    with fits.open(base, memmap=False) as h:
        ours = np.array(h["SCI"].data, dtype=np.float32)
        rest = [hdu.copy() for hdu in h]
    with fits.open(i2d_path, memmap=True) as h:
        ref = np.array(h["SCI"].data, dtype=np.float32)
        ref_wht = np.array(h["WHT"].data, dtype=np.float32) if "WHT" in h else np.ones_like(ref)
        ref[ref_wht <= 0] = np.nan
    if ref.shape != ours.shape:
        raise SkyforgeError(f"{base.name}: сетка сборки {ours.shape} не совпадает с i2d {ref.shape}")
    both = np.isfinite(ours) & np.isfinite(ref)
    diff = np.where(both, ours - ref, 0.0).astype(np.float32)
    num = ndimage.gaussian_filter(diff, sigma, mode="nearest")
    den = ndimage.gaussian_filter(both.astype(np.float32), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        smooth = np.where(den > 1e-3, num / den, 0.0).astype(np.float32)
    out = ours - smooth
    fill = ~np.isfinite(ours) & np.isfinite(ref)
    out[fill] = ref[fill]
    rest[1].data = out
    # вес заполненных — от i2d: загрузка FITS считает пиксели с WHT ≤ 0 пустыми (иначе заполнение пропадает при align)
    names = [hdu.name for hdu in rest]
    if "WHT" in names:
        w = np.array(rest[names.index("WHT")].data, dtype=np.float32)
        w[fill] = ref_wht[fill]
        rest[names.index("WHT")].data = w
    rest.append(fits.ImageHDU(fill.astype(np.uint8), name="FROMI2D"))
    rest[0].header["SKYBLEND"] = (f"i2d large scale, gauss {sigma:g} px", "own small scale + i2d large scale")
    # через временный файл: недописанный (обрыв, нехватка памяти) не выглядит готовым — сборка проверяет наличие файла
    tmp = dest.with_name(dest.name + ".part")
    fits.HDUList(rest).writeto(tmp, overwrite=True)
    tmp.replace(dest)
    finite = np.isfinite(ours)
    return {"sigma_px": sigma, "filled_from_i2d_px": int(fill.sum()),
            "filled_from_i2d_percent": round(float(fill.sum() / max(1, (finite | fill).sum()) * 100), 3),
            "smooth_diff_median": round(float(np.median(smooth[both])), 6),
            "smooth_diff_p95_abs": round(float(np.percentile(np.abs(smooth[both]), 95)), 6)}


def build(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    from skyforge.fetch import level2_files, require_products

    if not cfg.any_level2():
        return {"skipped": "source: level3"}
    # фаза 8: собираются только фильтры из уровня 2 (source: level2 — все, иначе level2_filters)
    l3 = {f: p for f, p in require_products(cfg, paths, level3=True).items() if cfg.uses_level2(f)}
    l2 = level2_files(cfg, paths)
    missing = [f for f in l3 if f not in l2]
    if missing:
        raise SkyforgeError(f"нет экспозиций уровня 2 для {missing}: skyforge fetch (source: level2)")
    d = build_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    meta_path = d / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"filters": {}}
    info: dict[str, Any] = {"filters": {}}
    previews: dict[str, list] = {"destripe": [], "overlaps": [], "stars": [], "astrometry": [], "curves": []}
    todo: dict[str, tuple[Path, str]] = {}
    for f, prod in l3.items():
        i2d_path = paths.abs(prod.path)
        key = _key(cfg, l2[f], i2d_path)
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
        old = meta["filters"].get(f)
        if old and old.get("digest") == digest and (d / old["file"]).exists():
            continue
        if old and (d / old["file"]).exists():
            (d / old["file"]).unlink()
        todo[f] = (i2d_path, digest)
    results: dict[str, tuple[dict, dict]] = {}
    if len(todo) > 1 and cfg.level2.workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        # короткие и длинные волны вперемешку: в памяти одновременно меньше больших сеток
        order = sorted(todo, key=lambda f: (list(l3).index(f) % 3, list(l3).index(f)))
        console.print(f"  сборка {len(todo)} фильтров в {min(cfg.level2.workers, len(todo))} процессах")
        with ProcessPoolExecutor(max_workers=min(cfg.level2.workers, len(todo))) as ex:
            futs = {f: ex.submit(_build_filter, cfg, f, todo[f][0], l2[f], d / f"{f}_l2_{todo[f][1]}.fits") for f in order}
            results = {f: fut.result() for f, fut in futs.items()}
    for f, prod in l3.items():
        old = meta["filters"].get(f)
        if f not in todo:
            console.print(f"  {f}: сборка из кеша ({old['file']})")
            info["filters"][f] = {**old, "cached": True, "fwhm_check": fwhm_check(old.get("star_rows", []), cfg.level2)}
            continue
        i2d_path, digest = todo[f]
        fi, pv = results[f] if f in results else _build_filter(cfg, f, i2d_path, l2[f], d / f"{f}_l2_{digest}.fits")
        fi.update(digest=digest, file=f"{f}_l2_{digest}.fits")
        meta["filters"][f] = fi
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
        info["filters"][f] = {**fi, "fwhm_check": fwhm_check(fi.get("star_rows", []), cfg.level2)}
        for k in previews:
            previews[k].append(pv[k])
    if cfg.level2.blend_large_scale == "i2d":
        # фаза 8: гибрид поверх собранного файла — свой файл, пересчёт только если его нет
        for f, prod in l3.items():
            v = meta["filters"].get(f)
            if not v:
                continue
            name = blend_name(v["file"], cfg)
            if not (d / name).exists():
                sigma = cfg.level2.blend_sigma_px or 64.0
                v["blend"] = blend_large_scale(d / v["file"], paths.abs(prod.path), sigma, d / name)
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
            b = v.get("blend", {})
            console.print(f"  {f}: гибрид (крупный масштаб от i2d, гаусс {b.get('sigma_px')} px): сглаженная разность "
                          f"медиана {b.get('smooth_diff_median')}, |95%| {b.get('smooth_diff_p95_abs')}; из i2d "
                          f"{b.get('filled_from_i2d_px')} px ({b.get('filled_from_i2d_percent')}%)")
            info["filters"].setdefault(f, {})["blend"] = b
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
    if any(previews["curves"]):
        _curves_sheet([x for x in previews["curves"] if x], d / "previews" / "l2_curves.png")
    if any(a for _, a in previews["astrometry"]):
        _astrometry_sheet(previews["astrometry"], d / "previews" / "l2_astrometry.png")
    for n in ("l2_destripe.png", "l2_overlaps.png", "l2_stars.png", "l2_astrometry.png", "l2_curves.png"):
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
    astro: list[dict[str, Any]] = []
    if c.tweak:
        # проход 1: подгонка по каждой экспозиции; решение — по группам детекторов (правила 3g)
        ref_stars = reference_stars(i2d, grid_wcs)
        for p in files:
            sci, _, fw, _ = load_exposure(p, c.dq_bad)
            astro.append(measure_tweak(sci, fw, ref_stars, c))
        decide_tweaks(astro, [p.name for p in files], c)
    exps, stripes = [], []
    demo = None
    for k, p in enumerate(files):
        if astro and astro[k]["exclude"]:
            continue
        sci, err, fw, _ = load_exposure(p, c.dq_bad, c.weights)
        if astro:
            fw, b_, a_ = apply_tweak(fw, astro[k])
            astro[k]["residuals_before"], astro[k]["residuals_after"] = b_, a_
            astro[k]["name"] = p.name
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
        e = {"name": p.name, "box": box, "data": project(clean, fw, grid_wcs, box), "err": project(err, fw, grid_wcs, box)}
        if c.match_per_amp or c.match_curve_y:
            aw = sci.shape[1] // max(1, c.amplifiers)
            amp_img = np.minimum(np.arange(sci.shape[1]) // aw, c.amplifiers - 1)[None, :].repeat(sci.shape[0], 0)
            y0, y1, x0, x1 = box
            ai, _ = reproject_interp((amp_img.astype(np.float32), fw), grid_wcs.slice((slice(y0, y1), slice(x0, x1))),
                                     shape_out=(y1 - y0, x1 - x0), order="nearest-neighbor")
            e["amp"] = np.where(np.isfinite(ai), ai, -1).astype(np.int8)
        if c.match_curve_y:  # строка детектора на сетке
            yd = np.arange(sci.shape[0], dtype=np.float32)[:, None].repeat(sci.shape[1], 1)
            y0, y1, x0, x1 = box
            e["ydet"], _ = reproject_interp((yd, fw), grid_wcs.slice((slice(y0, y1), slice(x0, x1))),
                                            shape_out=(y1 - y0, x1 - x0), order="bilinear")
            e["ydet"] = e["ydet"].astype(np.float32)
            e["ny"] = sci.shape[0]
        exps.append(e)
    curves_info: dict[str, Any] = {}
    curve_pv = None
    if c.match_curve_y and len(exps) > 1:
        from skyforge.background import pixel_noise

        sig1 = float(np.median([pixel_noise(e["data"], free_grid[e["box"][0]:e["box"][1], e["box"][2]:e["box"][3]]
                                            & np.isfinite(e["data"])) for e in exps]))
        cover = np.zeros(shape, dtype=np.int16)
        for e in exps:
            y0, y1, x0, x1 = e["box"]
            cover[y0:y1, x0:x1] += np.isfinite(e["data"])
        n_med = float(np.median(cover[free_grid & (cover > 0)])) if (free_grid & (cover > 0)).any() else 1.0
        sig_stack = sig1 / math.sqrt(max(n_med, 1.0))
        curves, knots, curves_info = match_curves_y(exps, free_grid, c, sig1, sig_stack)
        corr_sum = np.zeros(shape, dtype=np.float32)
        for e, g in zip(exps, curves):
            y0, y1, x0, x1 = e["box"]
            corr = np.interp(np.nan_to_num(e["ydet"], nan=0.0), knots, g).astype(np.float32)
            fin = np.isfinite(e["data"])
            e["data"] = np.where(fin, e["data"] - corr, e["data"]).astype(np.float32)
            corr_sum[y0:y1, x0:x1] += np.where(fin, corr, 0)
            del e["ydet"]
        corr_map = np.where(cover > 0, corr_sum / np.maximum(cover, 1), np.nan) / sig_stack
        amp_curves = [float(np.max(np.abs(g))) / sig_stack for g in curves]
        curves_info.update(sigma_stack=round(sig_stack, 6), knots=knots.tolist(),
                           curves=[[round(float(v) / sig_stack, 4) for v in g] for g in curves],   # σ стека
                           max_abs=round(max(amp_curves), 4), median_max_abs=round(float(np.median(amp_curves)), 4))
        curve_pv = (f, shrink(np.nan_to_num(corr_map), 600), curves_info["curves"], knots.tolist(),
                    [e["name"] for e in exps])
        console.print(f"    кривая по y: строк МНК {curves_info['rows']}, |кривая| макс по экспозиции медиана "
                      f"{curves_info['median_max_abs']:.3f}, макс {curves_info['max_abs']:.3f} σ стека")
    levels, pairs = (np.zeros(len(exps)), [])
    if c.match_levels and len(exps) > 1:
        if c.match_per_amp:
            levels, pairs = match_levels_per_amp(exps, free_grid)
        else:
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

    ov_metric, ov_map, sigma1 = overlap_residual(exps, res["sci"], res["nexp"], free_grid, c.overlap_block_px)
    stars = compare_stars(i2d, res["sci"], exps=exps)
    rb = float(np.median([s["rows_before"] for s in stripes]))
    ra = float(np.median([s["rows_after"] for s in stripes]))
    fi = {"exposures": len(exps), "rows_before": round(rb, 6), "rows_after": round(ra, 6),
          "pairs": len(pairs), "level_residual_sigma": round(ov_metric, 4), "sigma_single": round(sigma1, 6),
          "rejected_pixels": res["rejected"], "nexp_median": float(np.median(res["nexp"][res["nexp"] > 0])),
          "curves_y": curves_info,
          "stars": stars["summary"], "star_rows": stars["rows"],
          "astrometry": [_astro_row(p.name, a) for p, a in zip(files, astro)]}
    if astro:
        done = [a for a in astro if a["apply"]]
        own = [a for a in done if a["status"] in ("сдвиг", "поворот")]
        excl = sum(a["exclude"] for a in astro)
        fi["astrometry_summary"] = {
            "corrected_own": len(own), "corrected_group_median": len(done) - len(own),
            "not_corrected": len(astro) - len(done) - excl, "excluded": excl, "in_stack": len(astro) - excl,
            "rms_before_median": round(float(np.median([a["rms_before"] for a in own])), 4) if own else None,
            "rms_after_median": round(float(np.median([a["rms_after"] for a in own])), 4) if own else None}
        sm = fi["astrometry_summary"]
        console.print(f"    астрометрия: своя поправка {sm['corrected_own']}, медиана группы {sm['corrected_group_median']}, "
                      f"без поправки {sm['not_corrected']}, исключено {excl} (в стеке {sm['in_stack']}); "
                      f"остаток {sm['rms_before_median']} → {sm['rms_after_median']} px")
    console.print(f"    destripe |Δ строк| (медиана по экспозициям) {rb:.3g} → {ra:.3g}; пар перекрытий {len(pairs)}, "
                  f"остаток уровней {fi['level_residual_sigma']}σ; выбросов отброшено {res['rejected']}; "
                  f"звёзды: FWHM {stars['summary'].get('fwhm_ratio')}, поток {stars['summary'].get('flux_ratio')}")
    pv = {"destripe": _demo_images(f, demo), "overlaps": _overlap_images(f, res["nexp"], ov_map, ov_metric),
          "curves": curve_pv,
          "stars": stars["images"], "astrometry": (f, astro)}
    return fi, pv


def _astro_row(name: str, a: dict[str, Any]) -> dict[str, Any]:
    r = lambda v, n=4: None if v is None else round(float(v), n)  # noqa: E731
    ap = a.get("apply")
    return {"экспозиция": name.replace("_cal.fits", ""), "звёзд": a.get("stars", 0), "статус": a.get("status"),
            "dx": r(a.get("dx")), "dy": r(a.get("dy")), "поворот_град": r(a.get("rot_deg"), 5),
            "остаток_до": r(a.get("rms_before")), "остаток_после": r(a.get("rms_after")),
            "точность": r(a.get("precision")),
            "применено_dx": r(ap[0]) if ap else None, "применено_dy": r(ap[1]) if ap else None,
            "применён_поворот_град": r(math.degrees(ap[2]), 5) if ap else None, "исключена": a.get("exclude", False)}


def _curves_sheet(items: list, path: Path) -> None:
    """3k: поправка кривыми по y — средняя по экспозициям карта на сетке (σ стека) и кривые всех экспозиций."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(len(items), 2, figsize=(14, 5.5 * len(items)), squeeze=False)
    for r, (f, m, curves, knots, names) in enumerate(items):
        lim = max(0.3, float(np.nanpercentile(np.abs(m), 99)))
        im = ax[r, 0].imshow(m[::-1], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[r, 0].set_title(f"{f}: средняя поправка на сетке, σ стека"); ax[r, 0].axis("off")
        fig.colorbar(im, ax=ax[r, 0], fraction=0.04)
        for g, nm in zip(curves, names):
            det = nm.split("_")[-1].replace("_cal.fits", "")
            ax[r, 1].plot(knots, g, "-", lw=0.8, alpha=0.8, label=det)
        ax[r, 1].axhline(0, color="k", lw=0.5); ax[r, 1].grid(alpha=0.3)
        ax[r, 1].set_xlabel("строка детектора"); ax[r, 1].set_ylabel("кривая, σ стека")
        ax[r, 1].set_title(f"{f}: кривые экспозиций ({len(curves)})")
    fig.suptitle("Кривые фона по y детектора (3k): на карте не должно быть формы ореола")
    fig.tight_layout(); fig.savefig(path, dpi=60); plt.close(fig)


def _astrometry_sheet(items: list, path: Path) -> None:
    import matplotlib.pyplot as plt

    items = [(f, a) for f, a in items if a]
    fig, axes = plt.subplots(1, len(items), figsize=(3.6 * len(items), 3.2), squeeze=False)
    for ax, (f, astro) in zip(axes[0], items):
        b = [v for a in astro for v in a.get("residuals_before") or []]
        r = [v for a in astro for v in a.get("residuals_after") or []]
        bins = np.linspace(0, 0.6, 31)
        ax.hist(b, bins=bins, color="#c0392b", alpha=0.6, label=f"до (медиана {np.median(b) if b else 0:.3f})")
        ax.hist(r, bins=bins, color="#2471a3", alpha=0.6, label=f"после ({np.median(r) if r else 0:.3f})")
        done = sum(1 for a in astro if a.get("apply"))
        excl = sum(1 for a in astro if a.get("exclude"))
        ax.set_title(f"{f}: исправлено {done}/{len(astro)}" + (f", исключено {excl}" if excl else ""))
        ax.set_xlabel("|звезда − WCS|, px кадра")
        ax.legend(fontsize=8)
    fig.suptitle("Астрометрия экспозиций по звёздам i2d: остатки положений до и после подстройки")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


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


def _overlap_images(f: str, nexp: np.ndarray, ov_map: np.ndarray, metric: float) -> tuple[list, list]:
    n = shrink(nexp.astype(np.float32), 400)
    heat = np.nan_to_num(ov_map)
    return ([np.clip(n / max(float(n.max()), 1), 0, 1), np.clip(heat / 0.2, 0, 1)],
            [f"{f}: экспозиций на пиксель (до {int(nexp.max())})",
             f"{f}: |остаток| уровней по блокам, белое = 0.2σ; среднее {metric:.3f}σ"])


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
