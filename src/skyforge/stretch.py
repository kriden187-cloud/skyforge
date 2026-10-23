"""Растяжка: процентиль фона → 0, процентиль ярких деталей → 1, затем нелинейная кривая.

Каждый фильтр нормируется отдельно, поэтому разный уровень фона и единицы
(MJy/sr у JWST, e⁻/s у Hubble) не перекрашивают картинку.
NaN сохраняются до экспорта; вся статистика считается через nan-функции.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skyforge.config import Config, StretchConfig
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, save_png, shrink

# Статистика по прореженной сетке пикселей: детерминированно и в сотни раз быстрее сортировки всей мозаики.
STATS_SAMPLE_PIXELS = 4_000_000
HIST_BINS = 256


def sample(a: np.ndarray) -> np.ndarray:
    step = max(1, math.ceil(math.sqrt(a.size / STATS_SAMPLE_PIXELS)))
    return np.asarray(a[::step, ::step], dtype=np.float32)


def curve(x: np.ndarray, method: str, a: float) -> np.ndarray:
    if method == "asinh":
        return np.arcsinh(x / a) / np.arcsinh(1 / a)
    if method == "log":
        return np.log1p(x / a) / np.log1p(1 / a)
    if method == "sqrt":
        return np.sqrt(x)
    return x


def curve_inv(y: float, method: str, a: float) -> float:
    """Обратная к curve: какое значение до растяжки даёт y после."""
    if method == "asinh":
        return float(a * math.sinh(y * math.asinh(1 / a)))
    if method == "log":
        return float(a * math.expm1(y * math.log1p(1 / a)))
    if method == "sqrt":
        return float(y * y)
    return float(y)


def sky_toe(x: np.ndarray, u_sky: float, u_target: float, max_p: float | None = None) -> tuple[np.ndarray, float]:
    """Носок до растяжки (x в 0–1): u_sky → u_target. Выше u_sky — прямая через (u_sky, u_target) и (1, 1),
    то есть линейно по данным, как чёрная точка; ниже — u_target·(x/u_sky)^p, p подобрано так, что наклон
    в u_sky совпадает с прямой: кривая гладкая, монотонная, 0 → 0, без обрезки. Возвращает (x, p).
    max_p (фаза 4, stretch.sky_toe_max_p): при сильной растяжке и небе высоко по шкале p ~ 100–340 — «стенка»: шум
    чуть ниже неба уходит в 0. С ограничением значение у неба то же, ниже неба кривая положе (излом наклона на уровне
    неба); совпадение и значения, и наклона при крутом наклоне всё равно дало бы ноль в u_target/q ниже неба."""
    if not u_sky > u_target > 0:
        return x, 1.0
    q = (1 - u_target) / (1 - u_sky)
    p = q * u_sky / u_target
    if max_p is not None:
        p = min(p, max_p)
    low = u_target * np.power(np.maximum(x, 0) / np.float32(u_sky), np.float32(p))
    out = np.where(x >= u_sky, u_target + (x - u_sky) * np.float32(q), low)
    return out.astype(np.float32), float(p)


def masked_sample(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Выборка на той же решётке, что и sample(): шаг зависит от размера кадра, а не от числа пикселей маски —
    # иначе изменение маски на краю сдвигало бы всю выборку и процентили по всему кадру (3h, п.4).
    v = sample(a)[sample(np.asarray(mask, dtype=np.float32)) > 0.5]
    return v[np.isfinite(v)]


def noisy_sigma(a: np.ndarray, sky: np.ndarray, block: int) -> float:
    """σ неба самых шумных мест: робастный шум (MAD) в блоках block×block, где неба не меньше половины;
    98-й процентиль по блокам."""
    h, w = a.shape
    sig = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = sky[y:y + block, x:x + block]
            if m.mean() < 0.5:
                continue
            v = a[y:y + block, x:x + block][m]
            v = v[np.isfinite(v)]
            if v.size > 100:
                sig.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    return float(np.percentile(sig, 98)) if sig else 0.0


def stretch_array(arr: np.ndarray, s: StretchConfig, stats_from: np.ndarray | None = None,
                  masks: tuple[np.ndarray, np.ndarray] | None = None,
                  field: Any = None, gain: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, float]]:
    """stats_from — полный кадр, если arr его уменьшенная копия (для sweep).
    masks — (пустое небо, туманность) на сетке stats_from: при anchor masks чёрная точка берётся
    по небу, белая — по туманности. gain — множитель режима ядра (stretch.core, фаза 9) после чёрной и белой
    точек, до обрезки 0–1."""
    src = arr if stats_from is None else stats_from
    if field is not None:  # поле статистики (skyforge.field): процентили — только там, где есть все фильтры
        if masks is not None:
            masks = (field.mask(masks[0]), field.mask(masks[1]))
        src = field.values(src)
    smp = sample(src)
    if not np.isfinite(smp).any():
        raise SkyforgeError("в кадре нет ни одного числа — только NaN")
    anchor = "frame"
    if s.anchor == "masks" and masks is not None:
        sky_v, neb_v = masked_sample(src, masks[0]), masked_sample(src, masks[1])
        if sky_v.size > 100 and neb_v.size > 100:
            lo = float(np.percentile(sky_v, s.black))
            hi = float(np.percentile(neb_v, s.white))
            anchor = "masks"
            sky_med = float(np.median(sky_v))
            if s.black_mode == "pedestal":
                lo = sky_med - s.black_k * noisy_sigma(src, masks[0], s.noise_block_px)
                anchor = "masks, пьедестал"
    if anchor == "frame":
        lo, hi = (float(v) for v in np.nanpercentile(smp, [s.black, s.white]))
    if not hi > lo:
        raise SkyforgeError(f"процентили black={s.black} и white={s.white} дают одно значение {lo}: кадр пустой?")
    x = (np.asarray(arr, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo)
    if gain is not None:
        x *= gain
    np.clip(x, 0, 1, out=x)  # NaN остаются NaN
    toe = None
    if s.sky_level is not None and anchor != "frame":
        u_sky = (sky_med - lo) / (hi - lo)
        u_t = curve_inv(s.sky_level / 255, s.method, s.a)
        x, p = sky_toe(x, u_sky, u_t, s.sky_toe_max_p)
        toe = {"u_sky": round(u_sky, 6), "u_target": round(u_t, 6), "p": round(p, 3)}
    y = curve(x, s.method, s.a).astype(np.float32)
    stats = {
        "black_value": lo,
        "white_value": hi,
        "white_clipped_percent": round(float(np.nanmean(smp >= hi) * 100), 3),
        "median_after": round(float(np.nanmedian(sample(y))), 4),
        "anchor": anchor,
    }
    if toe is not None:
        stats["sky_toe"] = toe
    return y, stats


def stretched_dir(cfg: Config, paths: Paths) -> Path:
    return paths.style_work(cfg.slug, cfg.bin, cfg.style) / "stretched"


def stretch_key(cfg: Config, in_key: dict[str, Any]) -> dict[str, Any]:
    exclude = set() if cfg.stretch.sky_toe_max_p is not None else {"sky_toe_max_p"}
    if cfg.stretch.sky_reference == "mask":   # шаг 4d: поля опоры — только когда она не по умолчанию
        exclude |= {"sky_reference", "sky_reference_block_px", "sky_reference_n_blocks", "sky_reference_min_free"}
    if cfg.stretch.sky_reference_crop is None:
        exclude.add("sky_reference_crop")
    if cfg.stretch.exclude_point_sources is None:
        exclude.add("exclude_point_sources")
    if cfg.stretch.exclude_dominant_spot is None:
        exclude.add("exclude_dominant_spot")
    if cfg.stretch.core is None:
        exclude.add("core")
    st = cfg.stretch.model_dump(mode="json", exclude=exclude)
    if st.get("exclude_point_sources") and st["exclude_point_sources"].get("max_fwhm_psf") is None:
        del st["exclude_point_sources"]["max_fwhm_psf"]   # фаза 9: без значения — ключ кеша прежний
    key = {"input": in_key, "stretch": st}   # фаза 4: новые поля в ключе только когда заданы — кеши прежних объектов те же
    if cfg.stretch.exclude_point_sources is not None and cfg.stretch.exclude_point_sources.max_fwhm_psf is not None:
        key["point_width_version"] = POINT_WIDTH_VERSION
    if cfg.stretch.sky_reference == "crop":   # фаза 6: координаты кропа опоры (report в хеш не входит)
        key["sky_reference_crop_spec"] = next(c.model_dump(mode="json") for c in cfg.report.crops
                                              if c.name == cfg.stretch.sky_reference_crop)
    if cfg.clean.fill_gaps_px > 0:
        key["fill_gaps_px"] = cfg.clean.fill_gaps_px
    if cfg.stretch.sky_poly_order > 0:
        from skyforge.skymodel import SKY_POLY_VERSION

        key["sky_poly_version"] = SKY_POLY_VERSION
    if cfg.stretch.anchor == "masks":
        key["sky_grow_px"] = cfg.color.sky_grow_px  # маска неба зависит от отступа
    return key


def darkest_blocks(a: np.ndarray, ok: np.ndarray, block: int, n: int, min_free: float) -> tuple[np.ndarray, dict]:
    """Шаг 4d: n блоков block×block с самой низкой медианой a по пикселям ok (доля ok в блоке ≥ min_free).
    Возвращает маску опорного неба (ok внутри выбранных блоков) и {blocks: [[строка, столбец, медиана]], eligible, total}."""
    H, W = a.shape
    cand, total = [], 0
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            total += 1
            m = ok[y:y + block, x:x + block]
            if m.mean() >= min_free:
                cand.append((float(np.median(a[y:y + block, x:x + block][m])), y, x))
    if len(cand) < n:
        raise SkyforgeError(f"stretch.sky_reference: блоков с долей чистых пикселей ≥ {min_free} — {len(cand)} из {total}, "
                            f"нужно {n}: уменьши sky_reference_block_px или sky_reference_min_free")
    cand.sort()
    ref = np.zeros((H, W), dtype=bool)
    for _, y, x in cand[:n]:
        ref[y:y + block, x:x + block] = ok[y:y + block, x:x + block]
    return ref, {"blocks": [[y, x, round(v, 6)] for v, y, x in cand[:n]], "eligible": len(cand), "total": total}


def reference_sky(cfg: Config, paths: Paths, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Опорное небо по самым тёмным блокам опорного фильтра (cfg.reference) после clean: чистые пиксели — поле всех
    фильтров вне маски звёзд clean (с отступом 3 px)."""
    from scipy import ndimage

    from skyforge.clean import clean_dir

    d = clean_dir(cfg, paths)
    if not cfg.reference:
        raise SkyforgeError("stretch.sky_reference: darkest_blocks — нужен reference (опорный фильтр) в конфиге")
    full = np.ones(shape, dtype=bool)
    for f in cfg.filters:
        full &= np.isfinite(np.load(d / f"{f}.npy", mmap_mode="r"))
    stars = ndimage.binary_dilation(np.load(d / "star_weight.npy") > 0.01, iterations=3)
    s = cfg.stretch
    if s.sky_reference == "crop":
        from skyforge.align import with_resolved_crops
        from skyforge.preview import display_crop

        k = next(c for c in with_resolved_crops(cfg, paths).report.crops if c.name == s.sky_reference_crop)
        sl = display_crop(shape, k.x, k.y, k.size, cfg.bin)
        ref = np.zeros(shape, dtype=bool)
        ref[sl] = (full & ~stars)[sl]
        if ref.sum() < 0.25 * ref[sl].size:
            console.print(f"[yellow]  опорное небо: в кропе {k.name} чистых пикселей {ref.sum()} из {ref[sl].size}[/]")
        return ref, {"crop": k.name, "rows": [sl[0].start, sl[0].stop], "cols": [sl[1].start, sl[1].stop],
                     "pixels": int(ref.sum())}
    ref, info = darkest_blocks(np.load(d / f"{cfg.reference}.npy", mmap_mode="r"), full & ~stars,
                               max(16, s.sky_reference_block_px // cfg.bin), s.sky_reference_n_blocks, s.sky_reference_min_free)
    if info["eligible"] < 0.25 * info["total"]:
        console.print(f"[yellow]  опорное небо: блоков без звёзд {info['eligible']} из {info['total']} — опора может уйти "
                      f"в туманность (шаг 4d); уменьши sky_reference_min_free[/]")
    return ref, info


def load_masks(cfg: Config, paths: Paths) -> tuple[np.ndarray, np.ndarray] | None:
    """(пустое небо, туманность) с этапа clean; None, если масок нет. stretch.sky_reference: darkest_blocks —
    вместо маски неба опора по самым тёмным блокам (шаг 4d)."""
    from skyforge.clean import load_nebula_sd, load_sky_mask

    sky, sd = load_sky_mask(cfg, paths), load_nebula_sd(cfg, paths)
    if sky is None or sd is None:
        return None
    if cfg.stretch.sky_reference in ("darkest_blocks", "crop"):
        sky, _ = reference_sky(cfg, paths, sd.shape)
    neb = sd > 0
    if cfg.stretch.exclude_stars:
        from skyforge.clean import star_exclusion

        neb &= ~star_exclusion(cfg, paths, neb.shape)  # белая точка — по туманности, а не по звёздам в ней
    return sky, neb


POINT_WIDTH_VERSION = 2   # 2: окно максимума и резкости по PSF (при bin 1 наземные звёзды шире окна 5 px)


def _fwhm_px(r: np.ndarray, y: int, x: int, half: int) -> float | None:
    """FWHM находки (px) гауссом с постоянным фоном в окне (2·half+1)²; None — подгонка не сошлась."""
    from scipy.optimize import curve_fit

    w = r[y - half:y + half + 1, x - half:x + half + 1].astype(np.float64)
    if w.shape != (2 * half + 1, 2 * half + 1) or not w[half, half] > 0:
        return None
    w = w / w[half, half]   # единицы бывают ~1e-11 (тайлы ESO) — без нормировки подгонка стоит на начальном приближении
    yy, xx = np.indices(w.shape)

    def g(c, amp, x0, y0, sg, b):
        return amp * np.exp(-((c[1] - x0) ** 2 + (c[0] - y0) ** 2) / (2 * sg ** 2)) + b

    try:
        p, _ = curve_fit(g, (yy.ravel(), xx.ravel()), w.ravel(), p0=(1.0, half, half, 1.0, 0.0),
                         bounds=([0, half - 2, half - 2, 0.3, -np.inf], [np.inf, half + 2, half + 2, 3.0 * half, np.inf]),
                         maxfev=2000)
    except Exception:
        return None
    return 2.3548 * float(p[3])


def point_sources(a: np.ndarray, ps: Any, scale_arcsec: float,
                  psf_arcsec: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """stretch.exclude_point_sources: круги вокруг резких локальных максимумов (звёзд) до радиуса, где кольцо уходит
    в местный фон. При ps.max_fwhm_psf (фаза 9) звезда — только находка не шире max_fwhm_psf × PSF; PSF — бо́льшее из
    заголовка (psf_arcsec) и медианы FWHM верхних 5% находок по амплитуде; неподогнанные остаются звёздами. Возвращает маску и {stars,
    max_radius_arcsec, noise[, candidates, psf_px]}."""
    from scipy import ndimage

    from skyforge.field import block_median_map

    ok = np.isfinite(a)
    bg = block_median_map(a, max(16, int(round(ps.window_arcsec / scale_arcsec))), fill_empty=True)
    r = np.where(ok, a - bg, 0).astype(np.float32)
    d = np.diff(np.where(ok, a, np.nan), axis=1)
    d = d[np.isfinite(d)]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size else 0.0
    # окно максимума и резкости — 5 px (звезда ~2 px); при проверке ширины (фаза 9) — около 2,5 PSF: при bin 1 наземная
    # звезда 4–6 px, пик в окне 5×5 почти не выше среднего, и резкость отсеивала все звёзды (bin 4 — по-прежнему 5)
    k = 5
    if getattr(ps, "max_fwhm_psf", None) and psf_arcsec:
        k = max(5, 2 * int(round(1.25 * psf_arcsec / scale_arcsec)) + 1)
    peaks = np.argwhere((r == ndimage.maximum_filter(r, k)) & (r > ps.sigma * noise)
                        & (r > ps.sharpness * np.maximum(ndimage.uniform_filter(r, k), 1e-12)))
    width: dict[str, Any] = {}
    if getattr(ps, "max_fwhm_psf", None) and len(peaks):
        head_px = psf_arcsec / scale_arcsec if psf_arcsec else None
        half = max(4, int(np.ceil(2.5 * (head_px or 2.0))))
        fw = np.array([_fwhm_px(r, int(y), int(x), half) or np.nan for y, x in peaks])
        amp = r[peaks[:, 0], peaks[:, 1]]
        top = (amp >= np.percentile(amp, 95)) & np.isfinite(fw)
        meas_px = float(np.median(fw[top])) if top.any() else float(np.nanmedian(fw))
        # PSF заголовка — для исходного пикселя; при bin звезда шире (пиксель и недодискретизация) — берётся бо́льшее
        psf_px = max(head_px or 0.0, meas_px)
        keep = ~np.isfinite(fw) | (fw <= ps.max_fwhm_psf * psf_px)
        width = {"candidates": int(len(peaks)), "psf_px": round(psf_px, 3), "psf_header_px": round(head_px, 3) if head_px else None,
                 "psf_bright_px": round(meas_px, 3), "wider_than_psf": int((~keep).sum())}
        peaks = peaks[keep]
    H, W = a.shape
    rmax = max(2, int(ps.max_radius_arcsec / scale_arcsec))
    out = np.zeros(a.shape, dtype=bool)
    biggest = 0
    def window(y: int, x: int, half: int) -> tuple[tuple[slice, slice], np.ndarray]:
        y0, y1, x0, x1 = max(y - half, 0), min(y + half + 1, H), max(x - half, 0), min(x + half + 1, W)
        yy, xx = np.ogrid[y0 - y:y1 - y, x0 - x:x1 - x]
        return (slice(y0, y1), slice(x0, x1)), np.sqrt(yy ** 2 + xx ** 2).astype(np.int32)

    # окно растёт ступенями: кольца радиуса ≤ half лежат в окне целиком — результат тот же, что в окне rmax,
    # а у большинства звёзд кольцо уходит в фон за несколько пикселей (bin 1: rmax 127 px, звёзд тысячи)
    steps = sorted({min(8, rmax), min(32, rmax), rmax})
    for y, x in peaks:
        rad = rmax
        for half in steps:
            sl, rr = window(y, x, half)
            sub = r[sl]
            q = next((q for q in range(1, half + 1)          # первое кольцо, где медиана остатка — уже фон
                      if (ring := sub[rr == q]).size and np.median(ring) < ps.ring_sigma * noise), None)
            if q is not None:
                rad = q
                break
        biggest = max(biggest, rad)
        sl, rr = window(y, x, min(rad + 1, rmax))
        out[sl] |= rr <= rad + 1
    return out, {"stars": int(len(peaks)), "max_radius_arcsec": round(biggest * scale_arcsec, 2),
                 "noise": round(noise, 6), **width}


def dominant_spot(a: np.ndarray, neb: np.ndarray, white: float, sp: Any, scale_arcsec: float,
                  field: Any = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """stretch.exclude_dominant_spot (фаза 8): если среди пикселей выборки туманности выше белой точки (процентиль white,
    в поле статистики) доля одного связного пятна ≥ sp.share — маска пятна с отступом по спаду профиля: кольца от края
    пятна, пока медиана остатка над местной медианой не упадёт ниже ring_sigma·σ шума (как у точечных источников).
    Возвращает (маска или None, {share, ...})."""
    from scipy import ndimage

    from skyforge.field import block_median_map

    inf = np.zeros(a.shape, dtype=bool)
    if field is not None:
        inf[field.rows, field.cols] = field.full
    else:
        inf[:] = True
    sel = neb & np.isfinite(a) & inf
    if sel.sum() < 100:
        return None, {}
    hi = float(np.percentile(a[sel], white))
    hot = sel & (a >= hi)
    lab, n = ndimage.label(hot, structure=np.ones((3, 3), dtype=bool))
    if n == 0:
        return None, {}
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    share = float(sizes[k - 1] / sizes.sum())
    info: dict[str, Any] = {"share": round(share, 3), "regions": int(n)}
    if share < sp.share:
        return None, info
    rmax = max(2, int(sp.max_radius_arcsec / scale_arcsec))
    sl = ndimage.find_objects((lab == k).astype(np.int32))[0]
    y0, y1 = max(sl[0].start - rmax - 1, 0), min(sl[0].stop + rmax + 1, a.shape[0])
    x0, x1 = max(sl[1].start - rmax - 1, 0), min(sl[1].stop + rmax + 1, a.shape[1])
    sub = a[y0:y1, x0:x1]
    ok = np.isfinite(sub)
    bg = block_median_map(np.where(ok, sub, np.nan), max(16, int(round(sp.window_arcsec / scale_arcsec))), fill_empty=True)
    r = np.where(ok, sub - bg, np.nan)
    d = np.diff(np.where(np.isfinite(a), a, np.nan), axis=1)
    d = d[np.isfinite(d)]
    noise = float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size else 0.0
    dist = ndimage.distance_transform_edt(lab[y0:y1, x0:x1] != k)
    rad = rmax
    for q in range(1, rmax + 1):
        ring = r[(dist > q - 1) & (dist <= q)]
        ring = ring[np.isfinite(ring)]
        if ring.size and float(np.median(ring)) < sp.ring_sigma * noise:
            rad = q
            break
    out = np.zeros(a.shape, dtype=bool)
    out[y0:y1, x0:x1] = dist <= rad
    cy, cx = ndimage.center_of_mass(lab == k)
    info.update(pixels=int(sizes[k - 1]), center=[int(round(cy)), int(round(cx))], radius_arcsec=round(rad * scale_arcsec, 2),
                white_before=round(hi, 4))
    return out, info


def _spot_text(si: dict[str, Any] | None) -> str:
    if not si:
        return ""
    if "radius_arcsec" not in si:
        return f"; крупнейшее пятно над белой точкой — {si['share'] * 100:.0f}% (не исключается)"
    return (f"; пятно {si['share'] * 100:.0f}% пикселей над белой точкой ({si['white_before']:.4g}) исключено с отступом "
            f"{si['radius_arcsec']}″ — {si['nebula_excluded_percent']}% выборки туманности")


def filter_masks(cfg: Config, a: np.ndarray, masks: tuple[np.ndarray, np.ndarray] | None,
                 scale_arcsec: float | None, field: Any = None,
                 psf_arcsec: float | None = None) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict[str, Any]]:
    """Маски растяжки одного фильтра: при stretch.exclude_point_sources — туманность без его точечных источников;
    при stretch.exclude_dominant_spot — и без пятна, задающего белую точку в одиночку (фаза 8)."""
    ps, sp = cfg.stretch.exclude_point_sources, cfg.stretch.exclude_dominant_spot
    if (ps is None and sp is None) or masks is None or not scale_arcsec:
        return masks, {}
    info: dict[str, Any] = {}
    neb = masks[1]
    if ps is not None:
        src, info = point_sources(a, ps, scale_arcsec, psf_arcsec)
        info["nebula_excluded_percent"] = round(float((neb & src).sum() / max(neb.sum(), 1) * 100), 1)
        neb = neb & ~src
    if sp is not None:
        spot, sinfo = dominant_spot(a, neb, cfg.stretch.white, sp, scale_arcsec, field)
        if spot is not None:
            sinfo["nebula_excluded_percent"] = round(float((neb & spot).sum() / max(neb.sum(), 1) * 100), 2)
            neb = neb & ~spot
        info["dominant_spot"] = sinfo
    return (masks[0], neb), info


def psf_arcsec(cfg: Config, paths: Paths) -> dict[str, float]:
    """FWHM PSF полос из заголовков продуктов (PSF_FWHM, ″) — у тайлов ESO он есть; у i2d нет (пусто)."""
    from astropy.io import fits

    from skyforge.fetch import require_products

    out = {}
    for f, prod in require_products(cfg, paths).items():
        v = fits.getheader(paths.abs(prod.path), 0).get("PSF_FWHM")
        if v:
            out[f] = float(v)
    return out


def _c1_log(slope_target: float) -> float:
    """c > 0, при котором c / ln(1 + c) = slope_target (≥ 1): наклон логарифмической кривой в начале."""
    if slope_target <= 1.0 + 1e-9:
        return 1e-9
    lo, hi = 1e-9, 1e9
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if mid / math.log1p(mid) < slope_target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def core_gain(arrays: dict[str, np.ndarray], stats: dict[str, dict], cc: Any, scale_arcsec: float,
              field: Any = None) -> tuple[np.ndarray | None, dict[str, Any]]:
    """stretch.core (фаза 9): общий для полос множитель сжатия яркого центра по сглаженной яркости (см. CoreConfig)."""
    from scipy import ndimage

    xs = [(a - stats[f]["black_value"]) / (stats[f]["white_value"] - stats[f]["black_value"]) for f, a in arrays.items()]
    with np.errstate(invalid="ignore"):
        L = np.nanmean(np.stack(xs), axis=0).astype(np.float32)
    del xs
    ok = np.isfinite(L)
    sig = max(cc.sigma_arcsec / scale_arcsec, 0.5)
    num = ndimage.gaussian_filter(np.where(ok, L, 0).astype(np.float32), sig)
    den = ndimage.gaussian_filter(ok.astype(np.float32), sig)
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(den > 0.3, num / den, np.nan).astype(np.float32)
    t0, top = cc.knee, cc.top
    # только центр: самая большая по площади связная область выше knee (яркие звёзды поля выше knee тоже — их не
    # трогаем, иначе меняется всё поле)
    lab, n = ndimage.label(np.nan_to_num(S, nan=0.0) > t0)
    if n == 0:
        return None, {"applied": False, "knee": t0, "top": top, "inner_white": None}
    sizes = np.bincount(lab.ravel())[1:]
    centre = lab == (int(np.argmax(sizes)) + 1)
    vals = S[centre & np.isfinite(S)]
    smax = float(np.percentile(vals, cc.inner_percentile)) if vals.size else float("nan")
    info: dict[str, Any] = {"inner_white": round(smax, 4), "knee": t0, "top": top, "regions_above_knee": int(n)}
    if not np.isfinite(smax) or smax <= top:
        info["applied"] = False
        return None, info
    c = _c1_log((smax - t0) / (top - t0))
    u = np.clip((S - t0) / (smax - t0), 0, None)
    out = t0 + (top - t0) * np.log1p(c * u) / math.log1p(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        gain = np.where(centre & np.isfinite(S), out / S, 1.0).astype(np.float32)
    info.update(applied=True, c=round(c, 3), area_percent=round(float(centre.mean() * 100), 3),
                min_gain=round(float(np.nanmin(gain)), 4))
    return gain, info


def stretch_all(cfg: Config, arrays: dict[str, np.ndarray],
                masks: tuple[np.ndarray, np.ndarray] | None = None,
                field: Any = None, scale_arcsec: float | None = None,
                psfs: dict[str, float] | None = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    out, stats, fmasks = {}, {}, {}
    for f, a in arrays.items():
        mf, ps_info = filter_masks(cfg, a, masks, scale_arcsec, field, (psfs or {}).get(f))
        fmasks[f] = mf
        out[f], stats[f] = stretch_array(a, cfg.stretch.for_filter(f), masks=mf, field=field)
        if "dominant_spot" in ps_info:
            stats[f]["dominant_spot"] = ps_info.pop("dominant_spot")
        if ps_info:
            stats[f]["point_sources"] = ps_info
    if cfg.stretch.core is not None and scale_arcsec:
        # фаза 9: второй проход с множителем ядра — те же чёрная и белая точки, сжатие яркого центра
        gain, cinfo = core_gain(arrays, stats, cfg.stretch.core, scale_arcsec, field)
        if gain is not None:
            for f, a in arrays.items():
                out[f], st = stretch_array(a, cfg.stretch.for_filter(f), masks=fmasks[f], field=field, gain=gain)
                x = (a - st["black_value"]) / (st["white_value"] - st["black_value"]) * gain
                st["white_clipped_percent_core"] = round(float(np.nanmean(x >= 1) * 100), 3)
                stats[f].update(st)
        for f in stats:
            stats[f]["core"] = cinfo
    return out, stats


def stretch(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    from skyforge.clean import load_input  # clean сам использует растяжку для превью

    arrays, in_key = load_input(cfg, paths)
    if not arrays:
        raise SkyforgeError("среди совмещённых кадров нет фильтров из конфига")
    from skyforge.skymodel import apply_sky_poly, sky_poly_sheet

    extra_pv = []
    gaps_info: dict[str, int] = {}
    if cfg.clean.fill_gaps_px > 0:  # 3б: швы чипов — в стиле, не в общем кеше clean
        from skyforge.cosmics import fill_gaps_linear

        filled_maps = {}
        for f in list(arrays):
            arrays[f], filled_maps[f] = fill_gaps_linear(arrays[f], cfg.clean.fill_gaps_px)
            gaps_info[f] = int(filled_maps[f].sum())
        console.print("  швы заполнены (полуширина ≤ " + str(cfg.clean.fill_gaps_px) + " px): "
                      + ", ".join(f"{f} {n} px" for f, n in gaps_info.items()))
        _gaps_sheet(arrays, filled_maps, out_dir / "gaps_filled.png")
        extra_pv.append("gaps_filled.png")
    arrays, poly_models, poly_info = apply_sky_poly(cfg, paths, arrays)
    if poly_models:
        sky_poly_sheet(poly_models, poly_info, out_dir / "sky_poly.png")
        extra_pv.append("sky_poly.png")
        for f in arrays:
            console.print(f"  {f}: полином неба {cfg.stretch.sky_poly_order}-го порядка — размах {poly_info[f]['range_sigma']}σ, "
                          f"в зоне исключения {poly_info[f]['in_excluded_range_sigma']}σ")
    d = stretched_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    masks = load_masks(cfg, paths) if cfg.stretch.anchor == "masks" else None
    if cfg.stretch.anchor == "masks" and masks is None:
        console.print("[yellow]  stretch.anchor: masks — масок нет (clean выключен), растяжка по всему кадру[/]")
    from skyforge.align import read_aligned_meta, stats_field

    field = stats_field(cfg, paths, list(arrays.values()))
    scale = read_aligned_meta(cfg, paths).get("pixel_scale_arcsec")
    ps = cfg.stretch.exclude_point_sources
    psfs = psf_arcsec(cfg, paths) if ps is not None and ps.max_fwhm_psf else {}
    stretched, stats = stretch_all(cfg, arrays, masks, field, scale, psfs)
    ss = cfg.stretch.saturated_stars
    sinfo: list = []
    if ss.enabled:
        from skyforge.clean import load_saturated_stars
        from skyforge.satstars import screen, star_layers

        stars = load_saturated_stars(cfg, paths)
        core = core_gain(arrays, stats, cfg.stretch.core, scale, field)[0] if cfg.stretch.core is not None and scale else None
        if stars:
            lin, disp, sinfo = star_layers(arrays, {f: stats[f]["white_value"] for f in arrays}, stars, ss, cfg.bin)
            for f, a in arrays.items():
                # туманность без звезды, точки растяжки — по исходному кадру (не меняются)
                mf, _ = filter_masks(cfg, a, masks, scale, field, psfs.get(f))
                neb, _ = stretch_array(a - lin[f], cfg.stretch.for_filter(f), stats_from=a, masks=mf, field=field,
                                       gain=core)
                stretched[f] = screen(neb, disp[f])
            console.print(f"  звёзды с насыщенным ядром отдельно от туманности: {len(sinfo)}")
    for f, y in stretched.items():
        np.save(d / f"{f}.npy", y)
    for f in stretched:
        s = cfg.stretch.for_filter(f)
        console.print(f"  {f}: фон {stats[f]['black_value']:.4g} → 0, белая точка {stats[f]['white_value']:.4g} → 1, "
                      f"{s.method} a={s.a}, по {'маскам' if stats[f]['anchor'].startswith('masks') else 'кадру'}, "
                      f"выжжено {stats[f]['white_clipped_percent']}%"
                      + (f"; точечных источников {stats[f]['point_sources']['stars']}"
                         + (f" из {stats[f]['point_sources']['candidates']} (шире {cfg.stretch.exclude_point_sources.max_fwhm_psf}"
                            f" PSF {stats[f]['point_sources']['psf_px']} px — {stats[f]['point_sources']['wider_than_psf']}, остаются)"
                            if "candidates" in stats[f]["point_sources"] else "") + " (до "
                         f"{stats[f]['point_sources']['max_radius_arcsec']}″), из выборки туманности исключено "
                         f"{stats[f]['point_sources']['nebula_excluded_percent']}%" if "point_sources" in stats[f] else "")
                      + _spot_text(stats[f].get("dominant_spot")))
    ci = next((st["core"] for st in stats.values() if "core" in st), None)
    if ci:
        console.print(f"  режим ядра: " + (f"внутренняя белая точка {ci['inner_white']} внешней, сжатие с {ci['knee']} на "
                                            f"{ci['area_percent']}% кадра, множитель до {ci['min_gain']}; выжжено после: "
                                            + ", ".join(f"{f} {st.get('white_clipped_percent_core')}%" for f, st in stats.items())
                                            if ci.get("applied") else f"не нужен (внутренняя белая точка {ci['inner_white']})"))
    (d / "meta.json").write_text(json.dumps({"key": stretch_key(cfg, in_key), "filters": list(stretched), "stats": stats,
                                             "saturated_stars": sinfo},
                                            ensure_ascii=False, indent=2), encoding="utf-8")

    pv = cfg.preview.max_size
    thumbs = {f: shrink(y, pv) for f, y in stretched.items()}
    for f, t in thumbs.items():
        save_png(t, out_dir / f"stretch_{f}.png")
    contact_sheet(list(thumbs.values()), list(thumbs), out_dir / "stretch_sheet.png",
                  title=f"После растяжки ({cfg.stretch.method})")
    _hist_sheet(arrays, stretched, stats, out_dir / "stretch_hist.png")
    return {"stats": stats, "saturated_stars": sinfo, "sky_poly": poly_info, "gaps_filled": gaps_info,
            "previews": ["stretch_hist.png", "stretch_sheet.png"] + extra_pv}


def _gaps_sheet(arrays: dict[str, np.ndarray], filled: dict[str, np.ndarray], path: Path) -> None:
    """3б: карта заполненных пикселей (красные) поверх кадра опорного фильтра."""
    from skyforge.preview import quick_norm

    f0 = next(iter(arrays))
    base = quick_norm(arrays[f0], 1.0, 99.5, 0.05)
    any_f = np.logical_or.reduce(list(filled.values()))
    rgb = np.stack([base] * 3, -1)
    rgb[any_f] = [1.0, 0.1, 0.1]
    contact_sheet([rgb], [f"заполненные швы и чёрточки (красные): {int(any_f.sum())} px"], path, cols=1, tile_inches=10,
                  interpolation="nearest", title="Швы чипов: линейно поперёк шва, только где данные с обеих сторон")


def _hist_sheet(raw: dict[str, np.ndarray], done: dict[str, np.ndarray], stats: dict[str, dict], path: Path) -> None:
    n = len(raw)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.3 * n), squeeze=False)
    for i, f in enumerate(raw):
        r = sample(raw[f])
        r = r[np.isfinite(r)]
        lo, hi = stats[f]["black_value"], stats[f]["white_value"]
        span = hi - lo
        ax = axes[i, 0]
        ax.hist(r, bins=HIST_BINS, range=(lo - 0.5 * span, hi + 0.5 * span), color="#4a7fd1", log=True)
        ax.axvline(lo, color="k", lw=1, ls="--")
        ax.axvline(hi, color="#d14a4a", lw=1, ls="--")
        ax.set_ylabel(f, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title("до: исходные значения, пунктир — чёрная и белая точки")
        s = sample(done[f])
        ax = axes[i, 1]
        ax.hist(s[np.isfinite(s)], bins=HIST_BINS, range=(0, 1), color="#d18a4a", log=True)
        if i == 0:
            ax.set_title("после: 0–1, пик фона у нуля, хвост вправо")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
