"""Фаза 3f: сборка из экспозиций уровня 2 — destripe в кадре детектора, сведение уровней, стек."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skyforge.config import Config, Level2Config

CENTER = (150.0, -40.0)


def _wcs(shape, scale_arcsec, shift=(0.0, 0.0), rot=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(CENTER)
    w.wcs.crpix = [shape[1] / 2 + 0.5 + shift[0], shape[0] / 2 + 0.5 + shift[1]]
    s, t = scale_arcsec / 3600, np.radians(rot)
    w.wcs.cd = s * np.array([[-np.cos(t), np.sin(t)], [np.sin(t), np.cos(t)]])
    return w


STARS = [(-2.0, 1.5, 300.0), (1.8, -1.2, 250.0), (-0.5, -2.2, 200.0), (2.2, 2.0, 280.0), (0.3, 0.9, 220.0)]


def _sky(wcs, shape, scale):
    """Небо в MJy/sr-подобных единицах: фон 1, звёзды Гаусса σ=0.07″ (поверхностная яркость)."""
    yy, xx = np.indices(shape, dtype=np.float64)
    img = np.full(shape, 1.0)
    cosd = np.cos(np.radians(CENTER[1]))
    sig = 0.07 / scale
    for dx, dy, peak in STARS:
        x, y = wcs.world_to_pixel_values(CENTER[0] + dx / 3600 / cosd, CENTER[1] + dy / 3600)
        img += peak * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sig**2)
    return img


def _frame(rng, shape=(256, 256), n_amp=4, noise=0.2, stripe=0.6, amp_off=(0.0, 0.5, -0.3, 0.4)):
    yy, xx = np.indices(shape)
    img = rng.normal(0, noise, shape).astype(np.float32) + 1.0
    truth = img.copy()
    stars = []
    for y, x in [(40 + 60 * i + rng.uniform(-5, 5), 30 + 65 * j + rng.uniform(-5, 5)) for i in range(3) for j in range(4)]:
        s = 80 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2))
        img += s.astype(np.float32)
        truth += s.astype(np.float32)
        stars.append((y, x))
    aw = shape[1] // n_amp
    for k in range(n_amp):
        img[:, k * aw:(k + 1) * aw] += rng.normal(0, stripe, (shape[0], 1)).astype(np.float32) + amp_off[k]
    mask = np.zeros(shape, dtype=bool)
    for y, x in stars:
        mask |= np.hypot(yy - y, xx - x) < 7
    return img, truth, mask, stars


def test_destripe_frame_removes_amp_stripes_and_offsets():
    from skyforge.level2 import destripe_frame

    rng = np.random.default_rng(3)
    img, truth, mask, stars = _frame(rng)
    c = Level2Config(min_pixels=20, window=0)
    out, info = destripe_frame(img, mask, c)
    assert info["rows_before"] / max(info["rows_after"], 1e-9) >= 10
    aw = img.shape[1] // 4
    meds = [float(np.median((out - truth)[:, k * aw:(k + 1) * aw][~mask[:, k * aw:(k + 1) * aw]])) for k in range(4)]
    assert max(meds) - min(meds) < 0.05                      # сдвиги усилителей сведены
    yy, xx = np.indices(img.shape)
    for y, x in stars:
        ap = np.hypot(yy - y, xx - x) <= 5
        ring = (np.hypot(yy - y, xx - x) > 8) & (np.hypot(yy - y, xx - x) < 12)
        f_out = (out[ap] - np.median(out[ring])).sum()
        f_true = (truth[ap] - np.median(truth[ring])).sum()
        assert abs(f_out / f_true - 1) < 0.005


def test_stack_rejects_cosmic_ray_keeps_star():
    from skyforge.level2 import stack

    rng = np.random.default_rng(5)
    shape = (60, 60)
    yy, xx = np.indices(shape)
    star = 50 * np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / (2 * 1.5**2))
    exps = []
    for k in range(4):
        d = (1 + star + rng.normal(0, 0.1, shape)).astype(np.float32)
        if k == 2:
            d[10, 40] += 40                                  # космический луч в одной экспозиции
        exps.append({"box": (0, 60, 0, 60), "data": d, "err": np.full(shape, 0.1, np.float32)})
    res = stack(exps, shape, Level2Config())
    assert abs(res["sci"][10, 40] - 1) < 0.3
    assert res["rejected"] >= 1
    assert abs(res["sci"][30, 30] - 51) < 0.5                # ядро звезды не отброшено
    assert res["nexp"][30, 30] == 4 and res["con"][30, 30] == 0b1111


def test_match_levels_recovers_offsets():
    from skyforge.level2 import match_levels

    rng = np.random.default_rng(7)
    offs = [0.0, 0.8, -0.5]
    exps = []
    for k, (o, box) in enumerate(zip(offs, [(0, 100, 0, 100), (0, 100, 50, 150), (40, 140, 20, 120)])):
        h, w = box[1] - box[0], box[3] - box[2]
        exps.append({"box": box, "data": (rng.normal(0, 0.2, (h, w)) + 3 + o).astype(np.float32)})
    c, pairs = match_levels(exps, np.ones((140, 150), dtype=bool))
    assert len(pairs) == 3
    assert np.allclose(np.asarray(c) - c[0], np.asarray(offs) - offs[0], atol=0.03)


def test_build_end_to_end_on_synthetic_exposures(tmp_path):
    """Три экспозиции с полосами, своими уровнями и лучом → мозаика на сетке «i2d»."""
    from skyforge.level2 import _build_filter

    rng = np.random.default_rng(11)
    g_shape, g_scale = (180, 180), 0.031
    gw = _wcs(g_shape, g_scale)
    truth = _sky(gw, g_shape, g_scale)
    noisy = truth + rng.normal(0, 0.03, g_shape)              # у настоящей i2d есть шум — по нему строится маска
    i2d = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(noisy.astype(np.float32), header=gw.to_header(), name="SCI"),
                        fits.ImageHDU(np.ones(g_shape, np.float32), name="WHT")])
    i2d_path = tmp_path / "x_i2d.fits"
    i2d.writeto(i2d_path)
    files = []
    for k, (shift, ped) in enumerate([((0, 0), 0.0), ((15.3, -9.7), 0.4), ((-12.1, 11.4), -0.3)]):
        shape = (160, 160)
        w = _wcs(shape, g_scale, shift)
        img = _sky(w, shape, g_scale) + rng.normal(0, 0.05, shape) + ped
        for a in range(4):
            img[:, a * 40:(a + 1) * 40] += rng.normal(0, 0.15, (shape[0], 1))
        if k == 1:
            img[30, 50] += 30                                # космический луч
        hdr = w.to_header()
        prim = fits.PrimaryHDU()
        prim.header.update(TELESCOP="JWST", INSTRUME="NIRCAM", FILTER="F090W")
        p = tmp_path / f"e{k}_cal.fits"
        fits.HDUList([prim, fits.ImageHDU(img.astype(np.float32), header=hdr, name="SCI"),
                      fits.ImageHDU(np.full(shape, 0.05, np.float32), name="ERR"),
                      fits.ImageHDU(np.zeros(shape, np.uint32), name="DQ")]).writeto(p)
        files.append(p)
    # Кадр маленький (усилитель — 40 столбцов): маска звёзд с запасом 5 px закрыла бы строки целиком,
    # а такие строки destripe не измеряет (их значение — от соседей, полосы там остаются).
    cfg = Config.model_validate({"target": "x", "level2": {"min_pixels": 10, "window": 0, "mask_grow_px": 1},
                                 "stars": {"halo_arcsec": 0.2, "max_star_area_arcsec2": 0.5},
                                 "background": {"mask_smooth_arcsec": 0.1, "tile_mask_dilate_arcsec": 0.2}})
    out = tmp_path / "x_l2.fits"
    info, _ = _build_filter(cfg, "F090W", i2d_path, files, out)
    with fits.open(out) as h:
        sci = h["SCI"].data
        assert sci.shape == g_shape and "CON" in h and "NEXP" in h
        assert WCS(h["SCI"].header).celestial.wcs.compare(gw.wcs, tolerance=1e-9)
    ok = np.isfinite(sci) & (truth < 1.05)                 # пустое небо; у ядер звёзд — разница интерполяции
    resid = (sci - truth)[ok]
    assert np.std(resid - np.median(resid)) < 0.08         # полосы (σ 0.15) и уровни (до 0.4) убраны; шум кадра 0.05
    assert np.nanmax(np.abs(sci - truth)[truth < 1.05] - abs(np.median(resid))) < 1.0  # луча нет
    assert info["exposures"] == 3 and info["rejected_pixels"] >= 1
    st = info["stars"]
    assert st and abs(st["flux_ratio"] - 1) < 0.02 and abs(st["fwhm_ratio"] - 1) < 0.05


# --- фаза 3g: астрометрия экспозиций ---

def _star_sky(wcs, shape, stars, sigma_px=1.1, noise=0.02, rng=None):
    img = np.full(shape, 1.0)
    px, py = wcs.all_world2pix(stars[:, 0], stars[:, 1], 0)
    r = 8
    for x, y in zip(px, py):
        xi, yi = int(round(x)), int(round(y))
        if r <= xi < shape[1] - r and r <= yi < shape[0] - r:   # звезда рисуется в окне — кадр 2048² за доли секунды
            yy, xx = np.mgrid[yi - r:yi + r + 1, xi - r:xi + r + 1]
            img[yi - r:yi + r + 1, xi - r:xi + r + 1] += 40 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma_px**2))
    if rng is not None:
        img += rng.normal(0, noise, shape)
    return img.astype(np.float32)


def test_tweak_recovers_shift_and_rotation():
    """Экспозиция со сдвигом 0.3 px и поворотом 0.01° относительно своего WCS восстанавливается до 0.02 px."""
    from skyforge.level2 import tweak_exposure

    rng = np.random.default_rng(42)
    shape = (2048, 2048)                                         # как детектор NIRCam: поворот 0.01° — до 0.18 px на краю
    true = _wcs(shape, 0.031)
    cosd = np.cos(np.radians(CENTER[1]))
    offs = rng.uniform(-30, 30, (40, 2))                         # звёзды в пределах кадра, ″
    stars = np.column_stack([CENTER[0] + offs[:, 0] / 3600 / cosd, CENTER[1] + offs[:, 1] / 3600])
    sci = _star_sky(true, shape, stars, rng=rng)
    # «Ошибочный» WCS из заголовка: тот же, но сдвинутый на 0.3 px и повёрнутый на 0.01°
    from skyforge.level2 import _rotated

    header = _rotated(true, np.array([-0.3, 0.2]), np.radians(-0.01))
    new, info = tweak_exposure(sci, header, stars, Level2Config())
    assert info["status"] in ("сдвиг", "поворот") and info["stars"] >= 8
    x1, y1 = new.all_world2pix(stars[:, 0], stars[:, 1], 0)
    x0, y0 = true.all_world2pix(stars[:, 0], stars[:, 1], 0)
    inside = (x0 > 10) & (x0 < 2038) & (y0 > 10) & (y0 < 2038)
    err = np.hypot(x1 - x0, y1 - y0)[inside]
    assert err.max() < 0.02, f"ошибка положения после подстройки до {err.max():.3f} px"
    xb, yb = header.all_world2pix(stars[:, 0], stars[:, 1], 0)
    assert np.hypot(xb - x0, yb - y0)[inside].max() > 0.3      # до подстройки было хуже


def test_tweak_flags_too_few_stars():
    from skyforge.level2 import tweak_exposure

    rng = np.random.default_rng(1)
    shape = (300, 300)
    w = _wcs(shape, 0.031)
    cosd = np.cos(np.radians(CENTER[1]))
    stars = np.array([[CENTER[0] + d / 3600 / cosd, CENTER[1]] for d in (-2.0, 0.0, 2.0)])
    sci = _star_sky(w, shape, stars, rng=rng)
    new, info = tweak_exposure(sci, w, stars, Level2Config())
    assert info["status"] == "мало звёзд" and new is w        # по трём звёздам не подгоняем


def test_overlap_residual_same_definition_for_map_and_number():
    from skyforge.level2 import overlap_residual

    rng = np.random.default_rng(3)
    shape = (256, 256)
    sci = np.zeros(shape, np.float32)
    e1 = {"box": (0, 256, 0, 256), "data": rng.normal(0, 1, shape).astype(np.float32)}
    e2 = {"box": (0, 256, 0, 256), "data": (rng.normal(0, 1, shape) + 0.3).astype(np.float32)}  # уровень +0.3σ
    nexp = np.full(shape, 2, np.int16)
    metric, mp, s1 = overlap_residual([e1, e2], sci, nexp, np.ones(shape, bool), 64)
    assert abs(s1 - 1) < 0.1
    assert abs(metric - float(np.nanmean(mp))) < 1e-9
    assert 0.1 < metric < 0.2                                    # среднее |0| и |0.3| ≈ 0.15σ


def test_decide_tweaks_rules():
    from skyforge.level2 import decide_tweaks

    c = Level2Config()
    def fit(dx, dy, prec, rb=0.3, ra=0.1, th=0.0):
        return {"stars": 25, "used": 25, "dx": dx, "dy": dy, "theta": th, "rot_deg": np.degrees(th),
                "precision": prec, "rms_before": rb, "rms_after": ra}
    names = [f"jw01_02101_00001_nrcb{k}_cal.fits" for k in (1, 2, 3, 4)] + \
            ["jw01_02101_00002_nrcblong_cal.fits", "jw01_02101_00003_nrcblong_cal.fits", "jw01_02101_00004_nrcblong_cal.fits"]
    infos = [fit(0.30, 0.20, 0.02), fit(0.31, 0.21, 0.02), fit(0.12, 0.05, 0.02), fit(0.5, 0.5, 0.05),
             fit(0.2, 0.0, 0.02, ra=0.1),                 # длинные: точная — исправить
             fit(0.3, 0.0, 0.04, rb=0.25, ra=0.25),       # неточная, остаток до 0.25 — не исправлять, но оставить
             fit(0.4, 0.0, 0.04, rb=0.45, ra=0.25)]       # неточная и остаток до > 0.3 — исключить
    decide_tweaks(infos, names, c)
    assert infos[0]["status"] == "сдвиг" and infos[1]["status"] == "сдвиг"
    assert infos[2]["status"].startswith("медиана группы (отличался")   # отличается от медианы > 0.05 px
    assert infos[3]["status"].startswith("медиана группы (своя подгонка неточна")
    med = np.median([[0.30, 0.20], [0.31, 0.21], [0.12, 0.05]], axis=0)
    assert np.allclose(infos[2]["apply"][:2], med) and infos[2]["apply"][2] == 0
    assert infos[4]["apply"] and not infos[5]["apply"] and not infos[5]["exclude"]
    assert infos[6]["exclude"] and infos[6]["status"] == "исключена"


def test_rotation_needs_enough_stars():
    from skyforge.level2 import measure_tweak, _rotated

    rng = np.random.default_rng(9)
    shape = (2048, 2048)
    true = _wcs(shape, 0.031)
    cosd = np.cos(np.radians(CENTER[1]))
    offs = rng.uniform(-30, 30, (12, 2))                       # 12 звёзд < 20 — поворот не применяется
    stars = np.column_stack([CENTER[0] + offs[:, 0] / 3600 / cosd, CENTER[1] + offs[:, 1] / 3600])
    sci = _star_sky(true, shape, stars, rng=rng)
    header = _rotated(true, np.array([-0.3, 0.2]), np.radians(-0.01))
    info = measure_tweak(sci, header, stars, Level2Config())
    assert info["used"] < 20 and info["theta"] == 0.0


def test_fwhm_check_tolerance_by_filter_fwhm():
    from skyforge.level2 import CHECK_ONLY, fwhm_check

    c = Level2Config()
    small = [{"fwhm_i2d": 1.9, "fwhm_ours": 1.9 * k} for k in (0.944, 1.052, 1.0)]   # FWHM < 2.5 px: ±8%
    big = [{"fwhm_i2d": 2.9, "fwhm_ours": 2.9 * k} for k in (0.922, 1.0, 0.96)]      # иначе ±5%
    assert fwhm_check(small, c)["passed"] == 3 and fwhm_check(small, c)["tol_percent"] == 8
    r = fwhm_check(big, c)
    assert r["tol_percent"] == 5 and r["passed"] == 2 and r["total"] == 3
    assert CHECK_ONLY <= set(Level2Config.model_fields)   # допуски не входят в ключ сборки
