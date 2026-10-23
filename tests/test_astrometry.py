"""Фаза 8: астрометрия экспозиций уровня 2 по Gaia (skyforge.astrometry) — синтетика."""

from __future__ import annotations

import math

import numpy as np
from astropy.wcs import WCS

from skyforge import astrometry as A


def _tan(ra: float = 83.85, dec: float = -5.04, scale: float = 0.031, rot_deg: float = 30.0) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = [1024.5, 1024.5]
    t = math.radians(rot_deg)
    s = scale / 3600
    w.wcs.cd = [[-s * math.cos(t), s * math.sin(t)], [s * math.sin(t), s * math.cos(t)]]
    w.wcs.set()
    return w


def test_apply_pointing_moves_sky_as_model():
    """Точка неба p → c + R(θ)(p − c) + s в касательной плоскости поля — для любого пикселя детектора."""
    w = _tan()
    ra0, dec0 = 83.85167, -5.03889
    corr = {"origin": [ra0, dec0], "shift": [1.18, -0.27], "theta": math.radians(0.004), "center": [20.0, -40.0]}
    new = A.apply_pointing(w, corr)
    px = np.array([[0, 0], [2047, 0], [0, 2047], [2047, 2047], [700, 1300]], dtype=float)
    p = A._plane(*w.all_pix2world(px[:, 0], px[:, 1], 0), ra0, dec0)
    q = A._plane(*new.all_pix2world(px[:, 0], px[:, 1], 0), ra0, dec0)
    th, c = corr["theta"], np.array(corr["center"])
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    want = c + (p - c) @ rot.T + np.array(corr["shift"])
    assert np.abs(q - want).max() < 3e-4          # ″: 0,3 mas — 0,01 px коротких волн


def _star(img: np.ndarray, x: float, y: float, amp: float, sigma: float) -> None:
    yy, xx = np.indices(img.shape)
    img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma**2))
    img += 0.002 * amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (6 * sigma) ** 2))   # крылья


def test_detect_stars_normal_and_saturated():
    """Обычная звезда — центроид гауссом; насыщенная (ядро пустое, флаг SATURATED) — центр по симметрии крыльев."""
    rng = np.random.default_rng(1)
    img = rng.normal(0, 1.0, (300, 300)).astype(np.float64)
    _star(img, 80.37, 90.71, 200, 0.9)
    _star(img, 200.62, 180.24, 2e5, 0.9)
    sat = img > 3e4
    img[sat] = np.nan
    err = np.ones_like(img)
    st = A.detect_stars(img.astype(np.float32), err.astype(np.float32), sat)
    normal = st[st[:, 3] == 0]
    sats = st[st[:, 3] == 1]
    d1 = np.hypot(normal[:, 0] - 80.37, normal[:, 1] - 90.71).min()
    d2 = np.hypot(sats[:, 0] - 200.62, sats[:, 1] - 180.24).min()
    assert d1 < 0.05 and d2 < 0.05


def test_coarse_needs_clear_peak():
    rng = np.random.default_rng(2)
    g = rng.uniform(-200, 200, (60, 2))
    shift = np.array([1.18, -0.27])
    p = np.vstack([g[:20] - shift + rng.normal(0, 0.01, (20, 2)), rng.uniform(-200, 200, (500, 2))])
    off, votes, second = A._coarse(p, g)
    assert np.hypot(*(off - shift)) < 0.02 and votes >= 2 * second
    assert A._coarse(rng.uniform(-200, 200, (500, 2)), g) is None      # одни случайные совпадения — не сдвиг


def test_solve_recovers_exposure_shifts():
    """Три экспозиции: пары одних звёзд и звёзды Gaia — сдвиги восстановлены с точностью ошибок центроидов."""
    rng = np.random.default_rng(3)
    true = rng.uniform(-60, 60, (300, 2))
    shifts = np.array([[0.02, -0.01], [1.2, -0.27], [0.05, 0.03]])
    pos, e_of, star = [], [], []
    for e, s in enumerate(shifts):
        sel = rng.choice(len(true), 200, replace=False)
        pos += list(true[sel] - s + rng.normal(0, 0.003, (200, 2)))
        e_of += [e] * 200
        star += list(sel)
    pos, e_of, star = np.array(pos), np.array(e_of), np.array(star)
    obs = []
    for a in range(len(pos)):
        for b in np.flatnonzero((star == star[a]) & (e_of > e_of[a])):
            obs.append((a, b, e_of[a], e_of[b], pos[b] - pos[a], 0.003 * math.sqrt(2)))
    for a in np.flatnonzero(star < 15):                                # Gaia: 15 звёзд
        obs.append((a, None, e_of[a], None, true[star[a]] - pos[a], math.hypot(0.003, 0.001)))
    centers = np.array([pos[e_of == e].mean(axis=0) for e in range(3)])
    sol, cv, cv_rel, nres = A._solve(obs, 3, np.zeros(3, dtype=bool), centers, pos, np.full(len(pos), 0.003))
    assert np.abs(sol[:, :2] - shifts).max() < 0.002                   # ″
    assert np.all(np.sqrt(cv) < 0.002) and np.all(np.sqrt(cv_rel) <= np.sqrt(cv) + 1e-9)


def test_gaia_at_drops_masked_two_parameter_solutions():
    """2-параметрические решения Gaia (собственное движение и RUWE под маской) не берутся — и из запроса, и из кеша."""
    from astropy.table import MaskedColumn, Table

    def col(v, mask):
        return MaskedColumn(np.array(v, dtype=float), mask=np.array(mask))

    t = Table({"ra": [83.85, 83.86], "dec": [-5.04, -5.05], "ra_error": [0.1, 0.1], "dec_error": [0.1, 0.1],
               "pmra": col([1.0, 0.0], [False, True]), "pmdec": col([-1.0, 0.0], [False, True]),
               "pmra_error": col([0.05, 0.0], [False, True]), "pmdec_error": col([0.05, 0.0], [False, True]),
               "ruwe": col([1.0, 0.0], [False, True]), "ref_epoch": [2016.0, 2016.0]})
    ra, dec, err = A.gaia_at(t, 2026.0)
    assert len(ra) == 1 and abs(dec[0] - (-5.04 - 1.0 * 10 / 3.6e6)) < 1e-12


def test_apply_pointing_rotates_pc_cdelt_header():
    """Заголовок PC+CDELT: поворот не теряется (у wcslib PC главнее CD)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [83.85, -5.04]
    w.wcs.crpix = [1024.5, 1024.5]
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    w.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
    w.wcs.set()
    ra0, dec0 = 83.85167, -5.03889
    corr = {"origin": [ra0, dec0], "shift": [0.0, 0.0], "theta": math.radians(0.05), "center": [0.0, 0.0]}
    new = A.apply_pointing(w, corr)
    p = A._plane(*w.all_pix2world([0.0], [0.0], 0), ra0, dec0)
    q = A._plane(*new.all_pix2world([0.0], [0.0], 0), ra0, dec0)
    th = corr["theta"]
    want = p @ np.array([[math.cos(th), math.sin(th)], [-math.sin(th), math.cos(th)]])
    assert np.abs(q - want).max() < 3e-4 and np.abs(q - p).max() > 0.01
