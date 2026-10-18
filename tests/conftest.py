"""Синтетическое небо: одни и те же звёзды, снятые тремя «фильтрами» с разной сеткой.

F090W  — пиксель 0.031", без поворота;
F200W  — пиксель 0.031", поворот 12°, сдвиг опорной точки;
F470N  — пиксель 0.063", поворот −20° (в заголовке FILTER=F444W, PUPIL=F470N, как у NIRCam).
Данные пишутся big-endian, у каждого файла WHT=0 в углу и NaN-полоса по краю.
Слой CON делит кадр на две «плитки», у правой пьедестал выше на TILE_STEP;
у самой яркой звезды в F090W ядро выбито в NaN, как насыщение в настоящих i2d.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skyforge.config import Config
from skyforge.paths import Paths

CENTER = (150.0, -40.0)
RNG_SEED = 7
N_STARS = 14
MIN_SEPARATION = 0.6  # угловых секунд
NOISE = 0.5
# Несимметричная метка для теста ориентации: крупное яркое пятно в юго-восточном углу неба.
# Восток — +RA, юг — −Dec; на стандартной картинке (север вверху, восток слева) это низ слева.
MARKER_OFFSET_ARCSEC = (1.9, -1.9)  # (на восток, на север)
MARKER_FLUX, MARKER_SIGMA_ARCSEC = 3000.0, 0.25
TILE_STEP = 1.0  # 2σ шума: ступенька, которую видно глазом


def star_field() -> list[tuple[float, float, float]]:
    """(ra, dec, поток) — звёзды в пределах ±2.8" от центра, не ближе 0.6" друг к другу,
    чтобы центроид одной звезды не тянуло соседкой."""
    rng = np.random.default_rng(RNG_SEED)
    offsets: list[np.ndarray] = []
    while len(offsets) < N_STARS:
        d = rng.uniform(-2.8, 2.8, 2)
        if all(np.hypot(*(d - o)) >= MIN_SEPARATION for o in offsets):
            offsets.append(d)
    cosd = np.cos(np.radians(CENTER[1]))
    return [(CENTER[0] + dx / 3600 / cosd, CENTER[1] + dy / 3600, float(rng.uniform(200, 800)))
            for dx, dy in offsets]


def make_wcs(scale_arcsec: float, shape: tuple[int, int], rot_deg: float, crpix_shift: tuple[float, float]) -> WCS:
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(CENTER)
    w.wcs.crpix = [shape[1] / 2 + 0.5 + crpix_shift[0], shape[0] / 2 + 0.5 + crpix_shift[1]]
    s = scale_arcsec / 3600
    t = np.radians(rot_deg)
    w.wcs.cd = s * np.array([[-np.cos(t), np.sin(t)], [np.sin(t), np.cos(t)]])
    return w


def render(wcs: WCS, shape: tuple[int, int], scale_arcsec: float, nebula_gain: float, seed: int) -> np.ndarray:
    yy, xx = np.indices(shape, dtype=np.float64)
    img = np.full(shape, 5.0)
    # Туманность: кольцо радиусом 1.6" вокруг центра.
    ra, dec = wcs.pixel_to_world_values(xx, yy)
    r = np.hypot((ra - CENTER[0]) * np.cos(np.radians(CENTER[1])), dec - CENTER[1]) * 3600
    img += nebula_gain * 40 * np.exp(-0.5 * ((r - 1.6) / 0.35) ** 2)
    sigma = 0.07 / scale_arcsec  # PSF 0.07" в пикселях кадра
    for sra, sdec, flux in star_field():
        x, y = wcs.world_to_pixel_values(sra, sdec)
        img += flux * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / sigma**2)
    mra, mdec = marker_radec()
    mx, my = wcs.world_to_pixel_values(mra, mdec)
    ms = MARKER_SIGMA_ARCSEC / scale_arcsec
    img += MARKER_FLUX * np.exp(-0.5 * ((xx - mx) ** 2 + (yy - my) ** 2) / ms**2)
    img += np.random.default_rng(seed).normal(0, NOISE, shape)
    return img


def marker_radec() -> tuple[float, float]:
    return (CENTER[0] + MARKER_OFFSET_ARCSEC[0] / 3600 / np.cos(np.radians(CENTER[1])),
            CENTER[1] + MARKER_OFFSET_ARCSEC[1] / 3600)


def brightest_star_pixel(wcs: WCS) -> tuple[float, float]:
    ra, dec, _ = max(star_field(), key=lambda s: s[2])
    return wcs.world_to_pixel_values(ra, dec)


FILTERS = {
    # name: (FILTER, PUPIL, scale, shape, rot, shift, nebula)
    "F090W": ("F090W", "CLEAR", 0.031, (230, 250), 0.0, (0.0, 0.0), 0.6),
    "F200W": ("F200W", "CLEAR", 0.031, (240, 236), 12.0, (6.3, -4.1), 1.0),
    "F470N": ("F444W", "F470N", 0.063, (118, 124), -20.0, (-2.2, 1.7), 1.4),
}


def write_fits(path: Path, name: str, seed: int) -> None:
    filt, pupil, scale, shape, rot, shift, neb = FILTERS[name]
    wcs = make_wcs(scale, shape, rot, shift)
    data = render(wcs, shape, scale, neb, seed)
    con = np.where(np.arange(shape[1])[None, :] < shape[1] // 2, 3, 7).repeat(shape[0], axis=0).astype(np.int32)
    data[con == 7] += TILE_STEP
    data[:3, :] = np.nan
    wht = np.ones(shape, dtype=np.float32)
    wht[-6:, -6:] = 0
    if name == "F090W":
        x, y = brightest_star_pixel(wcs)
        yy, xx = np.indices(shape)
        core = np.hypot(xx - x, yy - y) < 2.5
        data[core] = np.nan
        wht[core] = 0
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="JWST", INSTRUME="NIRCAM", FILTER=filt, PUPIL=pupil, TARGPROP="TESTOBJ",
                          EFFEXPTM=1000.0, **{"DATE-BEG": "2022-06-03T00:00:00"})
    sci = fits.ImageHDU(data.astype(">f4"), header=wcs.to_header(), name="SCI")
    sci.header["BUNIT"] = "MJy/sr"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, sci, fits.ImageHDU(con[None], name="CON"), fits.ImageHDU(wht, name="WHT")]).writeto(path)


@pytest.fixture
def sky(tmp_path: Path) -> Paths:
    """Проект во временной папке с тремя FITS в data/inbox/testobj/."""
    root = tmp_path / "proj"
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("")
    for i, name in enumerate(FILTERS):
        write_fits(root / "data" / "inbox" / "testobj" / f"test_{name.lower()}_i2d.fits", name, seed=100 + i)
    return Paths(root)


@pytest.fixture
def cfg() -> Config:
    return Config.model_validate({
        "target": "Test Obj",
        "slug": "testobj",
        "mission": "jwst",
        "instrument": "nircam",
        "filters": ["F090W", "F200W", "F470N"],
        "reference": "F090W",
        "bin": 1,
        "stretch": {"method": "asinh", "black": 1.0, "white": 99.9, "a": 0.05},
        "color": {"saturation": 1.1},
        "finish": {"denoise": 0.3, "sharpen": 0.5, "north_up": True},
        # Кадр всего ~8″: ореол звезды по умолчанию (1.5″) накрыл бы почти всё поле.
        "stars": {"fill_cores": True, "halo_arcsec": 0.1, "max_star_area_arcsec2": 0.3},
        "background": {"subtract": True, "box_arcsec": 1.0, "mask_dilate_arcsec": 0.3, "mask_smooth_arcsec": 0.1,
                       "tile_mask_dilate_arcsec": 0.4, "tile_min_area_arcsec2": 0.2, "min_free": 0.2},
        "preview": {"max_size": 400, "checker_cell": 32, "zoom": 64},
    })
