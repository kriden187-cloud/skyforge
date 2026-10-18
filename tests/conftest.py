"""Синтетическое небо: одни и те же звёзды, снятые тремя «фильтрами» с разной сеткой.

F090W  — пиксель 0.031", без поворота;
F200W  — пиксель 0.031", поворот 12°, сдвиг опорной точки;
F470N  — пиксель 0.063", поворот −20° (в заголовке FILTER=F444W, PUPIL=F470N, как у NIRCam).
Данные пишутся big-endian, у каждого файла WHT=0 в углу и NaN-полоса по краю.
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
    img += np.random.default_rng(seed).normal(0, 0.5, shape)
    return img


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
    data[:3, :] = np.nan
    wht = np.ones(shape, dtype=np.float32)
    wht[-6:, -6:] = 0
    primary = fits.PrimaryHDU()
    primary.header.update(TELESCOP="JWST", INSTRUME="NIRCAM", FILTER=filt, PUPIL=pupil, TARGPROP="TESTOBJ",
                          EFFEXPTM=1000.0, **{"DATE-BEG": "2022-06-03T00:00:00"})
    sci = fits.ImageHDU(data.astype(">f4"), header=wcs.to_header(), name="SCI")
    sci.header["BUNIT"] = "MJy/sr"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, sci, fits.ImageHDU(wht, name="WHT")]).writeto(path)


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
        "preview": {"max_size": 400, "checker_cell": 32, "zoom": 64},
    })
