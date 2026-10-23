"""Фаза 9: наземная миссия (mission: ground) — реестр полос и продукт ESO для конвейера."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from skyforge.config import Config
from skyforge.filters import effective_filter, normalize_filter_name, sort_by_wavelength, wavelength_um


def test_ground_filter_registry():
    assert [normalize_filter_name(x) for x in ("g_SDSS", "R", "u", "NB_659", "i_SDSS")] == ["g", "r", "u", "ha", "i"]
    assert effective_filter("r_SDSS") == "r" and effective_filter("F444W;F405N") == "F405N"
    assert wavelength_um("u", "ground") == pytest.approx(0.354) and wavelength_um("ha", "ground") == pytest.approx(0.659)
    assert sort_by_wavelength(["i", "ha", "u", "r", "g"], "ground") == ["u", "g", "r", "ha", "i"]
    assert normalize_filter_name("F150W") == "F150W"                  # имена JWST/HST — как раньше
    with pytest.raises(ValueError):
        normalize_filter_name("q_SDSS")


def test_ground_config_and_digest():
    c = Config(target="NGC 1365", mission="ground", instrument="OmegaCAM", archive_collection="FDS",
               filters=["u_SDSS", "g", "r", "i"], reference="g", coords="53.40155 -36.14040")
    assert c.mission == "ground" and c.instrument == "omegacam" and c.filters == ["u", "g", "r", "i"]
    jw = Config(target="X", filters=["F150W"])
    assert jw.digest() == Config(target="X", filters=["F150W"], archive_collection=None).digest()   # поле без значения


def _tile(path, data, wht=None):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [53.4, -36.14]
    w.wcs.crpix = [-100.0, 200.0]                      # вырезка: CRPIX далеко за краем, как у SODA
    w.wcs.cd = [[-5.5555e-5, 0], [0, 5.5555e-5]]
    prim = fits.PrimaryHDU()
    prim.header.update({"TELESCOP": "ESO-VST", "INSTRUME": "OMEGACAM", "EXPTIME": 42210.0, "PHOTZP": 0.0,
                        "PHOTSYS": "AB", "PSF_FWHM": 1.11, "ABMAGSAT": 15.4, "BUNIT": "ADU/s"})
    hdus = [prim, fits.CompImageHDU(data.astype(np.float32), header=w.to_header())]
    fits.HDUList(hdus).writeto(path)


def test_make_product_sci_wht_and_nan_mask(tmp_path):
    """Продукт ESO для конвейера: SCI с WCS вырезки, WHT; вес 0 (или 0 в данных без карты весов) → NaN при чтении."""
    from skyforge.eso import make_product
    from skyforge.fitsio import load_sci, read_info

    rng = np.random.default_rng(0)
    data = rng.normal(1e-12, 1e-13, (60, 80))
    data[:5] = 0.0                                     # вне покрытия тайла — нули
    _tile(tmp_path / "sci.fits.fz", data)
    meta = {"dp_id": "ADP.2020-08-26T11:45:32.315", "obs_collection": "FDS", "cutout": "53.4 -36.14 0.2"}
    info = make_product(tmp_path / "sci.fits.fz", None, "g", meta, tmp_path / "ngc1365_g.fits")
    a, w = load_sci(tmp_path / "ngc1365_g.fits")
    assert np.isnan(a[:5]).all() and np.isfinite(a[5:]).all() and info["empty_percent"] == pytest.approx(5 / 60 * 100, 1e-3)
    assert np.allclose(w.pixel_to_world_values(0, 0), WCS(fits.getheader(tmp_path / "sci.fits.fz", 1)).pixel_to_world_values(0, 0))
    ri = read_info(tmp_path / "ngc1365_g.fits")
    assert ri.filter == "g" and ri.exptime == 42210.0 and ri.instrument == "OMEGACAM"
    with fits.open(tmp_path / "ngc1365_g.fits") as h:
        assert h[0].header["LICENSE"] == "CC BY 4.0" and h[0].header["ESO_DPID"] == meta["dp_id"]
    # с картой весов: вес 0 — пусто
    wht = np.ones_like(data)
    wht[:, :3] = 0
    _tile(tmp_path / "wht.fits.fz", wht)
    make_product(tmp_path / "sci.fits.fz", tmp_path / "wht.fits.fz", "r", meta, tmp_path / "ngc1365_r.fits")
    b, _ = load_sci(tmp_path / "ngc1365_r.fits")
    assert np.isnan(b[:, :3]).all() and np.isfinite(b[10:, 3:]).all()
