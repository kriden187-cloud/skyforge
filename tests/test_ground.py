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


def test_plateau_cores_saturated_star_becomes_hole():
    """Плато насыщения (вершина срезана на одном уровне) — дыра NaN; ненасыщенная звезда и широкое ядро галактики
    (соседний пиксель ниже пика больше чем на tol) — нет."""
    from skyforge.config import PlateauConfig
    from skyforge.stars import plateau_cores

    rng = np.random.default_rng(1)
    yy, xx = np.indices((200, 200))
    a = rng.normal(0, 1.0, (200, 200))
    sat = 5e4 * np.exp(-((yy - 50) ** 2 + (xx - 50) ** 2) / (2 * 2.0 ** 2))
    a += np.minimum(sat, 3e4)                                           # насыщенная звезда: плато 3e4
    a += 2e3 * np.exp(-((yy - 150) ** 2 + (xx - 60) ** 2) / (2 * 1.5 ** 2))   # обычная звезда
    a += 4e3 * np.exp(-((yy - 120) ** 2 + (xx - 150) ** 2) / (2 * 12.0 ** 2)) \
        + 3e3 * np.exp(-((yy - 120) ** 2 + (xx - 150) ** 2) / (2 * 1.2 ** 2))  # ядро галактики: острый центр на диске
    plateau_px = int((sat >= 3e4).sum())
    out, info = plateau_cores(a.astype(np.float32), PlateauConfig(saturation=None))
    assert info["plateaus"] == 1 and np.isnan(out[50, 50]) and info["largest_px"] == plateau_px
    assert np.isfinite(out[150, 60]) and np.isfinite(out[120, 150]) and np.isnan(out).sum() == info["pixels"]
    # след растекания заряда: столбец 82–95% уровня плато (не плоский — правило tol его не берёт) с каймой ниже неба
    a[54:80, 49:52] = 3e4 * (0.82 + 0.13 * rng.random((26, 3)))
    a[54:80, 47:49] = -500                                              # слева кайма 2 px — шире grow_px
    a[54:80, 52] = 3e4 * (0.45 + 0.1 * rng.random(26))                 # справа кайма за пикселем ниже порога (как у
    a[54:80, 53:55] = -500                                              # NGC 1365 в r): он в пределах grow_px
    out, info = plateau_cores(a.astype(np.float32), PlateauConfig(saturation=None))
    assert np.isfinite(out[70, 50])                                     # без порога насыщения след остаётся
    out, info = plateau_cores(a.astype(np.float32), PlateauConfig())
    assert np.isnan(out[54:80, 47:55]).all() and info["bleed_px"] >= 26 * 3   # след и вся кайма — дыра
    assert info["rim_px"] == 26 * 3 and info["plateaus"] == 1
    assert info["saturation_level"] == pytest.approx(0.8 * 3e4, rel=0.01)
    assert np.isfinite(out[150, 60]) and np.isfinite(out[120, 150])    # ненасыщенная звезда и ядро галактики — нет
