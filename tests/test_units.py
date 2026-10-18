"""Отдельные кирпичи: фильтры, конфиг, выбор продуктов MAST, WCS при биннинге, растяжка, цвет."""

from __future__ import annotations

import numpy as np
import pytest
from astropy.wcs import WCS, Sip
from pydantic import ValidationError

from skyforge.align import bin_image, binned_wcs, inscribed_rect
from skyforge.compose import assign_colors, blend, neutralize
from skyforge.config import ColorConfig, Config, StretchConfig
from skyforge.fetch import choose_observations, choose_product
from skyforge.filters import effective_filter, wavelength_um
from skyforge.stretch import stretch_array

from conftest import make_wcs


# --- фильтры ---

@pytest.mark.parametrize("raw,expected", [
    ("F444W;F405N", "F405N"),
    ("F444W;F470N", "F470N"),
    ("CLEAR;F090W", "F090W"),
    ("F150W2;F162M", "F162M"),
    ("F187N", "F187N"),
    ("CLEAR", None),
])
def test_effective_filter(raw, expected):
    assert effective_filter(raw) == expected


def test_wavelengths():
    assert wavelength_um("F090W", "jwst") == pytest.approx(0.90)
    assert wavelength_um("F1130W", "jwst") == pytest.approx(11.3)
    assert wavelength_um("F656N", "hst", "WFC3/UVIS") == pytest.approx(0.656)
    assert wavelength_um("F160W", "hst", "WFC3/IR") == pytest.approx(1.60)


# --- конфиг ---

def test_config_rejects_typos():
    with pytest.raises(ValidationError):
        Config.model_validate({"target": "X", "stretch": {"blak": 1}})


def test_config_reference_must_be_in_filters():
    with pytest.raises(ValidationError):
        Config.model_validate({"target": "X", "filters": ["F090W"], "reference": "F444W"})


def test_config_slug_and_with_param():
    c = Config.model_validate({"target": "NGC 3132", "filters": ["f090w"], "color": {"overrides": {"f090w": "FF8040"}}})
    assert c.slug == "ngc3132"
    assert c.filters == ["F090W"]
    assert c.color.overrides == {"F090W": "#ff8040"}
    c2 = c.with_param("stretch.a", 0.1)
    assert c2.stretch.a == 0.1 and c.stretch.a == 0.02
    with pytest.raises(ValueError):
        c.with_param("stretch.nope", 1)


def test_stretch_per_filter_override():
    s = StretchConfig(white=99.8, per_filter={"f187n": {"white": 99.95}})
    assert s.for_filter("F187N").white == 99.95
    assert s.for_filter("F090W").white == 99.8


# --- выбор в MAST (по форме реального ответа для NGC 3132) ---

def test_choose_observations_prefers_public_and_longer():
    rows = [
        {"obs_id": "a", "obsid": "1", "filters": "F444W;F405N", "t_exptime": 2319.0, "dataRights": "PUBLIC", "t_obs_release": 59773.6, "proposal_id": "2733"},
        {"obs_id": "b", "obsid": "2", "filters": "F405N", "t_exptime": 9999.0, "dataRights": "EXCLUSIVE_ACCESS", "t_obs_release": 61000.0, "proposal_id": "9"},
        {"obs_id": "c", "obsid": "3", "filters": "F090W", "t_exptime": 100.0, "dataRights": "PUBLIC", "t_obs_release": None, "proposal_id": None},
        {"obs_id": "d", "obsid": "4", "filters": "F090W", "t_exptime": 1460.0, "dataRights": "PUBLIC", "t_obs_release": None, "proposal_id": None},
    ]
    best = choose_observations(rows)
    assert best["F405N"].obs_id == "a"
    assert best["F090W"].obs_id == "d"
    assert best["F405N"].release == "2022-07-13"


def test_choose_product_skips_level2_i2d():
    obs = "jw02733-o001_t001_nircam_clear-f356w"
    prods = [
        {"productFilename": "jw02733001001_02105_00002_nrcblong_i2d.fits", "calib_level": 2, "size": 120, "productSubGroupDescription": "I2D", "productType": "SCIENCE"},
        {"productFilename": f"{obs}_segm.fits", "calib_level": 3, "size": 50, "productSubGroupDescription": "SEGM", "productType": "SCIENCE"},
        {"productFilename": f"{obs}_i2d.fits", "calib_level": 3, "size": 155, "productSubGroupDescription": "I2D", "productType": "SCIENCE"},
    ]
    assert choose_product(prods, obs, "jwst", 3)["productFilename"] == f"{obs}_i2d.fits"
    assert choose_product(prods, "other-obs", "jwst", 3) is None


# --- геометрия ---

@pytest.mark.parametrize("factor", [2, 3, 4])
def test_binned_wcs_points_to_block_centres(factor):
    w = make_wcs(0.031, (240, 240), 17.0, (3.3, -1.2))
    wb = binned_wcs(w, factor, (240, 240))
    for i in (0, 7, 31):
        # центр нового пикселя (i, i) = центр блока исходных пикселей
        orig = factor * i + (factor - 1) / 2
        assert np.allclose(wb.pixel_to_world_values(i, i), w.pixel_to_world_values(orig, orig), atol=1e-12)


def test_binned_wcs_with_sip():
    w = make_wcs(0.031, (240, 240), 5.0, (0, 0))
    w.wcs.ctype = ["RA---TAN-SIP", "DEC--TAN-SIP"]
    a = np.zeros((3, 3)); a[2, 0] = 2e-5; a[0, 2] = -1e-5; a[1, 1] = 3e-6
    b = np.zeros((3, 3)); b[0, 2] = 1.5e-5; b[2, 0] = 4e-6
    w.sip = Sip(a, b, None, None, w.wcs.crpix)
    w.wcs.set()
    wb = binned_wcs(w, 4, (240, 240))
    for i in (0, 10, 50):
        orig = 4 * i + 1.5
        assert np.allclose(wb.all_pix2world([[i, i]], 0), w.all_pix2world([[orig, orig]], 0), atol=1e-11)


def test_bin_image_ignores_nan():
    data = np.arange(16, dtype=np.float32).reshape(4, 4)
    data[0, 0] = np.nan
    w = make_wcs(0.031, (4, 4), 0, (0, 0))
    out, _ = bin_image(data, w, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(np.mean([1, 4, 5]))


def test_inscribed_rect_has_full_edges():
    yy, xx = np.indices((200, 200))
    # Ромб — повёрнутый квадрат: углы рамки пустые.
    mask = (np.abs(yy - 100) + np.abs(xx - 100)) < 90
    r0, r1, c0, c1 = inscribed_rect(mask, 1.0)
    assert mask[r0:r1, c0:c1].all()
    assert (r1 - r0) * (c1 - c0) > 0.4 * mask.sum()


# --- растяжка и цвет ---

def test_stretch_maps_percentiles_and_keeps_nan():
    rng = np.random.default_rng(0)
    a = rng.normal(10, 1, (300, 300)).astype(np.float32)
    a[0, :] = np.nan
    s = StretchConfig(black=1, white=99, a=0.1)
    y, st = stretch_array(a, s)
    assert np.isnan(y[0]).all()
    assert np.nanmin(y) == 0 and np.nanmax(y) == pytest.approx(1)
    assert st["black_value"] < st["white_value"]
    order = np.argsort(a[5])
    assert np.all(np.diff(y[5][order]) >= 0)  # кривая монотонна


def test_colors_by_wavelength_and_override():
    c = ColorConfig()
    cols = assign_colors(["F444W", "F090W", "F200W"], "jwst", "nircam", c)
    assert np.allclose(cols["F090W"], (0, 0, 1))
    assert np.allclose(cols["F200W"], (0, 1, 0))
    assert np.allclose(cols["F444W"], (1, 0, 0))
    c2 = ColorConfig(overrides={"F200W": "#ff8040"})
    assert np.allclose(assign_colors(["F090W", "F200W"], "jwst", "nircam", c2)["F200W"], (1, 128 / 255, 64 / 255))


@pytest.mark.parametrize("mode", ["screen", "average"])
def test_blend_three_filters_is_plain_rgb(mode):
    L = {f: np.full((4, 4), v, dtype=np.float32) for f, v in (("F090W", 0.2), ("F200W", 0.5), ("F444W", 0.9))}
    c = ColorConfig(blend=mode)
    rgb = blend(L, assign_colors(list(L), "jwst", "nircam", c), c)
    assert np.allclose(rgb[0, 0], (0.9, 0.5, 0.2), atol=1e-6)


def test_neutralize_makes_background_grey():
    rgb = np.zeros((100, 100, 3), dtype=np.float32) + np.array([0.1, 0.3, 0.1], dtype=np.float32)
    rgb[40:60, 40:60] = 0.9
    out, bg = neutralize(rgb, 10)
    assert np.allclose(out[0, 0], 0, atol=1e-6)
    assert bg[1] == pytest.approx(0.3)


def test_finish_keeps_brightness():
    """Денойз и резкость не должны гасить картинку (так было с skimage.unsharp_mask)."""
    from skyforge.config import FinishConfig
    from skyforge.finish import finish_array

    rng = np.random.default_rng(1)
    yy, xx = np.indices((200, 220))
    blob = np.exp(-((yy - 100) ** 2 + (xx - 110) ** 2) / (2 * 40**2))
    rgb = np.stack([blob, blob * 0.7, blob * 0.4], axis=-1).astype(np.float32)
    rgb += rng.normal(0, 0.01, rgb.shape).astype(np.float32)
    rgb = np.clip(rgb, 0, 1)
    out = finish_array(rgb, FinishConfig(denoise=0.3, sharpen=0.5))
    assert out.dtype == np.float32
    assert np.percentile(out, 99) == pytest.approx(np.percentile(rgb, 99), rel=0.1)
    assert np.median(out) == pytest.approx(np.median(rgb), abs=0.03)


def test_lift_brightens_midtones_and_keeps_hue():
    from skyforge.compose import lift

    rgb = np.array([[[0.0, 0.0, 0.0], [0.2, 0.1, 0.05], [0.9, 0.8, 0.7]]], dtype=np.float32)
    out = lift(rgb, 0.25)
    assert np.allclose(out[0, 0], 0)
    assert out[0, 1].sum() > rgb[0, 1].sum()  # средние тона светлее
    assert np.allclose(out[0, 1] / out[0, 1, 0], rgb[0, 1] / rgb[0, 1, 0], atol=1e-5)  # оттенок тот же
    assert out.max() <= 1
    assert lift(rgb, None) is rgb


# --- фаза 3: ядра звёзд и фон ---

def test_core_holes_filled_but_edges_and_big_gaps_kept():
    from skyforge.stars import fill_rings, find_core_holes

    yy, xx = np.indices((60, 60))
    a = (100 * np.exp(-((yy - 30) ** 2 + (xx - 30) ** 2) / 20)).astype(np.float32)
    rim_max = a[np.hypot(yy - 30, xx - 30) >= 3].max()
    a[np.hypot(yy - 30, xx - 30) < 3] = np.nan  # насыщенное ядро
    a[:, :2] = np.nan                            # пустой край мозаики
    a[50:52, 5:45] = np.nan                      # длинный пропуск, не ядро
    holes = find_core_holes(a, max_area_px=40)
    assert holes[30, 30] and not holes[10, 0] and not holes[50, 20]
    out = fill_rings(a, holes)
    assert np.isfinite(out[np.hypot(yy - 30, xx - 30) < 3]).all()
    assert out[30, 30] >= rim_max - 1e-4
    assert np.isnan(out[:, :2]).all() and np.isnan(out[50, 20])


def test_tile_pedestal_removes_step_and_keeps_nebula():
    from skyforge.background import tile_pedestal
    from skyforge.config import BackgroundConfig

    rng = np.random.default_rng(3)
    yy, xx = np.indices((200, 200))
    nebula = 30 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 15**2))
    tiles = np.where(xx < 120, 0, 1).astype(np.int32)  # граница плиток проходит сквозь туманность
    a = (5 + nebula + rng.normal(0, 1, (200, 200)) + np.where(tiles == 1, 1.5, 0)).astype(np.float32)
    c = BackgroundConfig(subtract=True, tile_mask_dilate_arcsec=0.5, tile_min_area_arcsec2=0.1, mask_smooth_arcsec=0.1)
    ped, info = tile_pedestal(a, tiles, c, scale_arcsec=0.05)
    out = a - ped
    assert info == {"tiles": 2, "measured": 2}
    step = np.median(out[:, 150:][tiles[:, 150:] == 1]) - np.median(out[:, :50])
    assert abs(step) < 0.15  # было 1.5
    assert out[100, 100] - np.median(out[:, :50]) == pytest.approx(30, abs=1.5)  # туманность не съедена


def test_with_param_creates_filter_weight():
    c = Config.model_validate({"target": "X", "filters": ["F212N"]})
    assert c.with_param("color.weights.F212N", 0.5).color.weights == {"F212N": 0.5}
