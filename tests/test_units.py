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


def test_shadows_brightens_midtones_and_keeps_hue():
    from skyforge.compose import shadows

    rgb = np.array([[[0.0, 0.0, 0.0], [0.2, 0.1, 0.05], [0.9, 0.8, 0.7]]], dtype=np.float32)
    out = shadows(rgb, 4)
    assert np.allclose(out[0, 0], 0)
    assert out[0, 1].sum() > rgb[0, 1].sum()  # средние тона светлее
    assert np.allclose(out[0, 1] / out[0, 1, 0], rgb[0, 1] / rgb[0, 1, 0], atol=1e-5)  # оттенок тот же
    assert out.max() <= 1
    assert shadows(rgb, 0) is rgb
    # больше — сильнее
    assert shadows(rgb, 7)[0, 1].sum() > shadows(rgb, 4)[0, 1].sum() > shadows(rgb, 2)[0, 1].sum()


def test_old_lift_key_gives_hint():
    with pytest.raises(ValidationError, match="shadows: 4"):
        Config.model_validate({"target": "X", "color": {"lift": 0.25}})


def test_star_desaturation_touches_stars_not_nebula():
    from skyforge.compose import desaturate_stars
    from skyforge.config import StarsConfig
    from skyforge.stars import StarMask, star_mask, star_weight

    rng = np.random.default_rng(5)
    yy, xx = np.indices((300, 300))
    a = rng.normal(0, 1, (300, 300)).astype(np.float32)
    a += 40 * np.exp(-((yy - 150) ** 2 + (xx - 200) ** 2) / (2 * 40**2))  # плотная туманность
    a += 300 * np.exp(-((yy - 60) ** 2 + (xx - 60) ** 2) / (2 * 1.5**2))  # звезда поля
    s = StarsConfig(halo_arcsec=0.4, max_star_area_arcsec2=2.0, local_window_arcsec=2.0)
    m = star_weight({"F090W": star_mask(a, s, 0.05)}, "F090W", s, 0.05)
    assert m.weight[60, 60] > 0.9 and m.weight[60, 66] > 0.3   # ядро и ореол в маске
    assert m.weight[150, 200] == 0 and m.weight[150, 170] == 0  # плотная туманность исключена

    # Цветная картинка: зелёная звезда на пустом небе и такая же — на оранжевом волокне.
    rgb = np.zeros((300, 300, 3), dtype=np.float32)
    fil = (np.abs(xx - 230) < 30)[..., None] * np.array([0.5, 0.3, 0.05], dtype=np.float32)  # волокно
    rgb += fil
    for cy, cx in ((60, 60), (60, 230)):
        star = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))[..., None]
        rgb += star * np.array([0.3, 0.7, 0.3], dtype=np.float32)
    rgb = np.clip(rgb, 0, 1)
    w = np.zeros((300, 300), dtype=np.float32)
    w[np.hypot(yy - 60, xx - 60) < 10] = 1
    w[np.hypot(yy - 60, xx - 230) < 10] = 1
    det = np.zeros((300, 300), dtype=np.float32)  # фильтр поиска: звёзды есть, волокна почти нет
    for cy, cx in ((60, 60), (60, 230)):
        det += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0**2))
    out = desaturate_stars(rgb, StarMask(w, 40, "F090W"), 0.0, det, nebula_level=0.05)
    assert np.ptp(out[60, 60]) < 0.02                                # звезда на небе серая
    assert np.ptp(out[60, 230]) < 0.5 * np.ptp(rgb[60, 230])         # ядро на волокне обесцвечено
    assert np.allclose(out[69, 230], rgb[69, 230], atol=0.03)         # само волокно у края ореола — того же цвета
    assert np.allclose(out[60, 222], rgb[60, 222], atol=0.03)         # и внутри маски, рядом с ядром
    assert np.allclose(out[150, 230], rgb[150, 230])                  # волокно вне маски не тронуто


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
    assert info["tiles"] == 2 and info["measured"] == 2
    step = np.median(out[:, 150:][tiles[:, 150:] == 1]) - np.median(out[:, :50])
    assert abs(step) < 0.15  # было 1.5
    assert out[100, 100] - np.median(out[:, :50]) == pytest.approx(30, abs=1.5)  # туманность не съедена


def test_with_param_creates_filter_weight():
    c = Config.model_validate({"target": "X", "filters": ["F212N"]})
    assert c.with_param("color.weights.F212N", 0.5).color.weights == {"F212N": 0.5}


def test_union_grid_extends_lattice_without_shift(tmp_path):
    """3h п.4: сетка по объединению покрытий — та же решётка, расширенная на целое число пикселей;
    внутри прежней сетки репроекция даёт те же пиксели."""
    from astropy.io import fits
    from skyforge.align import reproject_nan_safe, union_grid

    grid = make_wcs(0.1, (200, 200), 0.0, (0, 0))
    other = make_wcs(0.1, (200, 200), 20.0, (60, -40))              # второй фильтр сдвинут и повёрнут
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, (200, 200)).astype(np.float32)
    p = tmp_path / "other.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, header=other.to_header(), name="SCI")]).writeto(p)
    w2, shape2 = union_grid(grid, (200, 200), [p])
    assert shape2[0] > 200 and shape2[1] > 200
    off = np.round(np.asarray(w2.wcs.crpix) - np.asarray(grid.wcs.crpix)).astype(int)   # (x, y)
    assert np.allclose(np.asarray(w2.wcs.crpix) - np.asarray(grid.wcs.crpix), off)
    a = reproject_nan_safe(data, other, grid, (200, 200), "bilinear")
    b = reproject_nan_safe(data, other, w2, shape2, "bilinear")
    inner = b[off[1]:off[1] + 200, off[0]:off[0] + 200]
    ok = np.isfinite(a)
    assert np.array_equal(np.isfinite(inner), ok) and np.allclose(inner[ok], a[ok], atol=1e-6, rtol=0)
    assert np.isfinite(b).sum() > np.isfinite(a).sum()                 # снаружи прежней сетки тоже есть данные


def test_blend_renormalizes_where_filters_missing():
    """Где фильтра нет, цвет собирается из имеющихся: край не темнеет и не уходит в цвет отсутствующего."""
    c = ColorConfig()
    a = np.full((4, 4), 0.6, dtype=np.float32)
    b = a.copy()
    b[:, 2:] = np.nan                                                   # второго фильтра справа нет
    colors = {"A": (1.0, 0.5, 0.0), "B": (0.0, 0.5, 1.0)}
    rgb = blend({"A": a, "B": b}, colors, c)
    assert np.allclose(rgb[0, 0], [0.6, 0.6, 0.6], atol=1e-6)          # оба фильтра: серый той же яркости
    assert np.allclose(rgb[0, 3], [0.6, 0.6, 0.0], atol=1e-6)          # только A: его цвет, без потемнения G
    rgb_none = blend({"A": np.full((2, 2), np.nan, np.float32), "B": np.full((2, 2), np.nan, np.float32)}, colors, c)
    assert (rgb_none == 0).all()


def test_crops_on_sky_follow_grid_change():
    """Кроп с центром на небе показывает то же место на любой сетке: при расширении кадра на (dx, dy)
    пиксели центра сдвигаются ровно на столько же (3h: в 3g кропы в пикселях съехали при смене кадра)."""
    from skyforge.align import resolve_crops
    from skyforge.config import ReportCrop

    a = make_wcs(0.063, (2000, 2000), 35.0, (0, 0))
    b = a.deepcopy()
    b.wcs.crpix = np.asarray(b.wcs.crpix) + [236, 300]                  # тот же кадр внутри расширенного
    ra, dec = a.pixel_to_world_values(812, 1500)
    c = ReportCrop(name="sky", ra=float(ra), dec=float(dec), size=400)
    ca = resolve_crops([c], a, 2000, 1)[0]
    cb = resolve_crops([c], b, 2400, 1)[0]
    assert (ca.x, ca.y) == (812, 2000 - 1 - 1500)
    assert cb.x - ca.x == 236 and cb.y - ca.y == (2400 - 2000) - 300     # картинка: y вниз, кадр выше на 400
    c4 = resolve_crops([c], binned_wcs(a, 4, (2000, 2000)), 500, 4)[0]
    assert abs(c4.x - ca.x) <= 4 and abs(c4.y - ca.y) <= 4
    with pytest.raises(ValidationError):
        ReportCrop(name="x", ra=1.0, size=100)


def test_channel_rect_drops_strip_without_channel():
    from skyforge.align import channel_rect

    sw = np.zeros((300, 400), bool)
    lw = np.zeros((300, 400), bool)
    sw[10:290, 0:400] = True                                            # короткие волны шире слева и справа
    lw[0:300, 40:380] = True
    lw[150, 200] = False                                                # битый пиксель внутри — не в счёт
    r0, r1, c0, c1 = channel_rect({"SW": sw, "LW": lw}, [["LW"], ["SW"]])
    assert (r0, r1, c0, c1) == (10, 290, 40, 380)


def test_block_median_map_does_not_depend_on_frame_extension():
    """Сетка блоков идёт от начала поля: при расширении кадра медиана внутри прежней области та же."""
    from skyforge.field import block_median_map

    rng = np.random.default_rng(2)
    big = rng.normal(0, 1, (700, 800)).astype(np.float32)
    r, c = 113, 57
    inner = big[r:r + 500, c:c + 600]
    a = block_median_map(inner, 95, (0, 0))
    b = block_median_map(big, 95, (r, c))[r:r + 500, c:c + 600]
    m = 200                                                              # вдали от края окно видит одно и то же
    assert np.allclose(a[m:-m, m:-m], b[m:-m, m:-m], atol=1e-6)


def test_channel_rect_inset_from_coverage_edge():
    """Отступ внутрь от границы покрытия каждой группы: крайние столбцы с неполным набором не в кадре."""
    from skyforge.align import channel_rect

    sw = np.zeros((300, 400), bool)
    lw = np.zeros((300, 400), bool)
    sw[10:290, 0:400] = True
    lw[0:300, 40:380] = True
    r0, r1, c0, c1 = channel_rect({"SW": sw, "LW": lw}, [["LW"], ["SW"]], inset_px=8)
    assert (r0, r1, c0, c1) == (18, 282, 48, 372)


def test_channel_rect_with_exposure_and_all_filter_groups():
    """Края: полоса без одного фильтра сверху и полоса с малым числом экспозиций слева срезаются."""
    from skyforge.align import channel_rect

    a = np.ones((300, 400), bool)
    b = np.ones((300, 400), bool)
    b[270:, :] = False                                                  # у фильтра B нет полосы сверху (строки 270+)
    nexp_ok = np.ones((300, 400), bool)
    nexp_ok[:, :30] = False                                             # слева мало экспозиций
    r0, r1, c0, c1 = channel_rect({"A": a, "B": b, "A:nexp": nexp_ok}, [["A", "B"], ["A"], ["B"], ["A:nexp"]])
    assert (r0, r1, c0, c1) == (0, 270, 30, 400)


# --- 3k: пьедестал неба, носок, подъём над небом ---

def _sky_and_nebula(noise_left: float = 1.0, noise_right: float = 3.0, seed: int = 0):
    """Небо с разным шумом слева и справа (как области NEXP 8 и 1) и туманность в середине."""
    rng = np.random.default_rng(seed)
    a = np.full((256, 512), 10.0, dtype=np.float32)
    a[:, :256] += rng.normal(0, noise_left, (256, 256))
    a[:, 256:] += rng.normal(0, noise_right, (256, 256))
    neb = np.zeros(a.shape, dtype=bool)
    neb[96:160, 200:312] = True
    a[neb] = np.linspace(20, 400, neb.sum(), dtype=np.float32)
    return a, (~neb, neb)


def test_sky_toe_puts_sky_median_on_level_without_clipping():
    a, masks = _sky_and_nebula()
    s = StretchConfig(black_mode="pedestal", black_k=3, sky_level=6, a=0.1, white=99.9, noise_block_px=32)
    y, st = stretch_array(a, s, masks=masks)
    sky = masks[0]
    for half in (slice(0, 256), slice(256, 512)):   # тихое и шумное небо: медиана одна, 6 уровней
        assert abs(np.median(y[:, half][sky[:, half]]) * 255 - 6) < 0.3
    assert (y[sky] <= 0).mean() < 0.001             # обрезки в 0 нет и в шумной половине
    assert st["sky_toe"]["p"] > 1


def test_sky_toe_is_monotone_and_linear_above_sky():
    from skyforge.stretch import sky_toe

    x = np.linspace(0, 1, 20001, dtype=np.float32)
    t, _ = sky_toe(x, 0.05, 0.01)
    assert t[0] == 0 and abs(t[-1] - 1) < 1e-6
    assert np.all(np.diff(t) >= 0)
    i = np.searchsorted(x, 0.05)
    assert abs(t[i] - 0.01) < 1e-4
    above = x >= 0.05
    slope = np.diff(t[above]) / np.diff(x[above])
    assert np.allclose(slope, slope[0], rtol=1e-3)  # выше неба — прямая по данным
    below = np.diff(t[:i]) / np.diff(x[:i])
    assert below[-1] == pytest.approx(slope[0], rel=0.01)  # гладко: наклон носка в небе = наклон прямой


def test_shadows_above_sky_keeps_sky_level_and_lifts_signal():
    from skyforge.compose import shadows

    lvl = 6 / 255
    rng = np.random.default_rng(1)
    sky = np.clip(lvl + rng.normal(0, 2 / 255, (200, 200, 3)), 0, 1).astype(np.float32)
    out = shadows(sky, 4, "max", lvl)
    assert abs(np.median(out) - np.median(sky)) * 255 < 0.5     # небо не поднимается: контур маски не проступает
    sig = np.full((1, 1, 3), lvl + 0.1, dtype=np.float32)
    assert shadows(sig, 4, "max", lvl)[0, 0, 0] > lvl + 0.1      # сигнал над небом поднимается
    top = np.ones((1, 1, 3), dtype=np.float32)
    assert shadows(top, 4, "max", lvl)[0, 0, 0] == pytest.approx(1, abs=1e-6)


def test_sky_poly_takes_gradient_not_excluded_bump():
    """3k п.3: полином неба берёт крупный ход по небу вне исключения и не подстраивается под яркое внутри него."""
    from skyforge.skymodel import fit_poly

    rng = np.random.default_rng(2)
    H, W = 300, 400
    yy, xx = np.mgrid[0:H, 0:W]
    grad = 0.002 * xx - 0.001 * yy + 3e-6 * (xx - 200) ** 2
    bump = 5.0 * np.exp(-(((xx - 200) / 40.0) ** 2 + ((yy - 150) / 40.0) ** 2))
    a = (grad + bump + rng.normal(0, 0.05, (H, W))).astype(np.float32)
    excl = np.hypot(xx - 200, yy - 150) < 100
    m, info = fit_poly(a, ~excl, 2)
    g = grad - grad[~excl].mean()
    assert np.abs(m - g)[~excl].max() < 0.02
    assert np.abs(m - g)[excl].max() < 0.05          # внутри — продолжение хода, а не горб
