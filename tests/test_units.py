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


def test_choose_observations_prefers_hap_svm_for_hubble():
    rows = [
        {"obs_id": "u2rc0204t", "obsid": "1", "filters": "F502N", "t_exptime": 400.0, "dataRights": "PUBLIC", "provenance_name": "CALWFPC2"},
        {"obs_id": "hst_6221_02_wfpc2_pc_f502n_u2rc02", "obsid": "2", "filters": "F502N", "t_exptime": 400.0, "dataRights": "PUBLIC", "provenance_name": "HAP-SVM"},
        {"obs_id": "hst_skycell-p1", "obsid": "3", "filters": "F502N", "t_exptime": 5000.0, "dataRights": "PUBLIC", "provenance_name": "HAP-MVM"},
        {"obs_id": "hst_x_total", "obsid": "4", "filters": "detection", "t_exptime": 9000.0, "dataRights": "PUBLIC", "provenance_name": "HAP-SVM"},
    ]
    best = choose_observations(rows)
    assert list(best) == ["F502N"]
    assert best["F502N"].obs_id == "hst_6221_02_wfpc2_pc_f502n_u2rc02"


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


def _flagged_scene(seed: int = 5):
    """Градиент с шумом σ 1; прямоугольник отбракованных пикселей (40% NaN, уцелевшие — выбросы), выбросы у края;
    насыщенная звезда со сплошной дырой ядра."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((220, 220))
    plane = (100 + 0.5 * xx + 0.3 * yy).astype(np.float32)
    a = plane + rng.normal(0, 1, plane.shape).astype(np.float32)
    star = 5000 * np.exp(-((yy - 40) ** 2 + (xx - 40) ** 2) / 18)
    a += star.astype(np.float32)
    a[np.hypot(yy - 40, xx - 40) < 3.5] = np.nan                     # ядро: одна сплошная дыра
    box = (slice(90, 130), slice(80, 150))
    blk = a[box]
    bad = rng.random(blk.shape) < 0.3
    blk[bad] += rng.choice([-25, 25], blk.shape)[bad]                  # уцелевшие пиксели блока — выбросы
    blk[rng.random(blk.shape) < 0.4] = np.nan                        # россыпь дыр блоком с прямыми краями
    ring = np.zeros(a.shape, bool)
    ring[87:133, 77:153] = True
    ring[box] = False
    a[ring & (rng.random(a.shape) < 0.15)] += 20                      # выбросы у края блока
    return a, plane, box, yy, xx


@pytest.mark.parametrize("scale", [0.063, 0.031])
@pytest.mark.parametrize("seed", [0, 5, 10, 15, 20, 25, 31, 35, 40])
def test_flagged_area_filled_smoothly_without_straight_edges(seed, scale):
    """После фазы 7: прямоугольная область отбракованных пикселей на градиенте с выбросами по краю заполняется
    гладко по краю — без ступеньки на прямых краях и без выбросов внутри; ядро звезды областью не считается и
    заполняется прежним способом (не темнее края). Много зёрен и два масштаба (третий раунд проверки кода: одно зерно
    проходило по везению; при 0,063″/px полоса выбросов не дотягивалась до края блока)."""
    from skyforge.stars import fill_rings, fill_smooth, find_core_holes, flagged_areas

    a, plane, box, yy, xx = _flagged_scene(seed)
    region, info = flagged_areas(a, scale=scale)
    assert len(info) == 1 and info[0]["holes"] >= 10
    assert region[box].mean() > 0.9                                   # блок, не только дыры (чистые пиксели у края
                                                                      # могут остаться — выбросы проверяет остаток ниже)
    assert not region[np.hypot(yy - 40, xx - 40) < 6].any()           # ядро звезды — не область
    holes = find_core_holes(a, 3000)
    out = fill_rings(fill_smooth(a, region), holes & ~region)
    assert np.isfinite(out[box]).all()
    res = out - plane
    assert np.abs(res[box]).max() < 4                                  # ни выбросов, ни дыр
    for inside, outside in (((slice(90, 130), slice(80, 83)), (slice(90, 130), slice(74, 77))),
                            ((slice(90, 130), slice(147, 150)), (slice(90, 130), slice(153, 156))),
                            ((slice(90, 93), slice(80, 150)), (slice(84, 87), slice(80, 150))),
                            ((slice(127, 130), slice(80, 150)), (slice(133, 136), slice(80, 150)))):
        assert abs(res[inside].mean() - res[outside].mean()) < 0.6     # без ступеньки (σ разности средних ~0,13)
    rim = a[(np.hypot(yy - 40, xx - 40) >= 3.5) & (np.hypot(yy - 40, xx - 40) < 4.5)]
    assert out[40, 40] >= np.nanmax(rim) - 1e-3                        # ядро — максимумом соседей, как прежде


def _fill_all(a, scale=0.063):
    from skyforge.stars import fill_rings, fill_smooth, find_core_holes, flagged_areas

    region, info = flagged_areas(a, scale)
    out = fill_smooth(a, region)
    return fill_rings(out, find_core_holes(a, 3000) & ~np.isfinite(out)), region, info


def test_flagged_area_next_to_mosaic_gap_keeps_data():
    """Проверка кода: область у пропуска мозаики (NaN до края кадра) — куски без известного соседа не решаются
    (вырожденная система давала NaN на весь кусок), известные пиксели не становятся NaN."""
    for seed in range(6):
        a, plane, box, yy, xx = _flagged_scene(seed)
        a[:, 150:165] = np.nan                                         # пропуск вплотную к блоку, до края кадра
        out, region, info = _fill_all(a)
        assert info, seed
        assert np.isfinite(out[np.isfinite(a)]).all(), seed           # данные не потеряны
        assert np.isnan(out[:, 150:165]).all()                         # пропуск мозаики не заполняется


def test_fill_smooth_components_do_not_overwrite_each_other():
    """Проверка кода: рамка одной области накрывает другую — каждая решается своей, без шва на краю окна (на линейном
    градиенте решение уравнения Лапласа точное)."""
    from skyforge.stars import fill_smooth

    yy, xx = np.indices((60, 70))
    ramp = (yy + 2.0 * xx).astype(np.float32)
    reg = np.zeros(ramp.shape, bool)
    reg[5:30, 30:55] = True                                            # область A
    reg[35:39, 5:50] = True
    reg[10:39, 5:9] = True                                             # область B буквой Г, её рамка накрывает A
    a = np.where(reg, np.nan, ramp).astype(np.float32)
    out = fill_smooth(a, reg)
    assert np.abs(out[reg] - ramp[reg]).max() < 1e-2


def test_flagged_area_spares_star_core_and_nearby_faint_stars():
    """Проверка кода: насыщенное ядро звезды рядом с россыпью в область не входит и заполняется белым, как прежде;
    слабые звёзды в нескольких пикселях от края области не стираются."""
    a, plane, box, yy, xx = _flagged_scene(5)
    star = 5000 * np.exp(-((yy - 110) ** 2 + (xx - 72) ** 2) / 18)    # ядро в 8 px от края блока
    a = a + star.astype(np.float32)
    core = np.hypot(yy - 110, xx - 72) < 3.5
    a[core] = np.nan
    faint = [(84, 100), (84, 120), (136, 110)]                          # слабые звёзды в 6 px от блока
    for y, x in faint:
        a = a + (15 * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 0.45 ** 2))).astype(np.float32)
    out, region, info = _fill_all(a)
    assert info and not region[core].any()
    rim = a[(np.hypot(yy - 110, xx - 72) >= 3.5) & (np.hypot(yy - 110, xx - 72) < 4.5)]
    assert out[110, 72] >= np.nanmax(rim) - 1e-3                       # ядро — максимумом соседей
    for y, x in faint:
        assert not region[y, x] and out[y, x] == a[y, x]


@pytest.mark.parametrize("dist, rad", [(8, 2.5), (8, 3.0), (4, 3.5), (1, 3.5)])
def test_star_core_near_or_touching_area_stays_white(dist, rad):
    """Проверка кода (второй раунд): малое ядро (21–25 px) рядом с россыпью и ядро, касающееся её, — ядра, не область:
    центр не темнее края, как при прежнем fill_rings."""
    a, plane, box, yy, xx = _flagged_scene(0)
    x0 = 80 - dist
    a = a + (5000 * np.exp(-((yy - 110) ** 2 + (xx - x0) ** 2) / 18)).astype(np.float32)
    core = np.hypot(yy - 110, xx - x0) < rad
    a[core] = np.nan
    out, region, info = _fill_all(a)
    rim = a[(np.hypot(yy - 110, xx - x0) >= rad) & (np.hypot(yy - 110, xx - x0) < rad + 1) & np.isfinite(a)]
    assert info and not region[core].any()
    assert out[110, x0] >= np.max(rim) - 1e-3
    # как в clean: другой фильтр видит ту же звезду без насыщения — ядро подтверждено (четвёртый раунд проверки)
    from skyforge.stars import fill_rings, fill_smooth, find_core_holes, flagged_areas

    other = (plane + 0.1 * 5000 * np.exp(-((yy - 110) ** 2 + (xx - x0) ** 2) / 18)).astype(np.float32)
    region2, _ = flagged_areas(a, 0.063, [(other, np.zeros(a.shape, bool))])
    out2 = fill_rings(fill_smooth(a, region2), find_core_holes(a, 3000) & ~np.isfinite(fill_smooth(a, region2)))
    assert not region2[core].any() and out2[110, x0] >= np.max(rim) - 1e-3


def test_star_core_inside_scattered_annulus_keeps_core_and_inner_wings():
    """Проверка кода (второй раунд): яркая звезда с насыщенным ядром посреди россыпи (случай протозвезды) — ядро белое,
    крылья в кольце защиты не сглажены."""
    rng = np.random.default_rng(1)
    yy, xx = np.indices((220, 220))
    plane = (100 + 0.5 * xx + 0.3 * yy).astype(np.float32)
    star = (5000 * np.exp(-((yy - 110) ** 2 + (xx - 115) ** 2) / 18)).astype(np.float32)
    a = plane + star + rng.normal(0, 1, plane.shape).astype(np.float32)
    r = np.hypot(yy - 110, xx - 115)
    ann = (r >= 5.5) & (r < 25)
    bad = ann & (rng.random(a.shape) < 0.3)
    a[bad] += rng.choice([-25, 25], a.shape)[bad]
    a[ann & (rng.random(a.shape) < 0.4)] = np.nan
    a[r < 3.5] = np.nan
    out, region, info = _fill_all(a)
    assert info and not region[r < 3.5].any()
    rim = a[(r >= 3.5) & (r < 4.5)]
    assert out[110, 115] >= np.nanmax(rim) - 1e-3                      # ядро белое
    guard = (r >= 3.5) & (r < 3.5 + 3) & np.isfinite(a)                 # 3 px — внутри кольца защиты 0,25″
    assert np.array_equal(out[guard], a[guard])                        # крылья у ядра — как были


@pytest.mark.parametrize("seed", [1, 5, 9, 13, 17])
@pytest.mark.parametrize("density", [0.55, 0.7])
def test_dense_flagged_block_is_found(density, seed):
    """Проверка кода (второй раунд): плотная россыпь (дыры срастаются в несколько крупных компонент) — тоже область."""
    a, plane, box, yy, xx = _flagged_scene(seed)
    rng = np.random.default_rng(seed + 100)
    blk = a[box]
    blk[rng.random(blk.shape) < density] = np.nan
    out, region, info = _fill_all(a)
    assert info and region[box].mean() > 0.9
    assert np.abs((out - plane)[box]).max() < 6


@pytest.mark.parametrize("seed", range(6))
def test_clumped_low_coverage_strip_is_not_an_area(seed):
    """Проверка кода (второй раунд): полоса малого покрытия (шум ×2,8) с битыми пикселями комками 2×2 (11%) — не
    область, звёзды в ней не стираются."""
    from skyforge.stars import flagged_areas

    rng = np.random.default_rng(seed)
    a = rng.normal(100, 1, (400, 400)).astype(np.float32)
    a[:, 180:220] = rng.normal(100, 2.8, (400, 40))
    k = rng.random((200, 20)) < 0.11
    a[:, 180:220][np.kron(k, np.ones((2, 2), bool))] = np.nan
    for sc in (0.031, 0.063):
        region, info = flagged_areas(a, scale=sc)
        assert not region.any() and info == [], sc


def test_core_needs_a_source_in_another_filter():
    """Проверка кода (третий и четвёртый раунды): ядро подтверждается источником на том же месте в другом фильтре —
    сплошной дырой или ярким компактным пиком (звезда, насыщенная в одном фильтре, или узел выброса). Комок блока на
    ярком краю туманности (OMC3-N F200W), где в других фильтрах ровная туманность, — не ядро."""
    from skyforge.stars import core_parts, find_core_holes

    rng = np.random.default_rng(3)
    yy, xx = np.indices((160, 160))
    star = 2000 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / 18)
    a = (100 + star + rng.normal(0, 1, star.shape)).astype(np.float32)
    core = np.hypot(yy - 80, xx - 80) < 4
    a[core] = np.nan
    holes = find_core_holes(a, 3000)
    flat = (100 + 0.3 * xx + rng.normal(0, 1, star.shape)).astype(np.float32)        # ровная туманность
    none = np.zeros(a.shape, bool)
    assert core_parts(a, holes)[core].all()                                           # без других фильтров — ядро
    assert not core_parts(a, holes, [(flat, none)], 5).any()                          # в другом фильтре нет источника
    other_hole = none.copy()
    other_hole[78:83, 79:84] = True
    assert core_parts(a, holes, [(flat, other_hole)], 5)[core].all()                  # дыра в другом фильтре
    peak = (100 + 0.1 * star + rng.normal(0, 1, star.shape)).astype(np.float32)        # тот же источник без насыщения
    assert core_parts(a, holes, [(flat, none), (peak, none)], 5)[core].all()
    # пятый раунд (OMC3-N F480M): вытянутая яркая нить под частью — не точечный источник, не подтверждение
    ridge = (100 + 200 * np.exp(-((xx - 80) ** 2) / (2 * 4.0 ** 2)) + rng.normal(0, 1, star.shape)).astype(np.float32)
    assert not core_parts(a, holes, [(flat, none), (ridge, none)], 5).any()


def test_low_coverage_strip_with_sparse_bad_pixels_is_not_an_area():
    """Проверка кода: полоса малого покрытия (шум ×2,8) с редкими битыми пикселями (4%) и звёздами — не область."""
    from skyforge.stars import flagged_areas

    rng = np.random.default_rng(2)
    a = rng.normal(100, 1, (400, 400)).astype(np.float32)
    a[:, 180:220] = rng.normal(100, 2.8, (400, 40))
    yy, xx = np.indices(a.shape)
    for y in (100, 200, 300):
        a += (200 * np.exp(-((yy - y) ** 2 + (xx - 200) ** 2) / (2 * 1.5 ** 2))).astype(np.float32)
    strip = a[:, 180:220]
    strip[rng.random(strip.shape) < 0.04] = np.nan
    region, info = flagged_areas(a, scale=0.031)
    assert not region.any() and info == []


def test_flagged_area_skips_sparse_holes_and_star_cores():
    """Редкие одиночные дыры (россыпь NGC 3132) и сплошные ядра звёзд — не области."""
    from skyforge.stars import flagged_areas

    rng = np.random.default_rng(9)
    a = rng.normal(100, 1, (200, 200)).astype(np.float32)
    ys, xs = rng.integers(5, 195, 40), rng.integers(5, 195, 40)
    a[ys, xs] = np.nan                                                # 40 одиночных дыр по всему кадру
    yy, xx = np.indices(a.shape)
    a[np.hypot(yy - 100, xx - 100) < 8] = np.nan                       # большое сплошное ядро
    region, info = flagged_areas(a, scale=0.063)
    assert not region.any() and info == []


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


def test_sky_box_rect_center_and_size():
    """3б: прямоугольник на небе (центр, размер в ″) → пиксели сетки «север вверх»; за краем сетки — обрезан."""
    from conftest import CENTER

    from skyforge.align import sky_box_rect
    from skyforge.config import SkyBox

    w = make_wcs(0.1, (1000, 1200), 0.0, (0.0, 0.0))
    r0, r1, c0, c1 = sky_box_rect(SkyBox(ra=CENTER[0], dec=CENTER[1], width_arcsec=40, height_arcsec=20), w, (1000, 1200))
    assert (r1 - r0, c1 - c0) == (200, 400)
    assert abs((r0 + r1) / 2 - 499.5) <= 1 and abs((c0 + c1) / 2 - 599.5) <= 1
    r0, r1, c0, c1 = sky_box_rect(SkyBox(ra=CENTER[0], dec=CENTER[1], width_arcsec=400, height_arcsec=400), w, (1000, 1200))
    assert (r0, r1, c0, c1) == (0, 1000, 0, 1200)


def test_load_context_multiplane_same_partition_as_records(tmp_path):
    """Фаза 4: CON мозаики > 32 экспозиций — несколько плоскостей int32 (Sh2-305: 6 × 193 млн пикселей, запись из
    6 int32 на пиксель требовала > 4 ГБ). Хеш набора битов даёт то же разбиение на области, что и сравнение записей."""
    import numpy as np
    from astropy.io import fits

    from skyforge.fitsio import load_context

    rng = np.random.default_rng(0)
    H, W = 60, 70
    con = np.zeros((3, H, W), dtype=np.int32)
    for k in range(3):
        con[k] = rng.choice([0, 1, 5, 2**31 - 1, -2**31], size=(H, W))   # и отрицательные (старший бит)
    con[2, :10, :] = 7                                   # области, отличающиеся только третьей плоскостью
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((H, W), np.float32), name="SCI"),
                         fits.ImageHDU(con, name="CON")])
    p = tmp_path / "x_i2d.fits"
    hdul.writeto(p)
    lab = load_context(p)
    flat = np.ascontiguousarray(con.reshape(3, -1).T)
    _, ref = np.unique(flat.view(np.dtype((np.void, 12))), return_inverse=True)
    ref = ref.reshape(H, W)
    pairs = set(zip(lab.ravel().tolist(), ref.ravel().tolist()))   # разбиения совпадают ⇔ соответствие один к одному
    assert len(pairs) == len(set(lab.ravel().tolist())) == len(set(ref.ravel().tolist()))


def test_sky_toe_max_p_no_zeros_under_strong_stretch():
    """Фаза 4 шаг 4: при сильной растяжке (a 0,02) степенной носок 3k подбирает p ~ 100+ — «стенка»: шум чуть ниже
    неба уходит в 0. С ограничением stretch.sky_toe_max_p 4 значение у неба то же, ниже неба кривая положе: нулей
    < 0,5%, медиана неба на sky_level, выше неба кривая не меняется. Без флага — прежний носок (много нулей)."""
    import numpy as np

    from skyforge.config import StretchConfig
    from skyforge.stretch import stretch_array

    rng = np.random.default_rng(4)
    H, W = 200, 400
    a = np.zeros((H, W), dtype=np.float32)
    a[:, :200] = 10 + rng.normal(0, 1.0, (H, 200))                         # небо: медиана 10, σ 1
    a[:, 200:] = 10 + np.linspace(0, 150, 200)[None, :] + rng.normal(0, 1.0, (H, 200))   # туманность
    sky = np.zeros((H, W), bool); sky[:, :200] = True
    neb = ~sky
    common = dict(anchor="masks", black_mode="pedestal", black_k=8, sky_level=6, a=0.02, white=99.9, noise_block_px=16)
    old, so = stretch_array(a, StretchConfig(**common), masks=(sky, neb))
    new, sn = stretch_array(a, StretchConfig(**common, sky_toe_max_p=4), masks=(sky, neb))
    zeros = lambda y: (np.round(y[sky] * 255) <= 0).mean()
    assert so["sky_toe"]["p"] > 20 and zeros(old) > 0.05                    # стенка: прежний носок
    assert sn["sky_toe"]["p"] == 4 and zeros(new) < 0.005
    assert abs(np.median(new[sky]) * 255 - 6) < 0.6
    above = a > np.median(a[sky])
    assert np.array_equal(new[above], old[above])                          # выше неба — без изменений


def _nebula_with_empty_corner(seed: int, n: int = 768, block: int = 128):
    """Градиентная туманность на весь кадр, кроме левого верхнего блока: там пустое небо (медиана 2, σ 1)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    yy, xx = np.indices((n, n))
    a = (6 + 40 * (xx + yy) / (2 * n) + rng.normal(0, 1.0, (n, n))).astype(np.float32)
    a[:block, :block] = 2 + rng.normal(0, 1.0, (block, block))
    corner = np.zeros((n, n), bool)
    corner[:block, :block] = True
    return a, corner


def test_sky_reference_darkest_blocks_finds_empty_corner():
    """Фаза 4 шаг 4d: на сплошной туманности маска «пустого неба» лежит на слабой туманности; опора по самым тёмным
    блокам находит пустой угол, и медиана угла после растяжки — на sky_level (а не туманность на нём)."""
    import numpy as np

    from skyforge.config import StretchConfig
    from skyforge.stretch import darkest_blocks, stretch_array

    a, corner = _nebula_with_empty_corner(11)
    ok = np.ones(a.shape, bool)
    ref, info = darkest_blocks(a, ok, block=128, n=1, min_free=0.5)
    assert info["blocks"][0][:2] == [0, 0] and ref[corner].all() and not ref[~corner].any()
    neb = a > np.percentile(a, 60)
    s = StretchConfig(anchor="masks", black_mode="pedestal", black_k=3, sky_level=16, a=0.02, white=99.8,
                      noise_block_px=32, sky_toe_max_p=4)
    y, _ = stretch_array(a, s, masks=(ref, neb))
    assert abs(np.median(y[corner]) * 255 - 16) < 1.0
    # прежняя опора — вся «не туманность»: её медиана выше пустого угла, угол уходит под пол
    y_old, _ = stretch_array(a, s, masks=(~neb, neb))
    assert np.median(y_old[corner]) * 255 < 16 - 4


def test_sky_reference_min_free_too_strict_moves_into_nebula():
    """Ошибка шага 4d, первый запуск: при доле чистых пикселей 0,9 и маске звёзд ~20% кадра пустой угол не проходит,
    и опора уходит в туманность (медиана опоры выше угла); при 0,5 — угол находится."""
    import numpy as np

    from skyforge.stretch import darkest_blocks

    a, corner = _nebula_with_empty_corner(12)
    rng = np.random.default_rng(13)
    stars = rng.random(a.shape) < 0.2
    stars[384:512, 384:512] = False                         # один чистый блок посреди туманности (по сетке блоков)
    ok = ~stars
    strict, si = darkest_blocks(a, ok, block=128, n=1, min_free=0.9)
    loose, li = darkest_blocks(a, ok, block=128, n=1, min_free=0.5)
    assert si["eligible"] == 1 and np.median(a[strict]) > np.median(a[corner]) + 10
    assert li["blocks"][0][:2] == [0, 0] and li["eligible"] > si["eligible"]


def test_sky_reference_default_keeps_stretch_key():
    """По умолчанию (mask) поля опоры в ключ растяжки не входят — кеши NGC 3132 и Hubble прежние."""
    from skyforge.config import Config
    from skyforge.stretch import stretch_key

    k = stretch_key(Config.model_validate({"target": "t"}), {})["stretch"]
    assert not any(f.startswith("sky_reference") for f in k)
    on = Config.model_validate({"target": "t", "stretch": {"sky_reference": "darkest_blocks"}})
    assert stretch_key(on, {})["stretch"]["sky_reference"] == "darkest_blocks"


def test_download_trusts_server_size_over_stale_mast_size(tmp_path, monkeypatch):
    """Фаза 6: размер в списке продуктов MAST устарел (файл перепроцессирован) — сверка по Content-Length сервера,
    без пяти перекачек целого файла."""
    import skyforge.fetch as fetch

    body = b"x" * 10000
    calls = {"get": 0}

    class Head:
        ok, headers = True, {"Content-Length": str(len(body))}

    class Get:
        status_code = 200

        def __enter__(self):
            calls["get"] += 1
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield body

    monkeypatch.setattr(fetch.requests, "head", lambda *a, **k: Head())
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: Get())
    dest = tmp_path / "f.fits"
    fetch.download("mast:JWST/product/f.fits", dest, len(body) - 5760)
    assert dest.read_bytes() == body and calls["get"] == 1


def test_region_rects_two_separated_modules():
    """Фаза 6 (W3-2): два разнесённых квадрата, у каждого фильтра свой сдвиг — по прямоугольнику «все фильтры на краях»
    на модуль; щель короткого фильтра внутри модуля (полоса 6 px) кадр не режет."""
    from skyforge.align import region_rects

    a = np.zeros((200, 500), bool)
    b = np.zeros((200, 500), bool)
    for c0 in (20, 300):                                                 # модули A и B, промежуток 100 px
        a[20:180, c0:c0 + 180] = True
        b[30:190, c0 + 5:c0 + 185] = True                                # второй фильтр сдвинут на 10 и 5 px
    a[:, 110:116] = False                                                # щель детекторов в модуле A у фильтра a
    rects = region_rects({"a": a, "b": b}, [["a"], ["b"]], a | b, inset_px=2, gap_px=8)
    assert rects == [(32, 178, 27, 198), (32, 178, 307, 478)]


def test_region_rects_single_region_unchanged():
    """Одна область покрытия — тот же прямоугольник, что channel_rect (кадр объектов до фазы 6 не меняется)."""
    from skyforge.align import channel_rect, region_rects

    sw = np.zeros((300, 400), bool)
    lw = np.zeros((300, 400), bool)
    sw[10:290, 0:400] = True
    lw[0:300, 40:380] = True
    lw[5:8, 390:395] = True                                              # обрывок — меньше 5% покрытия
    assert region_rects({"SW": sw, "LW": lw}, [["LW"], ["SW"]], sw | lw, inset_px=8) == \
        [channel_rect({"SW": sw, "LW": lw}, [["LW"], ["SW"]], inset_px=8)]


def test_fill_gaps_ratio_and_untouched_outside():
    """Фаза 6: щель фильтра A заполняется B × гладкое отношение A/B; вне щели слои не меняются ни на бит."""
    from skyforge.compose import fill_gaps

    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:256, 0:256]
    b = (0.2 + 0.3 * np.exp(-((xx - 128) ** 2 + (yy - 100) ** 2) / 3000) + rng.normal(0, 0.002, (256, 256))).astype(np.float32)
    ratio = (1.5 + 0.002 * xx).astype(np.float32)                         # отношение плавно меняется по x
    a = (b * ratio).astype(np.float32)
    a_gap = a.copy()
    a_gap[:, 120:136] = np.nan                                            # щель 16 px через весь кадр
    b_gap = b.copy()
    b_gap[60:70, :] = np.nan                                              # у B своя щель
    cfg = Config(target="t", filters=["F182M", "F210M"],
                 color={"fill_gaps": {"groups": [["F182M", "F210M"]], "block_px": 32}})
    out, filled = fill_gaps(cfg, {"F182M": a_gap, "F210M": b_gap})
    ga, gb = filled["F182M"], filled["F210M"]
    assert ga[:, 120:136][np.isfinite(b_gap[:, 120:136])].all() and not ga[:, :120].any()
    assert np.array_equal(out["F182M"][~ga], a_gap[~ga], equal_nan=True)
    assert np.array_equal(out["F210M"][~gb], b_gap[~gb], equal_nan=True)
    err = np.abs(out["F182M"][ga] - a[ga]) / a[ga]
    assert np.median(err) < 0.01                                          # гладкое отношение восстановлено


def test_fill_gaps_crossing_from_third_filter():
    """Перекрестье щелей, где нет обоих фильтров пары, — из третьего фильтра группы."""
    from skyforge.compose import fill_gaps

    base = np.full((128, 128), 0.3, dtype=np.float32)
    a, b, c = base * 1.2, base.copy(), base * 0.8
    a[:, 60:70] = np.nan
    b[60:70, :] = np.nan
    cfg = Config(target="t", filters=["F182M", "F210M", "F300M"],
                 color={"fill_gaps": {"groups": [["F182M", "F210M", "F300M"]], "block_px": 32}})
    out, filled = fill_gaps(cfg, {"F182M": a, "F210M": b, "F300M": c})
    assert np.isfinite(out["F182M"]).all() and np.isfinite(out["F210M"]).all()
    assert np.allclose(out["F182M"][60:70, 60:70], 0.36, atol=1e-4)   # из F300M × (1,2 / 0,8)
    assert "F300M" not in filled


def test_point_sources_stars_not_filaments():
    """Фаза 6: точечные источники — звёзды (резкие пики), а не волокно туманности; у яркой звезды круг шире."""
    from skyforge.config import PointSourcesConfig
    from skyforge.stretch import point_sources

    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:300, 0:300]
    a = 1.0 + 0.5 * np.exp(-((xx - 150) ** 2) / (2 * 6 ** 2))            # волокно шириной ~14 px
    a = a + rng.normal(0, 0.01, a.shape)
    for (y, x, amp) in ((60, 60, 50.0), (200, 220, 3.0), (100, 150, 20.0)):   # звезда и на волокне
        a = a + amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.0 ** 2))
    mask, info = point_sources(a.astype(np.float32), PointSourcesConfig(), scale_arcsec=0.25)
    assert info["stars"] == 3
    assert mask[60, 60] and mask[200, 220] and mask[100, 150]
    assert not mask[250, 150] and not mask[20, 150]                      # волокно вдали от звёзд — не исключено
    from scipy import ndimage

    lab, _ = ndimage.label(mask)
    area = np.bincount(lab.ravel())
    assert area[lab[60, 60]] > area[lab[200, 220]]                        # яркая звезда — круг шире


def test_split_ranges():
    from skyforge.fetch import split_ranges

    assert split_ranges(1000, 250) == [(0, 249), (250, 499), (500, 749), (750, 999)]
    assert split_ranges(150, 250) == [(0, 149)]                                 # меньше куска — целиком
    r = split_ranges(1003, 100)
    assert len(r) == 11 and r[0][0] == 0 and r[-1][1] == 1002 and all(b + 1 == c for (_, b), (c, _) in zip(r, r[1:]))


def _fake_range_server(monkeypatch, body: bytes, fail_once: set | None = None):
    """Поддельный MAST: HEAD — размер и Accept-Ranges; GET с Range — нужный кусок (206). fail_once — начала диапазонов,
    на которых первый ответ обрывается после половины."""
    import requests as rq

    import skyforge.fetch as fetch

    failed = set()

    class Head:
        ok = True
        headers = {"Content-Length": str(len(body)), "Accept-Ranges": "bytes"}

    class Get:
        def __init__(self, a, b):
            self.a, self.b, self.status_code = a, b, 206

        def __enter__(self):
            return self

        def __exit__(self, *x):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            chunk = body[self.a:self.b + 1]
            if fail_once is not None and self.a in fail_once and self.a not in failed:
                failed.add(self.a)
                yield chunk[:len(chunk) // 2]
                raise rq.ConnectionError("обрыв")
            yield chunk

    def get(url, params=None, headers=None, stream=False, timeout=None):
        a, b = headers["Range"].split("=")[1].split("-")
        return Get(int(a), int(b) if b else len(body) - 1)

    monkeypatch.setattr(fetch.requests, "head", lambda *a, **k: Head())
    monkeypatch.setattr(fetch.requests, "get", get)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch, "PIECE", 1000)


def test_download_in_parallel_ranges_with_resume(tmp_path, monkeypatch):
    """Фаза 6: большой файл — частями параллельно (Range), обрыв одной части докачивается, склейка байт-в-байт."""
    import skyforge.fetch as fetch

    body = bytes(np.random.default_rng(1).integers(0, 256, 10000, dtype=np.uint8))
    _fake_range_server(monkeypatch, body, fail_once={2000, 7000})
    dest = tmp_path / "f.fits"
    assert fetch.download("mast:JWST/product/f.fits", dest, len(body), parts=4) == len(body)
    assert dest.read_bytes() == body and not list(tmp_path.glob("f.fits.part*"))
    assert len(fetch.split_ranges(len(body))) == 10                            # 10 кусков по 1000 из очереди


def test_download_many_parallel_files(tmp_path, monkeypatch):
    """Несколько файлов одновременно: каждый целиком, on_done — по разу на файл."""
    import skyforge.fetch as fetch

    body = bytes(np.random.default_rng(2).integers(0, 256, 5000, dtype=np.uint8))
    _fake_range_server(monkeypatch, body)
    done = []
    jobs = [("mast:x", tmp_path / f"f{i}.fits", len(body)) for i in range(3)]
    fetch.download_many(jobs, workers=3, parts=2, on_done=lambda i, size: done.append((i, size)))
    assert sorted(done) == [(0, 5000), (1, 5000), (2, 5000)]
    assert all((tmp_path / f"f{i}.fits").read_bytes() == body for i in range(3))
