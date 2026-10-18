"""Фаза 3d: полосы 1/f (destripe) и цветной фон (выравнивание каналов по пустому небу)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from skyforge.config import Config, DestripeConfig


def _stars_image(rng: np.random.Generator, shape=(300, 400), n=25, noise=1.0):
    yy, xx = np.indices(shape)
    img = rng.normal(0, noise, shape).astype(np.float32)
    stars = []
    for _ in range(n):
        y, x = rng.uniform(15, shape[0] - 15), rng.uniform(15, shape[1] - 15)
        flux = rng.uniform(2000, 8000)
        img += flux * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 1.5**2)) / (2 * np.pi * 1.5**2)
        stars.append((y, x, flux))
    return img, stars


def _aperture(img: np.ndarray, y: float, x: float, r: float = 6) -> float:
    """Поток звезды над местным фоном: сумма в апертуре минус медиана кольца 8–12 px на площадь."""
    yy, xx = np.indices(img.shape)
    d = np.hypot(yy - y, xx - x)
    ap = d <= r
    return float(img[ap].sum() - ap.sum() * np.median(img[(d >= 8) & (d <= 12)]))


def test_destripe_removes_row_offsets_and_keeps_stars():
    from skyforge.destripe import destripe, step_metric

    rng = np.random.default_rng(11)
    noise = 1.0
    clean, stars = _stars_image(rng, noise=noise)
    offsets = rng.normal(0, 3 * noise, clean.shape[0]).astype(np.float32)  # σ = 3 шума
    striped = clean + offsets[:, None]
    yy, xx = np.indices(clean.shape)
    mask = np.zeros(clean.shape, dtype=bool)
    for y, x, _ in stars:
        mask |= np.hypot(yy - y, xx - x) <= 8  # маска звёзд, расширенная
    labels = np.where(xx < 200, 0, 1).astype(np.int32)  # две области CON со своими полосами
    c = DestripeConfig()
    out, info = destripe(striped, labels, mask, c)
    before, after = step_metric(striped, mask, labels, c), step_metric(out, mask, labels, c)
    assert before / after >= 10, f"|Δ медиан соседних строк| {before:.3f} → {after:.3f}"
    for y, x, _ in stars:
        truth = _aperture(clean, y, x)
        assert abs(_aperture(out, y, x) - truth) / truth < 0.005
    assert info["regions"] == 2


def test_destripe_columns_default_on():
    from skyforge.destripe import destripe

    rng = np.random.default_rng(3)
    img = rng.normal(0, 1, (200, 200)).astype(np.float32) + rng.normal(0, 3, 200).astype(np.float32)[None, :]
    mask = np.zeros(img.shape, dtype=bool)
    rows_only, _ = destripe(img, None, mask, DestripeConfig(columns=False))
    both, _ = destripe(img, None, mask, DestripeConfig())  # столбцы включены по умолчанию
    col_std = lambda a: float(np.std(np.median(a, axis=0)))  # noqa: E731
    assert col_std(both) < 0.2 * col_std(rows_only)


def test_destripe_keeps_level_and_trend_in_hidden_rows():
    """Полосы уходят, а уровень и плавный ход фона (слабый ореол) остаются — в том числе в строках,
    где свободных пикселей меньше 50 и медиана берётся от соседних строк."""
    from skyforge.destripe import destripe

    rng = np.random.default_rng(5)
    n = 900  # настоящие области CON — тысячи строк; окно скользящей медианы целиком внутри данных
    trend = 3 + np.linspace(0, 5, n, dtype=np.float32)[:, None]     # уровень области и ход ореола
    stripes = rng.normal(0, 1.0, (n, 1)).astype(np.float32)
    img = rng.normal(0, 0.2, (n, 300)).astype(np.float32) + trend + stripes
    mask = np.zeros(img.shape, dtype=bool)
    mask[440:460, :280] = True                                         # в этих строках 20 свободных < 50
    out, _ = destripe(img, None, mask, DestripeConfig(window_rows=201))
    resid = (np.median(out, axis=1) - trend[:, 0])[150:-150]
    # Полосы ушли (≥4 раз); медленная часть случайных смещений неотличима от хода фона и остаётся —
    # это цена того, что ореол и уровень области не трогаются.
    assert np.std(resid) < 0.25 * np.std(stripes)
    assert abs(float(np.median(resid))) < 0.05                         # уровень не тронут
    # Ход фона в скрытых строках сохранён (их собственные полосы не измерить — значения берутся от соседей).
    assert abs(float(np.median(out[440:460])) - float(np.median(trend[440:460]))) < 0.3


def test_destripe_leaves_unmeasurable_region_alone():
    """Область без свободного неба (целиком под маской) не трогается: её уровень сшивает с соседями clean."""
    from skyforge.destripe import destripe

    rng = np.random.default_rng(6)
    img = rng.normal(0, 0.2, (200, 200)).astype(np.float32) + 1.0
    labels = np.zeros((200, 200), dtype=np.int32)
    labels[50:120, 30:110] = 1
    mask = labels == 1
    out, _ = destripe(img, labels, mask, DestripeConfig(window_rows=0, columns=False))
    assert np.array_equal(out[labels == 1], img[labels == 1])
    assert abs(float(np.median(out[labels == 0]))) < 0.05  # измеримая область приведена к нулю


def test_equalize_sky_makes_channels_equal_and_keeps_star_color():
    from skyforge.compose import equalize_sky

    rgb = np.zeros((200, 200, 3), dtype=np.float32) + np.array([0.10, 0.14, 0.09], dtype=np.float32)
    rgb += np.random.default_rng(1).normal(0, 0.005, rgb.shape).astype(np.float32)
    star = np.array([0.6, 0.4, 0.3], dtype=np.float32)
    rgb[95:105, 95:105] += star
    sky = np.ones((200, 200), dtype=bool)
    sky[85:115, 85:115] = False
    out, before, after = equalize_sky(rgb, sky)
    med = [float(np.median(out[..., k][sky])) for k in range(3)]
    assert max(med) - min(med) < 0.001
    excess_before = rgb[100, 100] - np.median(rgb[sky], axis=0)
    excess_after = out[100, 100] - np.median(out[sky], axis=0)
    assert np.allclose(excess_before, excess_after, atol=1e-5)  # цвет звезды над фоном тот же


def test_report_crops_written(sky, cfg, tmp_path):
    from skyforge.pipeline import run_all

    c = cfg.model_validate({**cfg.model_dump(mode="json"),
                            "report": {"crops": [{"name": "sky", "x": 40, "y": 140, "size": 48},
                                                 {"name": "core", "x": 95, "y": 90, "size": 32}]}})
    out = run_all(c, sky, offline=True, out_dir=tmp_path / "r")
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    files = {x["name"]: x["file"] for x in run["stages"]["finish"]["crops"]}
    assert set(files) == {"sky", "core"} and all((out / f).exists() for f in files.values())
    assert (out / "background_check.png").exists() and (out / "destripe_check.png").exists()
    assert "crop_sky.png" in (out / "report.html").read_text(encoding="utf-8")
    # Хеш обработки от кропов отчёта не зависит.
    assert c.digest() == cfg.digest()


def test_old_clean_cache_not_used(sky, cfg, tmp_path):
    from skyforge.align import align
    from skyforge.clean import clean, clean_dir

    align(cfg, sky, tmp_path / "o")
    assert clean(cfg, sky, tmp_path / "o")["cached"] is False
    assert clean(cfg, sky, tmp_path / "o")["cached"] is True
    meta_path = clean_dir(cfg, sky) / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["key"]["version"] -= 1  # кеш от прошлой версии алгоритма
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    assert clean(cfg, sky, tmp_path / "o")["cached"] is False


@pytest.mark.parametrize("enabled", [True, False])
def test_destripe_switch_changes_align_key(sky, cfg, tmp_path, enabled):
    from skyforge.align import align

    c = cfg.model_validate({**cfg.model_dump(mode="json"), "clean": {"destripe": {"enabled": enabled}}})
    info = align(c, sky, tmp_path / "o")
    assert bool(info.get("destripe")) is enabled


def test_match_regions_fixes_bright_tile_that_mask_would_hide():
    """Область с пьедесталом в 3σ целиком прячется маской источников; сшивка по границам её выравнивает,
    а туманность, пересекающая границу, не страдает."""
    from skyforge.background import match_regions

    rng = np.random.default_rng(8)
    yy, xx = np.indices((240, 320))
    regions = np.zeros((240, 320), dtype=np.int32)
    regions[:, 160:] = 1
    regions[60:120, 40:120] = 2                       # маленький яркий прямоугольник
    ped = np.array([0.0, -1.0, 3.0], dtype=np.float32)
    nebula = 20 * np.exp(-((yy - 180) ** 2 + (xx - 160) ** 2) / (2 * 40**2))  # через границу 0|1
    truth = (rng.normal(0, 1, yy.shape) + nebula).astype(np.float32)
    img = truth + ped[regions]
    offs = match_regions(img, regions, np.zeros(img.shape, dtype=bool), band_px=8)
    out = img - offs[regions]
    resid = out - truth
    lv = [float(np.median(resid[regions == r])) for r in range(3)]
    # Пьедесталы с разбросом 4σ сшиваются в ≥16 раз; остаток — шум короткой синтетической границы.
    assert max(lv) - min(lv) < 0.25, lv
    assert np.isfinite(offs).all()
    assert abs(float(np.median(resid[150:210, 130:190])) - lv[0]) < 0.25  # туманность сдвинута как фон


def test_tile_pedestal_anchors_masked_tile_to_neighbours():
    """Яркая плитка целиком под маской источников: раньше получала поправку ближайшей соседки и
    оставалась прямоугольником; теперь — по ступеньке на границе с измеренными соседями."""
    from skyforge.background import tile_pedestal
    from skyforge.config import BackgroundConfig

    rng = np.random.default_rng(9)
    tiles = np.zeros((300, 300), dtype=np.int32)
    tiles[:, 150:] = 1
    tiles[100:180, 60:130] = 2
    ped = np.array([0.0, 0.4, 3.0], dtype=np.float32)            # плитка 2 — на 3σ: маска закрывает её целиком
    a = (rng.normal(0, 1, tiles.shape) + 5 + ped[tiles]).astype(np.float32)
    c = BackgroundConfig(subtract=True, tile_mask_dilate_arcsec=0.3, tile_min_area_arcsec2=0.05,
                         mask_smooth_arcsec=0.2, mask_sigma=1.5)
    p, info = tile_pedestal(a, tiles, c, scale_arcsec=0.05)
    out = a - p
    lv = [float(np.median(out[tiles == k])) for k in range(3)]
    assert info["matched"] >= 1
    assert max(lv) - min(lv) < 0.25, lv


# --- доработка 3d: растяжка по маскам, поправка каналов с весом, цепочка плиток, сравнение с принятым ---

def _nebula_scene(rng, sky_rows=200):
    """Туманность в центре, пустое небо вокруг; sky_rows — сколько строк пустого неба добавить снизу."""
    h, w = 200 + sky_rows, 200
    yy, xx = np.indices((h, w))
    img = rng.normal(0, 1, (h, w)).astype(np.float32)
    img += (60 * np.exp(-((yy - 100) ** 2 + (xx - 100) ** 2) / (2 * 30**2))).astype(np.float32)
    nebula = np.hypot(yy - 100, xx - 100) < 70
    sky = np.hypot(yy - 100, xx - 100) > 85
    return img, sky, nebula


def test_mask_anchored_stretch_ignores_background_changes():
    from skyforge.config import StretchConfig
    from skyforge.stretch import stretch_array

    rng = np.random.default_rng(21)
    img, sky, neb = _nebula_scene(rng)
    s = StretchConfig(black=1.0, white=99.5, a=0.1, anchor="masks")
    ref, st = stretch_array(img, s, masks=(sky, neb))
    assert st["anchor"] == "masks"
    # 1) постоянное смещение всего кадра (фон опустился после очистки) — туманность та же
    shifted, _ = stretch_array(img - 3.0, s, masks=(sky, neb))
    assert np.allclose(shifted[neb], ref[neb], atol=1e-5)
    # 2) пустого неба в кадре стало вдвое больше (другая обрезка) — по маскам не меняется, по кадру меняется
    big, sky2, neb2 = _nebula_scene(np.random.default_rng(21), sky_rows=600)
    a_masks, _ = stretch_array(big, s, masks=(sky2, neb2))
    a_frame, _ = stretch_array(big, s.model_copy(update={"anchor": "frame"}))
    b_frame, _ = stretch_array(img, s.model_copy(update={"anchor": "frame"}))
    assert np.allclose(a_masks[:400][neb], ref[neb], atol=0.01)
    assert not np.allclose(a_frame[:400][neb], b_frame[neb], atol=0.01)


def test_equalize_sky_weight_keeps_nebula_color():
    from skyforge.compose import equalize_sky, sky_weight

    yy, xx = np.indices((200, 200))
    sd = (60 - np.hypot(yy - 100, xx - 100)).astype(np.float32)        # туманность радиуса 60
    rgb = np.zeros((200, 200, 3), dtype=np.float32) + np.array([0.10, 0.14, 0.09], dtype=np.float32)
    neb_color = np.array([0.6, 0.3, 0.2], dtype=np.float32)
    rgb[sd > 0] += neb_color
    sky = sd < -25
    out, _, _ = equalize_sky(rgb, sky, sky_weight(sd, 20, rgb.shape))
    med = [float(np.median(out[..., k][sky])) for k in range(3)]
    assert max(med) - min(med) < 0.001                                  # фон нейтральный
    assert np.allclose(out[100, 100], rgb[100, 100], atol=1e-6)          # цвет туманности не тронут
    corr = (out - rgb)[100, :, 1]                                        # сама поправка поперёк края
    assert np.abs(np.diff(corr)).max() < 0.005                           # мягкий край, без ступеньки


def test_propagate_chain_of_strips_from_large_to_small():
    """Цепочка узких полос без своего неба выравнивается от широкой измеренной плитки внутрь."""
    from skyforge.background import boundary_steps, propagate

    rng = np.random.default_rng(4)
    regions = np.zeros((200, 400), dtype=np.int32)
    for k, x0 in enumerate(range(200, 400, 40), start=1):               # пять узких полос справа
        regions[:, x0:x0 + 40] = k
    ped = np.array([0.0, 0.8, -0.5, 1.2, 0.3, -0.9], dtype=np.float32)
    img = (rng.normal(0, 1, regions.shape) + ped[regions]).astype(np.float32)
    known = np.full(6, np.nan, dtype=np.float32)
    known[0] = 0.0                                                       # измерена только широкая
    steps = boundary_steps(img, regions, np.zeros(regions.shape, bool), band_px=8)
    area = np.bincount(regions.ravel())
    out = propagate(known, steps, area)
    assert np.isfinite(out).all()
    assert np.abs(out - ped).max() < 0.25, (out, ped)


def test_baseline_compare_reports_texture_and_crops(tmp_path):
    from PIL import Image

    from skyforge.baseline import compare, save_baseline
    from skyforge.paths import Paths

    paths = Paths(tmp_path)
    run = tmp_path / "run"
    run.mkdir()
    rng = np.random.default_rng(2)
    base = (rng.random((120, 160, 3)) * 255).astype(np.uint8)
    Image.fromarray(base).save(run / "obj_alpha.png")
    saved = save_baseline(paths, "obj", "phase-x", run)
    assert [p.name for p in saved] == ["alpha_bin1.png"]
    cfg = Config.model_validate({"target": "obj", "style": "alpha", "report": {
        "baseline": "phase-x", "texture_ring": [5, 40], "crops": [{"name": "core", "x": 80, "y": 60, "size": 40}]}})
    flat = np.full_like(base, 128)                                       # фактура «выжжена»
    Image.fromarray(flat).save(tmp_path / "now.png")
    info = compare(cfg, paths, tmp_path, tmp_path / "now.png")
    assert info["texture_change_percent"] < -90
    assert (tmp_path / "baseline_diff.png").exists() and (tmp_path / "crops_compare.png").exists()
    cfg2 = cfg.model_validate({**cfg.model_dump(mode="json"), "report": {"baseline": "nope"}})
    assert "нет принятой" in compare(cfg2, paths, tmp_path, tmp_path / "now.png")["skipped"]


# --- фаза 3e: плоскость вместо константы ---

def _tiled_sky(rng, with_nebula=True):
    """Плитки со своим уровнем и наклоном: две широкие, три узкие полосы дизеринга, туманность сверху."""
    h, w = 300, 420
    yy, xx = np.indices((h, w))
    tiles = np.zeros((h, w), dtype=np.int32)
    tiles[:, 200:] = 1
    tiles[:, 180:200] = 2                      # узкие полосы — наклон не измерить
    tiles[:, 380:400] = 3
    tiles[150:170, :180] = 4
    lev = rng.uniform(-1.5, 1.5, 5)
    sx, sy = rng.uniform(-0.006, 0.006, 5), rng.uniform(-0.006, 0.006, 5)  # до ~2σ на ширину плитки
    bg = (lev[tiles] + sx[tiles] * xx + sy[tiles] * yy).astype(np.float32)
    img = rng.normal(0, 1, (h, w)).astype(np.float32) + 10 + bg
    if with_nebula:
        img += (40 * np.exp(-((yy - 80) ** 2 + (xx - 300) ** 2) / (2 * 30**2))).astype(np.float32)
    return img, tiles, bg


def test_tile_planes_remove_level_and_slope_steps():
    from skyforge.background import boundary_jumps, tile_pedestal, tile_planes
    from skyforge.config import BackgroundConfig

    rng = np.random.default_rng(13)
    img, tiles, bg = _tiled_sky(rng)
    yy, xx = np.indices(img.shape)
    sky = np.hypot(yy - 80, xx - 300) > 110
    c = BackgroundConfig(subtract=True, tile_mask_dilate_arcsec=0.5, tile_min_area_arcsec2=0.5,
                         mask_smooth_arcsec=0.3, tile_min_extent_arcsec=2.0)
    before, _ = boundary_jumps(img, tiles, sky, 8)
    corr, info = tile_planes(img, tiles, c, scale_arcsec=0.05)
    after, nb = boundary_jumps(img - corr, tiles, sky, 8)
    const, _ = tile_pedestal(img, tiles, c, scale_arcsec=0.05)
    after_const, _ = boundary_jumps(img - const, tiles, sky, 8)
    assert nb >= 4
    assert after < 0.2, f"скачки на границах {before:.2f}σ → {after:.2f}σ"
    assert after < after_const                 # плоскость лучше константы при наклоне
    assert info["measured"] >= 2 and info["matched"] >= 2   # узкие полосы — от соседей
    # туманность не съедена: её пик над исправленным фоном тот же
    clean = img - corr
    assert abs((clean[80, 300] - np.median(clean[sky])) - (40 + 0)) < 3


def test_fit_plane_ignores_outliers():
    from skyforge.background import fit_plane

    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 100, 4000), rng.uniform(0, 100, 4000)
    v = 2 + 0.03 * x - 0.02 * y + rng.normal(0, 0.1, 4000)
    v[:200] += 50                               # звёзды/лучи
    a, b, c = fit_plane(x, y, v)
    assert abs(a - 2) < 0.05 and abs(b - 0.03) < 0.002 and abs(c + 0.02) < 0.002
