"""3б (Hubble WFPC2): космические лучи в одиночной экспозиции и швы чипов — синтетика."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from skyforge.config import CosmicsConfig
from skyforge.cosmics import fill_gaps_linear, find_cosmics, quiet_sky, remove_cosmics


def _scene(seed: int = 1):
    """Три фильтра: шум, звёзды с PSF (σ 1 px) во всех трёх, лучи — свои в каждом фильтре."""
    rng = np.random.default_rng(seed)
    H = W = 240
    yy, xx = np.mgrid[0:H, 0:W]
    stars = [(int(rng.integers(15, H - 15)), int(rng.integers(15, W - 15)), float(rng.uniform(15, 200))) for _ in range(20)]
    layers, hits = {}, {}
    for k, f in enumerate(("F502N", "F656N", "F658N")):
        a = 10 + rng.normal(0, 1.0, (H, W))
        for y, x, amp in stars:
            a += amp * (0.5 + 0.5 * k / 2) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0)
        m = np.zeros((H, W), dtype=bool)
        for _ in range(40):                                   # одиночные пиксели
            y, x = int(rng.integers(3, H - 3)), int(rng.integers(3, W - 3))
            m[y, x] = True
        for _ in range(12):                                   # чёрточки 3–7 px под разными углами
            y, x = int(rng.integers(10, H - 10)), int(rng.integers(10, W - 10))
            ang, n = rng.uniform(0, np.pi), int(rng.integers(3, 8))
            for t in range(n):
                m[int(round(y + t * np.sin(ang))), int(round(x + t * np.cos(ang)))] = True
        for _ in range(8):                                    # длинные следы 10–25 px толщиной 1–2 px
            y, x = int(rng.integers(30, H - 30)), int(rng.integers(30, W - 30))
            ang, n, wd = rng.uniform(0, np.pi), int(rng.integers(10, 26)), int(rng.integers(1, 3))
            for t in range(n):
                for o in range(wd):
                    m[int(round(y + t * np.sin(ang) + o * np.cos(ang))), int(round(x + t * np.cos(ang) - o * np.sin(ang)))] = True
        for _ in range(6):                                    # пятна 2×2 и 3×3
            y, x, sz = int(rng.integers(5, H - 5)), int(rng.integers(5, W - 5)), int(rng.integers(2, 4))
            m[y:y + sz, x:x + sz] = True
        near_star = np.zeros((H, W), dtype=bool)
        for y, x, _ in stars:
            near_star |= np.hypot(yy - y, xx - x) < 5
        m &= ~near_star                                       # луч на звезде не размечаем — честная проверка
        a[m] += rng.uniform(40, 300, int(m.sum()))
        if f == "F502N":                                      # протяжённый узел только в одной линии — не луч
            a += 30 * np.exp(-((yy - 60) ** 2 + (xx - 200) ** 2) / (2 * 4.0 ** 2))
        layers[f], hits[f] = a.astype(np.float32), m
    return layers, hits, stars, (yy, xx)


def test_cosmics_removes_hits_and_keeps_stars():
    """Лучи уходят, звёзды (есть во всех фильтрах) и плавный узел одной линии целы. Четыре сцены: в среднем ≥ 95%
    следов, в худшем фильтре ≥ 85% — пересечение следов из всех трёх фильтров в одном месте по сверке фильтров
    неотличимо от звезды (сцена 1: следы F656N и F658N крест-накрест и точка F502N)."""
    caught_all = []
    for seed in (1, 2, 3, 4):
        layers, hits, stars, (yy, xx) = _scene(seed)
        masks = find_cosmics(layers, CosmicsConfig(clean=True), 0.091)
        core = np.zeros(yy.shape, dtype=bool)
        for y, x, _ in stars:
            core |= np.hypot(yy - y, xx - x) < 3
        for f, a in layers.items():
            caught = (masks[f] & hits[f]).sum() / hits[f].sum()
            caught_all.append(caught)
            assert caught > 0.85, f"сцена {seed}, {f}: поймано {caught:.2f} следов"
            assert not (masks[f] & core).any(), f"сцена {seed}, {f}: задеты звёзды"
            clean = remove_cosmics(a, masks[f])
            for y, x, _ in stars:
                win = (slice(y - 3, y + 4), slice(x - 3, x + 4))
                assert abs(clean[win].sum() - a[win].sum()) / a[win].sum() < 0.02
            assert (clean[hits[f] & masks[f]] - 10).std() < 3      # на месте удалённых следов — уровень неба
        knot = np.hypot(yy - 60, xx - 200) < 6
        assert not masks["F502N"][knot].any(), "узел, яркий только в одной линии, принят за луч"
    assert np.mean(caught_all) > 0.95


def test_fill_gaps_linear_fills_seams_not_edges():
    H, W = 200, 260
    yy, xx = np.mgrid[0:H, 0:W]
    a = (5 + 0.03 * xx + 0.02 * yy).astype(np.float32)
    truth = a.copy()
    seam = np.abs((xx - 130) - 0.3 * (yy - 100)) < 2.5          # наклонный шов шириной ~5 px
    dash = (yy == 40) & (xx > 20) & (xx < 31)                  # отбракованная чёрточка
    wide = (np.abs(yy - 160) < 9) & (xx < 60)                  # полоса 17 px — шире предела
    corner = (yy + xx) < 30                                    # угол вне поля
    a[seam | dash | wide | corner] = np.nan
    out, filled = fill_gaps_linear(a, 6)
    assert filled[seam & ~corner].mean() > 0.98 and filled[dash].all()
    assert np.abs(out[seam & filled] - truth[seam & filled]).max() < 0.05   # линейно поперёк шва
    assert not filled[corner].any()
    assert not filled[wide & (np.abs(yy - 160) < 7)].any()
    assert np.isnan(out[corner]).all()


def test_cosmics_trusted_filter_protects_stars_missing_in_one_line():
    """Звезда, слабая в [O III] (видна в F656N и F658N, нет в F502N), не принимается за луч, если F658N свободен
    от лучей (сложен из двух экспозиций, skip_filters): то, что в нём видно, — настоящее. Лучи F656N остаются лучами."""
    rng = np.random.default_rng(7)
    H = W = 160
    yy, xx = np.mgrid[0:H, 0:W]
    star = 120 * np.exp(-((yy - 80) ** 2 + (xx - 80) ** 2) / 2.0)
    layers = {"F502N": 10 + rng.normal(0, 1, (H, W)),
              "F656N": 10 + rng.normal(0, 1, (H, W)) + star,
              "F658N": 10 + rng.normal(0, 1, (H, W)) + star}
    hit = np.zeros((H, W), dtype=bool)
    hit[30, 40:52] = True
    layers["F656N"][hit] += 200
    layers = {f: a.astype(np.float32) for f, a in layers.items()}
    masks = find_cosmics(layers, CosmicsConfig(clean=True, skip_filters=["F658N"]), 0.091)
    assert not masks["F656N"][np.hypot(yy - 80, xx - 80) < 4].any(), "звезда в F656N принята за луч"
    assert masks["F656N"][hit].mean() > 0.9


def _sky_scene(seed: int):
    """Небо трёх фильтров: звёзды разной яркости (в F656N — 0,66 сигнала F658N по шуму, в F502N — 0,4), в F658N —
    свои артефакты, совпавшие в двух экспозициях без сдвига: одиночные пиксели, пары и короткие следы до 4 px."""
    rng = np.random.default_rng(seed)
    H = W = 240
    yy, xx = np.mgrid[0:H, 0:W]
    snr = np.geomspace(6, 150, 18)
    stars = [(int(rng.integers(12, H - 12)), int(rng.integers(12, W - 12)), float(s)) for s in snr]
    layers = {}
    for f, k in (("F502N", 0.4), ("F656N", 0.66), ("F658N", 1.0)):
        a = 10 + rng.normal(0, 1.0, (H, W))
        for y, x, s in stars:
            a += k * s * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0)
        layers[f] = a
    near = np.zeros((H, W), dtype=bool)
    for y, x, _ in stars:
        near |= np.hypot(yy - y, xx - x) < 6
    hit = np.zeros((H, W), dtype=bool)
    for _ in range(30):
        y, x = int(rng.integers(5, H - 5)), int(rng.integers(5, W - 5))
        n = int(rng.choice([1, 1, 2, 3, 4]))
        ang = rng.uniform(0, np.pi)
        for t in range(n):
            hit[int(round(y + t * np.sin(ang))), int(round(x + t * np.cos(ang)))] = True
    hit &= ~near
    layers["F658N"][hit] += rng.uniform(15, 150, int(hit.sum()))
    return {f: a.astype(np.float32) for f, a in layers.items()}, hit, stars, (yy, xx)


def test_cosmics_sky_pass_cleans_trusted_filter_points_absent_elsewhere():
    """3б шаг 3: на небе в F658N (две экспозиции без сдвига) остаются точки, совпавшие в обеих экспозициях. Во втором
    проходе по небу F658N тоже чистится: пятно — луч, если его нет ни в одном из двух других фильтров. Звёзды
    (есть в F656N слабее, в F502N ещё слабее) не задеты, включая самые слабые; без флага F658N не чистится."""
    soft = CosmicsConfig(clean=True, skip_filters=["F658N"])   # пороги первого прохода — в clean F658N чистится с ними
    caught_all = []
    for seed in (11, 12, 13):
        layers, hit, stars, (yy, xx) = _sky_scene(seed)
        assert not find_cosmics(layers, soft, 0.091)["F658N"].any()
        m = find_cosmics(layers, soft, 0.091, check_trusted=True)["F658N"]
        caught = (m & hit).sum() / hit.sum()
        caught_all.append(caught)
        assert caught > 0.9, f"сцена {seed}: поймано {caught:.2f} точек F658N"
        for y, x, s in stars:
            assert not m[np.hypot(yy - y, xx - x) < 3].any(), f"сцена {seed}: задета звезда {s:.0f}σ в ({y}, {x})"
    assert np.mean(caught_all) > 0.95


def test_quiet_sky_excludes_faint_halo_beyond_nebula_mask():
    """Внешняя оболочка за маской туманности: плавный подъём фона (8σ пикселя у маски, к нулю за 90 px) — не «тихое небо»,
    точки на ней F658N не чистятся. Пустое небо и яркий одиночный артефакт на нём (медиана 25 px его не видит) — тихие."""
    rng = np.random.default_rng(5)
    H, W = 300, 300
    yy, xx = np.mgrid[0:H, 0:W]
    a = 10 + rng.normal(0, 1.0, (H, W))
    r = np.hypot(yy - 150, xx - 150)
    a += 8.0 * np.clip((150 - r) / 90, 0, 1) + 40 * (r < 60)       # туманность r < 60 (маска), оболочка 60–150
    a[30, 40] += 300                                              # артефакт на пустом небе
    sky = r > 70                                                  # небо по маске туманности: оболочка в него входит
    q = quiet_sky(a.astype(np.float32), sky, 4.0)
    assert q[(r > 200) & (np.abs(yy - 30) + np.abs(xx - 40) > 0)].mean() > 0.95
    assert q[30, 40]
    assert not q[(r > 72) & (r < 100)].any(), "оболочка за маской принята за тихое небо"   # фон ≥ 4,4σ


def test_sky_stars_found_by_local_noise_only_on_quiet_sky():
    """3б шаг 3: звёзды поля на небе ищутся по шуму неба, а не по σ всего кадра (в тесной рамке Hubble туманность
    раздувала её в 8–17 раз — из 8 звёзд неба в маску попадала одна). Звёзды на туманности и пики на внешней оболочке
    за маской в маску не идут; на пустом небе нет ложных звёзд."""
    from skyforge.config import StarsConfig
    from skyforge.stars import sky_stars

    rng = np.random.default_rng(9)
    H = W = 400
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - 200, xx - 200)
    a = 10 + rng.normal(0, 1.0, (H, W)) + 400 * (r < 110) + 30 * np.clip((180 - r) / 70, 0, 1)   # туманность + оболочка
    sky_st = [(20, 30, 25.0), (370, 40, 60.0), (30, 360, 200.0), (380, 380, 40.0), (200, 385, 30.0)]
    other = [(200, 200, 300.0), (150, 260, 300.0), (200, 340, 60.0)]   # на туманности и на оболочке за маской (r 140)
    for y, x, amp in sky_st + other:
        a += amp * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0)
    outside = r > 120
    quiet = quiet_sky(a.astype(np.float32), outside, 4.0)
    m = sky_stars(a.astype(np.float32), outside, quiet, StarsConfig(), 0.091)
    for y, x, _ in sky_st:
        assert m[y, x], f"звезда неба ({y}, {x}) не найдена"
    for y, x, _ in other:
        assert not m[np.hypot(yy - y, xx - x) < 4].any(), f"({y}, {x}) — не звезда неба"
    lab, n = ndimage.label(m)
    assert n == len(sky_st), f"ложных звёзд: {n - len(sky_st)}"


def test_star_saturation_scope_sky_keeps_stars_on_nebula():
    """color.star_saturation_scope: sky — обесцвечиваются только звёзды, центр которых вне маски туманности; звезда
    на туманности и её ореол не меняются, звезда неба сереет целиком (и та часть ореола, что заходит за край маски)."""
    from skyforge.compose import scope_to_sky
    from skyforge.stars import StarMask

    H = W = 120
    yy, xx = np.mgrid[0:H, 0:W]
    sd = (60 - xx).astype(np.float32)                      # туманность слева (x < 60), край маски — x = 60
    w = np.zeros((H, W), dtype=np.float32)
    for y, x in ((30, 40), (90, 64)):                      # звезда на туманности; звезда неба у самого края маски
        w = np.maximum(w, np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3.0 ** 2)).astype(np.float32))
    out = scope_to_sky(StarMask(w, 15, "F658N"), sd)
    assert out.weight[np.hypot(yy - 30, xx - 40) < 12].max() == 0
    near = (np.hypot(yy - 90, xx - 64) < 12) & (w > 0.01)          # хвосты веса < 0,01 (< 1% обесцвечивания) — 0
    assert np.array_equal(out.weight[near], w[near])


def test_finish_nodata_excludes_filled_seams():
    """3б шаг 4: шов, пустой во всех фильтрах (покрытие 0), после заполнения в стиле — данные, а не nodata: финиш его
    не обнуляет и в PNG он не прозрачный (была тёмная линия через оболочку). Углы вне поля остаются nodata."""
    from skyforge.finish import nodata_mask

    H, W = 200, 260
    yy, xx = np.mgrid[0:H, 0:W]
    cov = np.full((H, W), 3, dtype=np.int16)
    seam = np.abs((xx - 130) - 0.3 * (yy - 100)) < 2.5
    corner = (yy + xx) < 30
    cov[seam | corner] = 0
    assert np.array_equal(nodata_mask(cov, 0), seam | corner)
    nd = nodata_mask(cov, 6)
    assert not nd[seam & ~ndimage.binary_dilation(corner, iterations=8)].any()
    assert nd[corner].all()


def test_cosmics_true_sigma_of_smoothed_map_stops_chance_protection():
    """3б шаг 4: шум drizzle-кадров коррелирован (WFPC2: σ среднего 3×3 — 0,45σ пикселя, а не 0,33), порог «источник
    есть» 5σ/3 был на деле 3,7σ среднего — шум фильтра без лучей случайно защищал ~2,4% точек неба, на небе итога
    оставалось 65 следов. С σ, измеренной по самой карте средних 3×3, случайной защиты нет; звёзды целы."""
    rng = np.random.default_rng(21)
    H = W = 300
    yy, xx = np.mgrid[0:H, 0:W]

    def corr_noise():
        n = ndimage.uniform_filter(rng.normal(0, 1.0, (H, W)), 2)    # как drizzle: соседние пиксели связаны
        return n / n.std()

    stars = [(int(rng.integers(15, H - 15)), int(rng.integers(15, W - 15)), float(s)) for s in np.geomspace(15, 200, 15)]
    layers = {}
    for f, k in (("F502N", 0.5), ("F656N", 0.7), ("F658N", 1.0)):
        a = 10 + corr_noise()
        for y, x, s in stars:
            a += k * s * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / 2.0)
        layers[f] = a
    near = np.zeros((H, W), dtype=bool)
    for y, x, _ in stars:
        near |= np.hypot(yy - y, xx - x) < 6
    hits = {}
    for f in ("F502N", "F656N"):
        m = np.zeros((H, W), dtype=bool)
        for _ in range(1500):
            m[int(rng.integers(3, H - 3)), int(rng.integers(3, W - 3))] = True
        m &= ~near
        layers[f][m] += rng.uniform(40, 200, int(m.sum()))
        hits[f] = m
    layers = {f: a.astype(np.float32) for f, a in layers.items()}
    c = CosmicsConfig(clean=True, skip_filters=["F658N"], sigma=3.5, sharpness=1.5, max_area_arcsec2=6.0)
    old = find_cosmics(layers, c, 0.091)
    new = find_cosmics(layers, c, 0.091, smoothed_sigma=True)
    core = np.zeros((H, W), dtype=bool)
    for y, x, _ in stars:
        core |= np.hypot(yy - y, xx - x) < 3
    for f, m in hits.items():
        c_old, c_new = (old[f] & m).sum() / m.sum(), (new[f] & m).sum() / m.sum()
        assert c_new >= 0.999 and c_new > c_old, f"{f}: поймано {c_old:.4f} → {c_new:.4f}"   # было 0,9945 / 0,9890
        assert not (new[f] & core).any(), f"{f}: задеты звёзды"


def test_star_saturation_scope_sky_merged_halos_decided_per_star():
    """Ревью шага 4: ореолы звезды на туманности у края маски и звезды неба рядом сливаются в одну область веса —
    решение «звезда неба или нет» принимается по каждой звезде (ядро веса > 0,5), а не по всей слившейся области;
    исход не зависит от того, какая звезда на строку выше."""
    from skyforge.compose import scope_to_sky
    from skyforge.stars import StarMask

    H, W = 140, 200
    yy, xx = np.mgrid[0:H, 0:W]
    sd = (105 - xx).astype(np.float32)                       # туманность x < 105
    for neb_star, sky_star in (((60, 85), (62, 125)), ((62, 85), (60, 125))):
        disk = np.zeros((H, W), dtype=bool)
        for y, x in (neb_star, sky_star):
            disk |= np.hypot(yy - y, xx - x) <= 16                 # ореол 1,5″ при 0,091″/px
        w = ndimage.gaussian_filter(disk.astype(np.float32), 16 / 3)
        lab, n = ndimage.label(w > 0.01)
        assert n == 1                                          # ореолы слились
        out = scope_to_sky(StarMask(w, 15, "F658N"), sd).weight
        assert out[np.hypot(yy - neb_star[0], xx - neb_star[1]) < 8].max() == 0, "звезда на туманности обесцвечена"
        core = np.hypot(yy - sky_star[0], xx - sky_star[1]) < 8
        assert np.array_equal(out[core], w[core]), "звезда неба не обесцвечена"


def test_saturation_above_sky_keeps_sky_noise_off_zero_and_nebula_unchanged():
    """3б закрытие: насыщенность только над уровнем неба (color.saturation_above_sky — разгон в уровнях). Небо на
    уровне 6 с шумным синим каналом насыщенность не раздвигает за 0 (было 1,24% нулей в синем); пиксель туманности
    выше неба + разгон — ровно как при обычной насыщенности."""
    from skyforge.compose import saturate

    rng = np.random.default_rng(3)
    sky = np.stack([np.full(20000, 6.0), np.full(20000, 6.0), 6 + rng.normal(0, 3.0, 20000)], -1) / 255
    sky = np.clip(sky, 0, 1).astype(np.float32)[None]
    old = saturate(sky, 1.2, "limit")
    new = saturate(sky, 1.2, "limit", level=6 / 255, ramp=12 / 255)
    zeros = lambda x: (np.round(x[..., 2] * 255) <= 0).mean()
    assert zeros(new) <= zeros(sky) + 0.005 and zeros(old) > zeros(sky) + 0.02   # нулей не добавляет (было +2,9%)
    neb = np.array([[[0.6, 0.4, 0.1], [0.2, 0.5, 0.7], [0.30, 0.25, 0.2]]], dtype=np.float32)
    assert np.allclose(saturate(neb, 1.2, "limit", level=6 / 255, ramp=12 / 255), saturate(neb, 1.2, "limit"))
    assert np.array_equal(saturate(sky, 1.2, "limit", level=6 / 255, ramp=0.0), old)


def test_finish_nodata_fill_sky_level_keeps_alpha():
    """3б закрытие: углы без данных в showcase заливаются ровным уровнем неба (без шума), только там, где нет ни одного
    фильтра; в PNG они по-прежнему прозрачные по альфе; black — как раньше (measured)."""
    from skyforge.finish import fill_nodata

    rgb = np.full((10, 12, 3), 0.3, dtype=np.float32)
    nodata = np.zeros((10, 12), dtype=bool)
    nodata[:3, :4] = True
    out = fill_nodata(rgb, nodata, "sky", 6 / 255)
    assert np.allclose(out[nodata], 6 / 255) and np.array_equal(out[~nodata], rgb[~nodata])
    assert np.all(fill_nodata(rgb, nodata, "black", 6 / 255)[nodata] == 0)


def test_finish_nodata_excludes_cores_filled_by_clean():
    """Фаза 4 (Sh2-305): ядро звезды пусто во всех фильтрах на этапе align (покрытие 0), clean его заполнил — это
    данные; финиш не делает его прозрачным (была чёрная точка в центре яркой звезды). Пустота вне поля — nodata."""
    from skyforge.finish import nodata_mask

    cov = np.full((50, 60), 4, dtype=np.int16)
    cov[20:24, 30:34] = 0                                  # ядро
    cov[:, :5] = 0                                         # край вне поля
    filled = np.ones((50, 60), dtype=bool)
    filled[:, :5] = False
    nd = nodata_mask(cov, 0, filled)
    assert not nd[20:24, 30:34].any() and nd[:, :5].all()
    assert np.array_equal(nodata_mask(cov, 0), cov == 0)   # без карты заполненного — как раньше
