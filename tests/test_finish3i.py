"""Фаза 3i: звёзды отдельно от туманности, локальный контраст по маске туманности."""

from __future__ import annotations

import numpy as np

from skyforge.config import FinishConfig
from skyforge.finish import finish_array


def _scene(n=256):
    yy, xx = np.indices((n, n))
    neb = 0.3 + 0.1 * np.sin(xx / 9.0) * np.cos(yy / 13.0)                 # туманность с деталью
    star = 0.6 * np.exp(-((yy - 128) ** 2 + (xx - 100) ** 2) / (2 * 2.0**2))
    rgb = np.clip(np.stack([neb + star, neb * 0.8 + star, neb * 0.6 + star], -1), 0, 1).astype(np.float32)
    weight = (np.hypot(yy - 128, xx - 100) <= 10).astype(np.float32)        # маска звезды с ореолом
    sd = (90 - np.hypot(yy - 128, xx - 128)).astype(np.float32)            # туманность радиуса 90
    return rgb, weight, sd, yy, xx


def test_defaults_keep_finish_as_before():
    rgb, w, sd, *_ = _scene()
    f = FinishConfig(denoise=0.1, sharpen=0.5)
    assert np.array_equal(finish_array(rgb, f, w, sd), finish_array(rgb, f))


def test_star_scale_dims_star_without_dark_ring():
    rgb, w, sd, yy, xx = _scene()
    f = FinishConfig.model_validate({"stars": {"scale": 0.5}})
    out = finish_array(rgb, f, w, sd)
    peak0 = rgb[128, 100].mean() - 0.3
    peak1 = out[128, 100].mean() - 0.3
    assert 0.4 * peak0 < peak1 < 0.65 * peak0                             # звезда вдвое тише
    ring = (np.hypot(yy - 128, xx - 100) > 6) & (np.hypot(yy - 128, xx - 100) < 18)  # ореол маски
    assert (out[ring] - rgb[ring]).min() > -0.02                          # без тёмного кольца
    far = np.hypot(yy - 128, xx - 100) > 25
    assert np.allclose(out[far], rgb[far], atol=1e-6)                     # вне маски — как было


def test_local_contrast_inside_nebula_only_and_white_safe():
    rgb, w, sd, yy, xx = _scene()
    rgb[60:70, 60:70] = 0.98                                              # почти белое пятно в туманности
    f = FinishConfig.model_validate({"local_contrast": {"amount": 0.6, "radius": 20}})
    out = finish_array(rgb, f, None, sd)
    inside = sd > 40
    assert out[inside].std() > rgb[inside].std() * 1.05                   # деталь сильнее
    assert np.allclose(out[sd <= 0], rgb[sd <= 0], atol=1e-6)             # вне маски ноль
    assert out.max() <= 1.0 and out[60:70, 60:70].max() <= 1.0


def test_white_point_ignores_bright_star_in_nebula_mask():
    """Белая точка по туманности: добавление яркой звезды в маску туманности её не меняет, если звезда
    исключена (маска звёзд и диск вокруг звезды с насыщенным ядром)."""
    from skyforge.clean import saturated_stars
    from skyforge.config import StretchConfig
    from skyforge.stretch import stretch_array

    rng = np.random.default_rng(1)
    n = 400
    yy, xx = np.indices((n, n))
    img = (1.0 + rng.normal(0, 0.05, (n, n))).astype(np.float32)
    neb = np.hypot(yy - 200, xx - 200) < 150
    img[neb] += (0.5 + 0.3 * np.sin(xx[neb] / 7.0)).astype(np.float32)
    sky = ~neb
    s = StretchConfig(white=99.6, black=0.5)
    w0 = stretch_array(img, s, masks=(sky, neb))[1]["white_value"]
    star = img + (5000 * np.exp(-((yy - 230) ** 2 + (xx - 180) ** 2) / (2 * 6.0**2))
                  + 50 / (1 + np.hypot(yy - 230, xx - 180) / 5) ** 3).astype(np.float32)
    w_star = stretch_array(star, s, masks=(sky, neb))[1]["white_value"]
    assert w_star > 1.5 * w0                                               # звезда в выборке задирает белую точку
    holes = {"A": np.hypot(yy - 230, xx - 180) < 6, "B": np.hypot(yy - 230, xx - 180) < 5}
    (r, c, area, nf), = saturated_stars(holes, 50, 2)
    disk = (yy - r) ** 2 + (xx - c) ** 2 <= 150**2
    w_excl = stretch_array(star, s, masks=(sky, neb & ~disk))[1]["white_value"]
    w0_excl = stretch_array(img, s, masks=(sky, neb & ~disk))[1]["white_value"]
    assert abs(w_excl - w0_excl) < 1e-2 * w0_excl                          # со звездой и без — одинаково


def test_shadows_and_saturation_by_max_never_push_channel_to_one():
    """3i: подъём теней по самому яркому каналу и насыщенность с ограничением — канал не выходит за 1,
    оттенок при подъёме не меняется; режимы luma/clip (measured) — как раньше."""
    from skyforge.compose import saturate, shadows

    rng = np.random.default_rng(4)
    rgb = rng.uniform(0, 0.97, (200, 200, 3)).astype(np.float32)
    rgb[:100, :, 2] = rng.uniform(0.6, 0.97, (100, 200))                   # яркая голубая область
    lifted = shadows(rgb, 4.0, "max")
    assert lifted.max() < 1.0
    ratio = lifted / np.maximum(rgb, 1e-6)
    assert np.allclose(ratio[..., 0], ratio[..., 2], rtol=1e-4)            # один множитель на пиксель
    sat = saturate(lifted, 1.4, "limit")
    assert sat.max() < 1.0 and sat.min() >= 0
    assert np.allclose(sat.max(-1), lifted.max(-1), atol=1e-5)               # самый яркий канал не тронут
    old = shadows(rgb, 4.0)                                                # luma: синий упирается в 1
    assert (old[:100, :, 2] >= 0.999).mean() > 0.3


def test_saturated_star_layer_has_one_shape_and_one_color():
    from skyforge.config import SaturatedStarsConfig
    from skyforge.satstars import star_layers

    n = 400
    yy, xx = np.indices((n, n))
    r = np.hypot(yy - 200, xx - 200)
    neb = 1.0 + 0.2 * np.sin(xx / 11.0)
    psf = 2000 * np.exp(-r**2 / (2 * 3.0**2)) + 4000 / (1 + r / 4) ** 3   # крыло в кольце 20–60 px ≫ туманности
    a = (neb + psf).astype(np.float32)                                   # фильтр A
    b = (2 * neb + 0.5 * psf).astype(np.float32)                         # фильтр B: звезда вчетверо «краснее» туманности
    lin, disp, info = star_layers({"A": a, "B": b}, {"A": 2.0, "B": 4.0}, [[200, 200, 100, 2]], SaturatedStarsConfig())
    col = info[0]["color"]
    assert col["A"] == 1.0 and 0.2 < col["B"] < 0.3                       # B/A в единицах белых точек: 0.5/2 = 0.25
    m = disp["A"] > 0.05
    assert np.allclose(disp["B"][m] / disp["A"][m], col["B"], rtol=0.02)   # один цвет во всех пикселях
    rest = a - lin["A"]
    assert rest[(r > 20) & (r < 150)].min() > 0.3                        # туманность не провалена
