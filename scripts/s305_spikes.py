"""Фаза 4, Sh2-305, шаг 3 п.2: проба обесцвечивания лучей яркой звезды — маска звезды, расширенная вдоль шести лучей
(направления — по пикам яркости F444W в кольце 12–30″), в пределах ореола 30″; обесцвечивание тем же механизмом, что
звёзды поля (compose.desaturate_stars, star_saturation 0): сереет только доля света звезды над местным фоном, яркость
пикселя та же. Лист: кроп 120″ при 100% до и после, маска; числа — по критериям владельца.

Запуск: uv run python scripts/s305_spikes.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask
from skyforge.compose import LUMA, desaturate_stars
from skyforge.config import Config
from skyforge.finish import finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop
from skyforge.stars import StarMask

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_3b import render  # noqa: E402

CORE_ARCSEC, HALO_ARCSEC, HALF_WIDTH_ARCSEC = 4.0, 30.0, 0.8


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase")
    arrays, _ = load_input(c, P)
    _, _, meta = load_aligned(c, P)
    s = meta["pixel_scale_arcsec"]
    sd = load_nebula_sd(c, P)
    ctx = {"field": stats_field(c, P, list(arrays.values())), "sky": load_sky_mask(c, P), "sd": sd,
           "fin": finish_masks(c, P)}
    img = render(c, P, arrays, ctx)[::-1].copy()           # сетка: строка 0 — низ
    H, W = sd.shape
    star = next(k for k in with_resolved_crops(c, P).report.crops if k.name == "star")
    sl = display_crop((H, W), star.x, star.y, star.size, c.bin)
    f444 = np.nan_to_num(arrays["F444W"])
    win = f444[sl]
    cy, cx = np.unravel_index(np.argmax(ndimage.gaussian_filter(win, 2)), win.shape)
    cy, cx = cy + sl[0].start, cx + sl[1].start
    yy, xx = np.mgrid[0:H, 0:W]
    r = np.hypot(yy - cy, xx - cx) * s
    ang = np.degrees(np.arctan2(yy - cy, xx - cx)) % 360
    ring = (r > 12) & (r < 30)
    bg = ndimage.median_filter(f444, 15)
    ex = np.clip(f444 - bg, 0, None)
    prof = np.bincount(ang[ring].astype(int) % 360, weights=ex[ring], minlength=360)
    prof = ndimage.uniform_filter1d(prof, 3, mode="wrap")
    peaks = []
    for a in np.argsort(-prof):                          # шесть сильнейших направлений, не ближе 30° друг к другу
        if all(min(abs(a - p), 360 - abs(a - p)) >= 30 for p in peaks):
            peaks.append(int(a))
        if len(peaks) == 6:
            break
    peaks.sort()
    mask = r < CORE_ARCSEC
    for a in peaks:
        t = np.radians(a)
        along = (xx - cx) * np.cos(t) + (yy - cy) * np.sin(t)
        perp = np.abs(-(xx - cx) * np.sin(t) + (yy - cy) * np.cos(t)) * s
        mask |= (along > 0) & (perp < HALF_WIDTH_ARCSEC) & (r < HALO_ARCSEC)
    weight = ndimage.gaussian_filter(mask.astype(np.float32), 1.0)
    sm = StarMask(weight, max(3, int(round(6.0 / s))), "F444W")
    after = desaturate_stars(img, sm, 0.0, None)
    diff = np.abs(after - img).max(-1)
    chroma = lambda a: a.max(-1) - a.min(-1)
    spike = mask & (r > CORE_ARCSEC) & (r < HALO_ARCSEC)
    neb = spike & (sd > 0)
    lum_b, lum_a = img @ LUMA, after @ LUMA
    rows = [f"центр звезды (сетка bin {c.bin}): ({cy}, {cx}); направления лучей, °: {peaks}",
            f"вне маски (вес < 0,001): макс |Δ| {diff[weight < 1e-3].max() * 255:.2f} уровня",
            f"лучи (маска, 4–30″): цветность медиана {np.median(chroma(img[spike])) * 255:.1f} → "
            f"{np.median(chroma(after[spike])) * 255:.1f} уровней; яркость макс |Δ| {np.abs(lum_a - lum_b)[spike].max() * 255:.2f}",
            f"туманность на лучах (маска ∩ туманность): {int(neb.sum())} px; цветность медиана "
            f"{np.median(chroma(img[neb])) * 255:.1f} → {np.median(chroma(after[neb])) * 255:.1f} уровней"
            if neb.any() else "туманность на лучах: нет пикселей маски туманности"]
    cut = lambda a: a[sl][::-1]                          # картинка: строка 0 — верх; contact_sheet рисует низом
    mvis = np.repeat(weight[sl][..., None], 3, -1)
    contact_sheet([cut(img)[::-1], cut(after)[::-1], mvis],
                  ["звезда 120″ · 100% · как есть", "звезда 120″ · 100% · лучи обесцвечены (проба)", "маска (вес)"],
                  out / "20_лучи_проба.png", cols=3, tile_inches=6, interpolation="nearest",
                  title="Sh2-305: проба обесцвечивания лучей яркой звезды (одна, без перебора)")
    (out / "20_лучи_проба.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
