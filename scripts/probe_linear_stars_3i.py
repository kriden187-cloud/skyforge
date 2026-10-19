"""Проба 3i: звёзды с насыщенным ядром отдельно от туманности до растяжки. Конвейер не меняется."""
import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.transform import SimilarityTransform, warp

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask, load_star_weight
from skyforge.compose import base_color, tail_color
from skyforge.config import Config
from skyforge.finish import finish_masks, finish_region
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop
from skyforge.stars import core_area_px, find_core_holes
from skyforge.stretch import curve, load_masks, stretch_array

OUT = pathlib.Path(sys.argv[1])
# параметры пробы (один вариант)
MIN_CORE_PX, MIN_FILTERS = 50, 2          # звезда с насыщенным ядром: дыра ≥ 50 px в ≥ 2 фильтрах
R_IN, R_OUT = 100, 150                    # слой звезды до R_IN целиком, к R_OUT плавно в ноль, px
PROFILE_Q = 30                            # процентиль по кольцу: ниже медианы — туманность и лучи не завышают звезду
STAR_A = 0.01                             # asinh слоя звезды; белая точка — ядро (максимум слоя)

P = Paths(pathlib.Path(".").resolve())
cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
aligned, _, meta = load_aligned(cfg, P)
arrays, _ = load_input(cfg, P)
H, W = next(iter(arrays.values())).shape
scale = meta["pixel_scale_arcsec"]

# звёзды с насыщенным ядром
per = {f: find_core_holes(a, core_area_px(cfg.stars.max_core_arcsec, scale)) for f, a in aligned.items()}
union = np.logical_or.reduce(list(per.values()))
lab, n = ndimage.label(union, structure=np.ones((3, 3)))
stars = []
for i in range(1, n + 1):
    comp = lab == i
    nf = sum(per[f][comp].any() for f in per)
    if comp.sum() >= MIN_CORE_PX and nf >= MIN_FILTERS:
        cy, cx = ndimage.center_of_mass(comp)
        stars.append((cy, cx, int(comp.sum()), nf))
print("звёзды с насыщенным ядром (картинка x, y; площадь; фильтров):",
      [(round(cx), round(H - 1 - cy), a, k) for cy, cx, a, k in stars])

# слой звёзд: радиальный профиль над уровнем туманности на внешнем кольце
star_layer = {f: np.zeros((H, W), np.float32) for f in arrays}
yy, xx = np.indices((2 * R_OUT + 1, 2 * R_OUT + 1))
for cy, cx, _, _ in stars:
    y0, x0 = int(round(cy)) - R_OUT, int(round(cx)) - R_OUT
    sl = (slice(max(y0, 0), min(y0 + 2 * R_OUT + 1, H)), slice(max(x0, 0), min(x0 + 2 * R_OUT + 1, W)))
    r = np.hypot(yy - (cy - y0), xx - (cx - x0))[sl[0].start - y0:sl[0].stop - y0, sl[1].start - x0:sl[1].stop - x0]
    ri = np.floor(r).astype(int)
    taper = np.where(r < R_IN, 1.0, np.clip(0.5 + 0.5 * np.cos(np.pi * (r - R_IN) / (R_OUT - R_IN)), 0, 1))
    taper[r >= R_OUT] = 0
    for f, a in arrays.items():
        sub = a[sl]
        ok = np.isfinite(sub)
        level = np.percentile(sub[ok & (r >= R_OUT - 20) & (r < R_OUT)], 50)
        prof = np.full(R_OUT + 1, np.nan)
        for k in range(R_OUT + 1):
            v = sub[ok & (ri == k)]
            if v.size:
                prof[k] = np.percentile(v, PROFILE_Q)
        prof = ndimage.median_filter(np.nan_to_num(prof, nan=level), size=3, mode="nearest")
        s = np.clip(prof[np.minimum(ri, R_OUT)] - level, 0, None) * taper
        star_layer[f][sl] += s.astype(np.float32)

# растяжка: туманность — прежние точки (статистика по исходным кадрам), звезда — своя белая точка
masks = load_masks(cfg, P)
field = stats_field(cfg, P, list(arrays.values()))
lay_3h, lay_probe = {}, {}
for f, a in arrays.items():
    s = cfg.stretch.for_filter(f)
    lay_3h[f], st = stretch_array(a, s, masks=masks, field=field)
    neb, _ = stretch_array(a - star_layer[f], s, stats_from=a, masks=masks, field=field)
    sw = float(star_layer[f].max()) or 1.0
    st_d = curve(np.clip(star_layer[f] / sw, 0, 1), "asinh", STAR_A).astype(np.float32)
    lay_probe[f] = (1 - (1 - np.nan_to_num(neb)) * (1 - st_d)).astype(np.float32)
    lay_probe[f][~np.isfinite(a)] = np.nan
    print(f"  {f}: белая точка туманности {st['white_value']:.4g}, ядро звезды {sw:.4g} (×{sw / st['white_value']:.0f})")

sd = load_nebula_sd(cfg, P)
sky = load_sky_mask(cfg, P)
sm = load_star_weight(cfg, P)
crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
comp = {}
for name, L in (("3h", lay_3h), ("probe", lay_probe)):
    b, _ = base_color(cfg, L, sky, sd, field=field)
    comp[name] = tail_color(cfg, b, L, sm, sd)
pipe = np.load(P.work(cfg.slug, 1) / "style-showcase" / "composite.npy")
print("проверка: восстановленная композиция = конвейер:", np.allclose(comp["3h"], pipe, atol=1e-6))
c = crops["core"]
slc = display_crop(comp["3h"].shape, c.x, c.y, c.size, 1)
for name in comp:
    x = comp[name][slc]
    print(f"core, {name}: доля пикселей с каналом ≥ 0,99 до finish: {(x >= 0.99).any(-1).mean() * 100:.1f}%")

# итог с финишем showcase (звёзды поля 0.7/1, контраст 0.6)
sw_, sd_, origin = finish_masks(cfg, P)
fin = {}
for name in comp:
    fin[name] = {}
    for cn in ("core", "inner"):
        c = crops[cn]
        sl = display_crop(comp[name].shape, c.x, c.y, c.size, 1)
        fin[name][cn] = finish_region(comp[name], cfg.finish, sw_, sd_, origin, sl)
        if name == "probe":
            x = fin[name][cn]
            print(f"{cn}, проба после finish: канал ≥ 0,99 у {(x >= 0.99).any(-1).mean() * 100:.1f}% пикселей")
base3h = np.asarray(Image.open("data/baselines/ngc3132/phase-3h/showcase_bin1.png").convert("RGB")).astype(np.float32)[::-1] / 255

# NASA на нашей сетке: подобие по центральной звезде и звезде внизу справа
ref = np.asarray(Image.open("data/reference/ngc3132_nasa.jpg").convert("RGB")).astype(np.float32) / 255
def peak(img, x, y, r):
    g = ndimage.gaussian_filter(img.mean(-1), 2)
    y0, x0 = max(0, y - r), max(0, x - r)
    win = g[y0:y + r, x0:x + r]
    j, i = np.unravel_index(np.argmax(win), win.shape)
    return x0 + i, y0 + j
ours = np.asarray(Image.open(sorted(P.out_root.glob("ngc3132/*_styles"))[-1] / "showcase" / "ngc3132_showcase.png").convert("RGB")).astype(np.float32) / 255
o1, o2 = peak(ours, 1133, 876, 30), peak(ours, 1901, 1701, 30)
n1, n2 = peak(ref, 1491, 1313, 80), peak(ref, 2500, 2388, 80)
t = SimilarityTransform()
t.estimate(np.array([o1, o2], float), np.array([n1, n2], float))
nasa = warp(ref, t, output_shape=ours.shape[:2], order=1)[::-1]    # в ориентации сетки (строка 0 — низ)
print("NASA: масштаб %.3f, поворот %.2f°" % (t.scale, np.degrees(t.rotation)))

imgs, labels = [], []
for cn in ("core", "inner"):
    c = crops[cn]
    sl = display_crop(comp["3h"].shape, c.x, c.y, c.size, 1)
    for k, gain in ((1, "100%"), (5, "×5")):
        for name, im in (("NASA", nasa[sl]), ("phase-3h", base3h[sl]), ("проба", fin["probe"][cn])):
            imgs.append(np.clip(im * k, 0, 1))
            labels.append(f"{cn} · {name} · {gain}")
contact_sheet(imgs, labels, OUT / "09_проба_звёзды_до_растяжки.png", cols=3, tile_inches=4.2, interpolation="nearest",
              title="Проба: звёзды с насыщенным ядром отдельно от туманности до растяжки (core, inner)")
json.dump({"stars": [(int(round(cx)), int(round(H - 1 - cy)), int(a), int(k)) for cy, cx, a, k in stars]},
          open(OUT / "проба_звёзды.json", "w", encoding="utf-8"), ensure_ascii=False)

# крупно: стык звезды и туманности у центральной звезды, ×3 ближайшим соседом, и слои пробы по фильтрам
c = crops["core"]
sl = display_crop(comp["3h"].shape, c.x, c.y, c.size, 1)
cy0, cx0 = (sl[0].stop - sl[0].start) // 2, (sl[1].stop - sl[1].start) // 2
z = (slice(cy0 - 30, cy0 + 30), slice(cx0 - 30, cx0 + 30))
big = [np.kron(np.clip(fin[n]["core"][z], 0, 1), np.ones((3, 3, 1))) for n in ("3h", "probe")]
lay = [np.kron(np.clip(lay_probe[f][sl][z], 0, 1), np.ones((3, 3))) for f in lay_probe]
contact_sheet(big + lay, ["phase-3h (финиш 3i), 60 px ×3", "проба, 60 px ×3"] + [f"проба, слой {f}" for f in lay_probe],
              OUT / "10_проба_стык_у_ядра.png", cols=4, tile_inches=3.2, interpolation="nearest",
              title="Стык звезды и туманности у ядра центральной звезды: разноцветный шум — фильтры расходятся в слое звезды")
