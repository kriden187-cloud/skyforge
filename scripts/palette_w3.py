"""Фаза 6, W3-2: сетка палитра × белая точка — лист 40, bin 4.

Палитры (порядок по длине волны в каждой — оттенок убывает от F182M к F444W):
  А — текущая (configs/w3_2/showcase.yaml);
  Б — семь оттенков с равным шагом по кругу: 240, 200, 160, 120, 80, 40, 0° (насыщенность 0,85, яркость 1 — как
      в среднем у палитры А);
  В — как Б, плюс color.weights 0,7 на F405N и F444W.
Белая точка 99,8 (текущая) и 99,9. Колонки — 6 вариантов (А · 99,8 — текущий итог); строки — целиком, lanes,
cloud, globule, sky2 при 100%.
Меры в подписи: цветность (max − min канала, медиана по туманности без звёзд; при stretch.exclude_point_sources —
и без точечных источников всех фильтров), разброс оттенка (круговое σ оттенка
пикселей с цветностью > 20), нули опорного неба R/G/B, выжжено в туманности, зелень звёзд вне туманности p50.
Перед сеткой — сверка: отрисовка текущего конфига против итогового PNG конвейера (максимум разности, уровни).

Лист 41 (третий аргумент «41», гипотеза сверх задания): яркость слоёв выровнена белой точкой на фильтр — короткие
F182M, F210M на 99,0 и F300M на 99,5 (при 99,8 их медиана в туманности 0,32–0,41 против 0,60–0,68 у длинных:
белую точку коротких ставят крылья звёзд). Колонки: А текущий, А + выравнивание, Б + выравнивание.

Запуск: uv run python scripts/palette_w3.py <папка> <png итога конвейера (bin 4)> [41]
"""

from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import with_resolved_crops
from skyforge.clean import clean_dir
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, shrink
from skyforge.stretch import reference_sky

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from palette_s305 import hue_deg, workspace  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

ORDER = ("F182M", "F210M", "F300M", "F335M", "F360M", "F405N", "F444W")
ROWS = (("целиком", None), ("lanes 100%", "lanes"), ("cloud 100%", "cloud"), ("globule 100%", "globule"),
        ("sky2 100%", "sky2"))


def hsv_hex(h: float, s: float = 0.85, v: float = 1.0) -> str:
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return "#" + "".join(f"{round(x * 255):02x}" for x in (r, g, b))


def main() -> None:
    out, pipeline_png = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    sheet = sys.argv[3] if len(sys.argv) > 3 else "40"
    (out / f"варианты_{sheet}").mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/w3_2/showcase.yaml", "showcase")
    if sheet == "43":   # фаза 6: насыщенность × белая точка на выровненных слоях (exclude_point_sources), палитра А
        variants = [(f"насыщенность {sat:g} · белая {white:g}" + (" (конфиг)" if sat == base.color.saturation and
                                                                  white == base.stretch.white else ""),
                     f"color.saturation={sat}" + (f";stretch.white={white}" if white != base.stretch.white else ""))
                    for white in (99.8, 99.9) for sat in (1.4, 2.0, 3.0)]
    else:
        pal_a = {f: base.color.overrides[f] for f in ORDER}
        pal_b = {f: hsv_hex(h) for f, h in zip(ORDER, (240, 200, 160, 120, 80, 40, 0))}
        for name, pal in (("А", pal_a), ("Б", pal_b)):
            hs = [hue_deg(pal[f]) for f in ORDER]
            print(f"палитра {name}: " + ", ".join(f"{f} {pal[f]} {h:.0f}°" for f, h in zip(ORDER, hs))
                  + (" — порядок сохранён" if all(a > b for a, b in zip(hs, hs[1:])) else " — ПОРЯДОК НАРУШЕН"), flush=True)
        spec_b = ";".join(f"color.overrides.{f}={c}" for f, c in pal_b.items())
        spec_c = spec_b + ";color.weights.F405N=0.7;color.weights.F444W=0.7"
        variants = []
        for pname, spec in (("А", ""), ("Б", spec_b), ("В", spec_c)):
            for white in (99.8, 99.9):
                s = spec + (f";stretch.white={white}" if white != base.stretch.white else "")
                variants.append((f"{pname} · белая {white:g}" + (" (текущий)" if pname == "А" and white == 99.8 else ""), s))
        balance = "stretch.per_filter.F182M.white=99.0;stretch.per_filter.F210M.white=99.0;stretch.per_filter.F300M.white=99.5"
        if sheet == "41":
            variants = [("А · белая 99.8 (текущий)", ""), ("А + выравнивание слоёв", balance),
                        ("Б + выравнивание слоёв", spec_b + ";" + balance)]

    w = workspace(base, P)
    sd, sky = w["ctx"]["sd"], w["ctx"]["sky"]
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in w["arrays"].values()])
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    stars = ndimage.binary_dilation(sw > 0.01, iterations=1)
    neb = (sd > 0) & ~stars & full
    if base.stretch.exclude_point_sources is not None:   # фаза 6: без звёзд, которые маска clean пропускает
        from skyforge.align import read_aligned_meta
        from skyforge.stretch import point_sources

        scale = read_aligned_meta(base, P)["pixel_scale_arcsec"]
        for a in w["arrays"].values():
            neb &= ~point_sources(a, base.stretch.exclude_point_sources, scale)[0]
    neb = neb[::-1]
    star_out = ((sw > 0.5) & (sd < 0))[::-1]
    ref, _ = reference_sky(base, P, sd.shape)
    ref = (ref & full)[::-1]
    crops = {k.name: k for k in with_resolved_crops(base, P).report.crops}

    def cut(img: np.ndarray, name: str | None) -> np.ndarray:
        if name is None:
            return shrink(img, 900)
        k = crops[name]
        g = display_crop((H, W), k.x, k.y, k.size, base.bin)
        return img[H - g[0].stop:H - g[0].start, g[1]]

    def measures(img: np.ndarray) -> dict:
        p = img[neb]
        chroma = (p.max(-1) - p.min(-1)) * 255
        sel = p[chroma > 20]
        ang = np.deg2rad(np.array([colorsys.rgb_to_hsv(*v)[0] * 360 for v in sel[::max(1, len(sel) // 20000)]]))
        R = np.hypot(np.sin(ang).mean(), np.cos(ang).mean()) if ang.size else 1.0
        sk, st = img[ref] * 255, img[star_out] * 255
        return {"chroma": float(np.median(chroma)), "hue_sd": float(np.rad2deg(np.sqrt(-2 * np.log(max(R, 1e-9))))),
                "zeros": [float((sk[:, i] <= 0.5).mean() * 100) for i in range(3)],
                "burned": float((p.max(-1) >= 254 / 255).mean() * 100),
                "green": float(np.percentile(st[:, 1] - np.maximum(st[:, 0], st[:, 2]), 50)) if st.size else float("nan")}

    cols = []
    for lab, spec in variants:
        img = render(apply(base, spec), P, w["arrays"], w["ctx"])
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / f"варианты_{sheet}" / f"{lab.split(' (')[0]}.png")
        if spec == "" or "(конфиг)" in lab:
            ref_png = np.asarray(Image.open(pipeline_png).convert("RGB"), dtype=np.float32)
            d = np.abs(ref_png - np.round(img * 255)).max()
            print(f"сверка: отрисовка текущего конфига против итога конвейера — максимум разности {d:.0f} уровней",
                  flush=True)
        cols.append((lab, img, measures(img)))
        print(f"  готово: {lab}", flush=True)

    tiles, labels = [], []
    for rl, cn in ROWS:
        for lab, img, m in cols:
            tiles.append(cut(img, cn))
            if cn is None:
                labels.append(f"{lab}\nцветность {m['chroma']:.0f} · разброс оттенка {m['hue_sd']:.0f}°\n"
                              f"нули неба {' / '.join(f'{z:.2f}' for z in m['zeros'])}% · выжжено {m['burned']:.2f}% · "
                              f"зелень звёзд {m['green']:+.0f}")
            else:
                labels.append(f"{lab} · {rl}")
    title = ("W3-2 bin 4: палитры А (текущая), Б (по кругу 240…0°), В (Б + вес 0,7 F405N, F444W) × белая 99,8 / 99,9"
             if sheet == "40" else "W3-2 bin 4, гипотеза: выравнивание яркости слоёв белой точкой на фильтр "
                                   "(F182M, F210M 99,0; F300M 99,5)")
    name = {"40": "40_палитра_и_белая_точка.png", "41": "41_выравнивание_слоёв.png",
            "43": "43_насыщенность_и_белая_точка.png"}[sheet]
    if sheet == "43":
        title = ("W3-2 bin 4, 8 фильтров, белая точка без точечных источников: насыщенность 1,4 / 2,0 / 3,0 × "
                 "белая 99,8 / 99,9")
    contact_sheet(tiles, labels, out / name, cols=len(cols) if sheet != "43" else 3,
                  tile_inches=3.6 if sheet == "40" else 4.6,
                  interpolation="nearest", title=title)
    rows = ["вариант\tцветность\tразброс оттенка, °\tнули опорного неба R/G/B, %\tвыжжено, %\tзелень звёзд p50"]
    rows += [f"{lab}\t{m['chroma']:.0f}\t{m['hue_sd']:.1f}\t{' / '.join(f'{z:.2f}' for z in m['zeros'])}\t"
             f"{m['burned']:.2f}\t{m['green']:+.0f}" for lab, _, m in cols]
    if sheet != "43":
        rows.append("палитра А: " + ", ".join(f"{f} {pal_a[f]}" for f in ORDER))
        rows.append("палитра Б и В: " + ", ".join(f"{f} {pal_b[f]}" for f in ORDER))
    (out / f"{sheet}_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
