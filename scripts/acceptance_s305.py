"""Фаза 4, Sh2-305: листы приёмки по итоговому PNG showcase (bin 1).

30 — целиком при обычной яркости и ×8.
31 — восемь кропов из конфига при 100% (звезда 120″ — среди них).
32 — sky, sky2, стык ×5 и рядом карта областей CON F444W.
33 — таблица чисел по критериям: небо R/G/B и нули, выжжено в туманности, зелень звёзд неба.

Запуск: uv run python scripts/acceptance_s305.py <png итога> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import aligned_dir, load_aligned, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, shrink


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(c, P)
    load_aligned(c, P)
    sd, sky = load_nebula_sd(c, P), load_sky_mask(c, P)
    H, W = sd.shape
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    img = rgba[..., :3].astype(np.float32)[::-1] / 255          # сетка: строка 0 — низ
    assert img.shape[:2] == (H, W), (img.shape, (H, W))
    crops = with_resolved_crops(c, P).report.crops

    contact_sheet([shrink(img, 1200), np.clip(shrink(img, 1200) * 8, 0, 1)], ["целиком", "целиком ×8"],
                  out / "30_целиком.png", cols=2, tile_inches=8, interpolation="lanczos",
                  title="Sh2-305 showcase bin 1: целиком и ×8")
    tiles, labels = [], []
    for k in crops:
        sl = display_crop((H, W), k.x, k.y, k.size, 1)
        tiles.append(img[sl]); labels.append(f"{k.name} · 100% · {k.size * 0.0629:.0f}″")
    contact_sheet(tiles, labels, out / "31_кропы_100.png", cols=4, tile_inches=4.5, interpolation="nearest",
                  title="Sh2-305 showcase bin 1: восемь кропов при 100%")
    tl = np.load(aligned_dir(c, P) / "F444W.tiles.npy")
    pal = np.random.default_rng(0).uniform(0.15, 1, (int(np.nanmax(tl)) + 2, 3)).astype(np.float32)
    tiles, labels = [], []
    for name in ("sky", "sky2", "seam"):
        k = next(x for x in crops if x.name == name)
        sl = display_crop((H, W), k.x, k.y, k.size, 1)
        tiles.append(np.clip(img[sl] * 5, 0, 1)); labels.append(f"{name} · ×5")
        t = tl[sl]
        m = pal[np.where(t >= 0, t, -1).astype(int)]
        m[t < 0] = 0
        tiles.append(m); labels.append(f"{name} · области CON F444W")
    contact_sheet(tiles, labels, out / "32_небо_стык_CON.png", cols=2, tile_inches=4.5, interpolation="nearest",
                  title="Sh2-305 showcase bin 1: небо и стык плиток ×5 с картой CON F444W")
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    S = sky & full
    sk = img[S] * 255
    sw = np.load(clean_dir(c, P) / "star_weight.npy")
    st = img[(sw > 0.5) & (sd < 0)] * 255
    gx = st[:, 1] - np.maximum(st[:, 0], st[:, 2])
    green = [float(np.percentile(gx, q)) for q in (50, 75, 95)]   # шаг 7: медиана скрывает хвост зелёных звёзд
    burned = float((img[sd > 0].max(-1) >= 254 / 255).mean() * 100)
    rows = ["критерий\tзначение\tпорог",
            f"небо R / G / B (медиана, уровни)\t{' / '.join(f'{np.median(sk[:, k]):.0f}' for k in range(3))}"
            f"\t{c.stretch.sky_level:g} (stretch.sky_level)",
            f"нулей на небе R / G / B, %\t{' / '.join(f'{(sk[:, k] <= 0).mean() * 100:.2f}' for k in range(3))}\t< 0,5",
            f"выжжено в туманности (канал ≥ 254), %\t{burned:.2f}\t≤ 0,5",
            f"зелень звёзд неба: G − max(R, B), p50 / p75 / p95, уровни\t{' / '.join(f'{g:+.0f}' for g in green)}"
            f"\tp50 ≤ 0",
            f"прозрачных пикселей (нет данных)\t{int((rgba[..., 3] == 0).sum())}\t—",
            "прямые границы на стыке при ×5\tсм. лист 32\tнет"]
    (out / "33_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
