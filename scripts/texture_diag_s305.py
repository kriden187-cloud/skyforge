"""Фаза 4, шаг 5: дополнительные меры «желе» по сохранённым вариантам scripts/texture_s305.py (без перерисовки).

В кропе filaments по маске туманности (и по всей маске, где сказано):
- G / max(R, B) — медиана (заданная мера «чистый зелёный» берёт порог 1,5);
- доли G > 1,2·max(R, B) и G > max(R, B) (зелёный — главный канал), % всей маски;
- мелкая фактура — σ яркости после вычитания размытия (гаусс 2 px при bin 4 = 8 px при bin 1), уровни;
- крупный перепад — σ размытой яркости, уровни;
- мелкая фактура, отнесённая к яркости — σ остатка после размытия / средняя яркость, по маске кропа и по яркой
  половине (яркость выше медианы маски кропа) — без роста от общего контраста;
- разброс оттенка — σ оттенка (градусы) по пикселям с цветностью > 20 уровней; медиана оттенка; медиана цветности.

Запуск: uv run python scripts/texture_diag_s305.py <папка шага 5>
"""

from __future__ import annotations

import colorsys
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import with_resolved_crops
from skyforge.clean import load_nebula_sd
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from faint_s305 import thevenot_on_grid  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    sd = load_nebula_sd(base, P)
    H, W = sd.shape
    neb = (sd > 0)[::-1]
    c = {k.name: k for k in with_resolved_crops(base, P).report.crops}["filaments"]
    s = c.size // base.bin
    cx, cy = int(round(c.x / base.bin)), int(round(c.y / base.bin))
    y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
    sl = (slice(y0, y0 + s), slice(x0, x0 + s))
    files = sorted((root / "варианты").glob("*.png"))
    ref = np.asarray(Image.open(next(f for f in files if f.name.startswith("color.shadows.value=4_color.saturation=1.2_"
                                                                          "finish.local_contrast.amount=0.3"))),
                     dtype=np.float32) / 255
    thev, _ = thevenot_on_grid(np.asarray(Image.fromarray((ref * 255).astype(np.uint8)).resize((W * base.bin, H * base.bin)),
                                          dtype=np.float32) / 255 @ LUMA, base.bin)
    items = [("Тевено", thev[:H, :W])] + [(f.stem, np.asarray(Image.open(f), dtype=np.float32) / 255) for f in files]
    rows = ["вариант\tG/max(R,B) медиана в filaments\tG>1,2·max(R,B), % маски\tG>max(R,B), % маски"
            "\tмелкая фактура σ, уровни\tкрупный перепад σ, уровни\tфактура / яркость\tфактура / яркость в яркой половине"
            "\tразброс оттенка σ, °\tоттенок, °\tцветность, уровни"]
    for lab, img in items:
        p = img[neb]
        g_rb = p[:, 1] / np.maximum(np.maximum(p[:, 0], p[:, 2]), 1e-3)
        t = img[sl]
        m = neb[sl]
        lum = t @ LUMA * 255
        blur = ndimage.gaussian_filter(lum, 2)
        q = t[m]
        br = m & (lum > np.median(lum[m]))
        mx, mn = q.max(-1), q.min(-1)
        chroma = (mx - mn) * 255
        ok = chroma > 20
        hue = np.array([colorsys.rgb_to_hsv(*v)[0] * 360 for v in q[ok][::7]])
        ang = np.deg2rad(hue)
        mean = np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        hs = np.rad2deg(np.sqrt(-2 * np.log(np.hypot(np.sin(ang).mean(), np.cos(ang).mean()))))
        q_rb = q[:, 1] / np.maximum(np.maximum(q[:, 0], q[:, 2]), 1e-3)
        rows.append(f"{lab}\t{np.median(q_rb):.2f}\t{(g_rb > 1.2).mean() * 100:.1f}\t{(g_rb > 1).mean() * 100:.1f}"
                    f"\t{(lum - blur)[m].std():.1f}\t{blur[m].std():.1f}\t{(lum - blur)[m].std() / lum[m].mean():.3f}"
                    f"\t{(lum - blur)[br].std() / lum[br].mean():.3f}\t{hs:.1f}\t{np.rad2deg(mean) % 360:.0f}"
                    f"\t{np.median(chroma):.0f}")
    (root / "62_доп_меры.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
