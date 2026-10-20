"""3б, приёмка: листы по итоговому PNG Hubble showcase (bin 2).

61 — кропы 100%: center, edge; edge ещё парой «усиление F658N 1,0 / 1,3» (превью, как лист шаг4/32).
62 — небо ×5 (sky, sky2), шов ×3 (gap), углы ×5 с границей заливки (голубая линия), звёзды неба ×4.
63 — карта заполненного одной картинкой: швы (заполнение в стиле) — красным, углы без данных (залиты уровнем неба) — синим.

Запуск: uv run python scripts/acceptance_3b.py <png итога> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned, load_coverage, stats_field, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.cosmics import fill_gaps_linear
from skyforge.finish import finish_masks, nodata_mask
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_3b import render  # noqa: E402

KNOWN = [(250, 824), (316, 834), (381, 838), (483, 119), (910, 884), (962, 766)]   # звёзды неба, сетка bin 2


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/ngc3132hst/showcase.yaml", "showcase")
    raw, _, _ = load_aligned(c, P)
    arrays, _ = load_input(c, P)
    sd = load_nebula_sd(c, P)
    H, W = sd.shape
    rgba = np.asarray(Image.open(png).convert("RGBA")).astype(np.float32) / 255
    img = rgba[..., :3]                                   # строка 0 — верх (картинка); углы — RGB заливки
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}

    def cut(im, name):
        k = crops[name]
        s = k.size // c.bin
        cx, cy = int(round(k.x / c.bin)), int(round(k.y / c.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return im[y0:y0 + s, x0:x0 + s]

    # 61: центр, край; край парой 1,0 / 1,3
    ctx = {"field": stats_field(c, P, list(arrays.values())), "sky": load_sky_mask(c, P), "sd": sd,
           "fin": finish_masks(c, P)}
    g13 = render(c, P, arrays, ctx, {"F658N": 1.3})
    g10 = render(c, P, arrays, ctx, {"F658N": 1.0})
    tiles = [cut(img, "center"), cut(img, "edge"), cut(g10, "edge"), cut(g13, "edge")]
    labels = ["center · итог · 100%", "edge · итог · 100%", "edge · усиление F658N 1,0 (выбрано) · 100%",
              "edge · усиление F658N 1,3 (для записи) · 100%"]
    contact_sheet([t[::-1] for t in tiles], labels, out / "61_кропы_100.png", cols=4, tile_inches=4.2,
                  interpolation="nearest", title="Hubble showcase bin 2: кропы 100% (край — ещё парой 1,0 / 1,3)")

    # 62: небо ×5, шов ×3, углы ×5 с границей заливки, звёзды неба ×4
    cov = load_coverage(c, P)
    nod = nodata_mask(cov, c.clean.fill_gaps_px)[::-1]          # в координатах картинки
    edge = nod & ~ndimage.binary_erosion(nod, iterations=1, border_value=1)
    tiles, labels = [], []
    for n in ("sky", "sky2"):
        tiles.append(np.clip(cut(img, n) * 5, 0, 1)); labels.append(f"{n} · ×5")
    g = cut(img, "gap")
    tiles.append(np.kron(g[:g.shape[0] // 3, :g.shape[1] // 3] if g.shape[0] > 120 else g, np.ones((3, 3, 1))))
    labels.append("gap (шов) · ×3 увеличение")
    lab, nlab = ndimage.label(nod)
    sizes = np.bincount(lab.ravel())[1:]
    for i in np.argsort(-sizes)[:2]:                       # два угла: точка границы ближе всего к центру кадра
        ys, xs = np.nonzero(edge & (lab == i + 1))
        j = int(np.argmin((ys - H / 2) ** 2 + (xs - W / 2) ** 2))
        y, x = int(np.clip(ys[j], 50, H - 51)), int(np.clip(xs[j], 50, W - 51))
        t = np.clip(img[y - 50:y + 50, x - 50:x + 50] * 5, 0, 1).copy()
        t[edge[y - 50:y + 50, x - 50:x + 50]] = [0.2, 0.7, 1.0]
        tiles.append(t); labels.append(f"угол ({H - 1 - y},{x}) · ×5 · голубое — граница заливки")
    for y, x in KNOWN:
        r = H - 1 - y                                     # сетка → картинка
        tiles.append(np.kron(img[r - 10:r + 11, x - 10:x + 11], np.ones((4, 4, 1)))); labels.append(f"звезда ({y},{x}) · ×4")
    contact_sheet([t[::-1] for t in tiles], labels, out / "62_небо_шов_углы_звёзды.png", cols=4, tile_inches=3.6,
                  interpolation="nearest", title="Hubble showcase bin 2: небо ×5, шов ×3, углы ×5, звёзды неба ×4")

    # 63: карта заполненного
    filled = np.logical_or.reduce([fill_gaps_linear(a, c.clean.fill_gaps_px)[1] for a in arrays.values()])[::-1]
    base = shrink(img.mean(-1), 1100)
    ov = np.stack([base] * 3, -1)
    fm, nm = shrink(filled.astype(np.float32), 1100) > 0, shrink(nod.astype(np.float32), 1100) > 0.5
    ov[fm] = [1.0, 0.15, 0.15]
    ov[nm] = [0.2, 0.4, 1.0]
    contact_sheet([ov[::-1]], [f"швы заполнены: {int(filled.sum())} px (красные); углы залиты уровнем неба: {int(nod.sum())} px (синие)"],
                  out / "63_карта_заполненного.png", cols=1, tile_inches=9, interpolation="nearest",
                  title="Заполнено в showcase: швы чипов (линейно поперёк, ≤ 6 px) и углы без данных (ровный уровень неба 6)")
    print(f"швов заполнено {int(filled.sum())} px, углов залито {int(nod.sum())} px; альфа 0 в PNG: {int((rgba[..., 3] == 0).sum())} px; "
          f"RGB углов (уровни): {np.unique((img[nod] * 255).round(), axis=0)[:3].tolist()}")


if __name__ == "__main__":
    main()
