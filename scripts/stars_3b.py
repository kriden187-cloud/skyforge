"""3б, шаг 3: звёзды поля на небе — перебор color.star_saturation (область sky). Кропы звёзд неба ×4, превышение
ядра над фоном R/G/B, разница с вариантом 1 на туманности (должна быть 0).

Запуск: uv run python scripts/stars_3b.py <папка> <имя> <значение> [<значение> ...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned, stats_field
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.finish import finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_3b import render  # noqa: E402

KNOWN = [(152, 411), (250, 824), (316, 834), (381, 838), (483, 119), (895, 728), (910, 884), (962, 766)]


def main() -> None:
    out, name, values = pathlib.Path(sys.argv[1]), sys.argv[2], [float(v) for v in sys.argv[3:]]
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/ngc3132hst/showcase.yaml", "showcase")
    arrays, _ = load_input(base, P)
    load_aligned(base, P)
    ctx = {"field": stats_field(base, P, list(arrays.values())), "sky": load_sky_mask(base, P),
           "sd": load_nebula_sd(base, P), "fin": finish_masks(base, P)}
    sd = ctx["sd"]
    w = np.load(clean_dir(base, P) / "star_weight.npy")
    lab, n = ndimage.label((w > 0.5) & (sd < 0), structure=np.ones((3, 3)))
    pos = ndimage.maximum_position(np.nan_to_num(arrays["F658N"]), lab, np.arange(1, n + 1))
    H, W = sd.shape
    pos = [p for p in pos if 10 <= p[0] < H - 10 and 10 <= p[1] < W - 10]
    rows = [f"звёзд неба в маске (вес > 0,5): {n}; известных 8 из них в маске: "
            f"{sum(bool(w[y - 1:y + 2, x - 1:x + 2].max() > 0.5) for y, x in KNOWN)}"]
    imgs = {}
    for v in values:
        c = base.with_param("color.star_saturation", v)
        imgs[v] = render(c, P, arrays, ctx)[::-1]                 # строка 0 — низ, как сетка
        print(f"  готово: {v}", flush=True)
    ref = imgs[1.0] if 1.0 in imgs else imgs[values[0]]
    neb = (sd >= 0) | ((sd < 0) & (w <= 0.01))
    tiles, labels = [], []
    for y, x in pos:
        for v in values:
            im = imgs[v]
            t = im[y - 10:y + 11, x - 10:x + 11]
            core = t[9:12, 9:12].reshape(-1, 3).mean(0) * 255
            bg = np.median(t.reshape(-1, 3), 0) * 255
            d = core - bg
            tiles.append(np.kron(t, np.ones((4, 4, 1))))
            labels.append(f"({y},{x}) s={v:g}: {d[0]:.0f}/{d[1]:.0f}/{d[2]:.0f}")
    for v in values:
        d = np.abs(imgs[v] - ref) * 255
        rows.append(f"star_saturation {v:g}: вне звёзд неба |Δ| к {1.0 if 1.0 in imgs else values[0]:g} — макс {d[neb].max():.0f} уровня")
        ds = []
        for y, x in pos:
            t = imgs[v][y - 1:y + 2, x - 1:x + 2].reshape(-1, 3).mean(0) - np.median(imgs[v][y - 10:y + 11, x - 10:x + 11].reshape(-1, 3), 0)
            ds.append(t * 255)
        ds = np.array(ds)
        g = (ds[:, 1] > 1.15 * np.maximum(ds[:, 0], ds[:, 2])).sum()
        rows.append(f"    ядро над фоном, медиана R/G/B {' / '.join(f'{q:.0f}' for q in np.median(ds, 0))}; зелёных (G > 1,15·max(R, B)) {g} из {len(ds)}")
    # тайлы уже в порядке сетки (строка 0 — низ), как ждёт contact_sheet
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(values) * 2, tile_inches=2.2,
                  interpolation="nearest",
                  title="Звёзды неба (маска clean, по F658N) при color.star_saturation (область sky): 21 px ×4; "
                        "подпись — ядро над фоном R/G/B")
    (out / f"{name}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
