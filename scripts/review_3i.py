"""3i: листы приёмки — ядро ×3, inner/core рядом с NASA, звёзды с насыщенным ядром, доли каналов по шагам,
три карты разности (выборка, белая точка и режим теней, слой звезды).

Запуск: uv run python scripts/review_3i.py <папка> <phase-3h png> <A: только выборка> <B: + режим и white> <C: + слой звезды>
(A, B, C — итоговые PNG showcase соответствующих прогонов).
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import read_aligned_meta, stats_field, with_resolved_crops
from skyforge.clean import load_nebula_sd, load_saturated_stars, load_sky_mask
from skyforge.compose import apply_shadows, base_color, load_stretched, saturate
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402


def load(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255     # картинка: строка 0 — верх


def crop(img: np.ndarray, x: int, y: int, s: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(np.clip(x - s // 2, 0, w - s)), int(np.clip(y - s // 2, 0, h - s))
    return img[y0:y0 + s, x0:x0 + s]


def zoom(a: np.ndarray, k: int = 3) -> np.ndarray:
    return np.kron(a, np.ones((k, k, 1)))


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    p3h, pa, pb, pc = sys.argv[2:6]
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    imgs = {"phase-3h": load(p3h), "A: выборка без звёзд": load(pa), "B: + тени по max, white 99,98": load(pb),
            "C: + слой звезды": load(pc)}
    H = imgs["C: + слой звезды"].shape[0]
    nasa = nasa_on_grid(imgs["C: + слой звезды"], str(P.abs(cfg.reference_image)), [(1133, 876), (1901, 1701)],
                        [(1491, 1313), (2500, 2388)])[::-1]                       # в ориентации картинки
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    stars = {(r, c) for r, c, _, _ in load_saturated_stars(cfg, P)}
    pos = {"центральная": None, "внизу справа": None, "слева": None}
    for r, c in stars:
        y = H - 1 - r
        key = "центральная" if abs(c - 1133) < 30 else ("внизу справа" if c > 1500 else "слева")
        pos[key] = (c, y)

    # 1. ядро центральной звезды 60 px ×3 и кропы inner, core 100%
    seq = [("NASA", nasa)] + [(k, v) for k, v in imgs.items() if k in ("phase-3h", "B: + тени по max, white 99,98",
                                                                         "C: + слой звезды")]
    cx, cy = pos["центральная"]
    tiles, labels = [], []
    for name, im in seq:
        tiles.append(zoom(crop(im, cx, cy, 60)))
        labels.append(f"ядро · {name} · 60 px ×3")
    for cn in ("core", "inner"):
        cr = crops[cn]
        for name, im in seq:
            tiles.append(crop(im, cr.x, cr.y, cr.size))
            labels.append(f"{cn} · {name} · 100%")
    contact_sheet([t[::-1] for t in tiles], labels, out / "18_ядро_core_inner.png", cols=4, tile_inches=3.8,
                  interpolation="nearest", title="Центральная звезда: NASA, phase-3h, без слоя звезды, со слоем")

    # 2. звезда слева (кроп star_left) и звезда внизу справа: 100%, ×3 у ядра, ×5 по яркости
    tiles, labels = [], []
    for key, (x, y) in (("слева", pos["слева"]), ("внизу справа", pos["внизу справа"])):
        for name, im in (("phase-3h", imgs["phase-3h"]), ("без слоя", imgs["B: + тени по max, white 99,98"]),
                         ("со слоем", imgs["C: + слой звезды"])):
            tiles += [crop(im, x, y, 300), zoom(crop(im, x, y, 80)), np.clip(crop(im, x, y, 300) * 5, 0, 1)]
            labels += [f"{key} · {name} · 300 px", f"{key} · {name} · 80 px ×3", f"{key} · {name} · ×5"]
    contact_sheet([t[::-1] for t in tiles], labels, out / "19_звёзды_слева_и_справа.png", cols=3, tile_inches=3.6,
                  interpolation="nearest", title="Звёзды с насыщенным ядром: тёмных колец и ступенек быть не должно")

    # 2б. ядра всех трёх звёзд, 60 px ×3: NASA, phase-3h, без слоя, со слоем
    tiles, labels = [], []
    for key in ("центральная", "слева", "внизу справа"):
        x, y = pos[key]
        for name, im in (("NASA", nasa), ("phase-3h", imgs["phase-3h"]), ("без слоя", imgs["B: + тени по max, white 99,98"]),
                         ("со слоем", imgs["C: + слой звезды"])):
            tiles.append(zoom(crop(im, x, y, 60)))
            labels.append(f"{key} · {name} · 60 px ×3")
    contact_sheet([t[::-1] for t in tiles], labels, out / "21_ядра_трёх_звёзд_x3.png", cols=4, tile_inches=3.4,
                  interpolation="nearest", title="Ядра звёзд с насыщенным ядром, 60 px ×3: без провалов и цветных пятен")

    # 3. доли каналов ≥ 0,99 по шагам (слои растяжки C с диска)
    layers, _ = load_stretched(cfg, P)
    field = stats_field(cfg, P, list(layers.values()))
    sd, sky = load_nebula_sd(cfg, P), load_sky_mask(cfg, P)
    b, _ = base_color(cfg, layers, sky, sd, field=field)
    x0 = np.clip(b, 0, 1)[::-1]
    x1 = apply_shadows(np.clip(b, 0, 1), cfg.color.shadows, sd)
    x2 = saturate(x1, cfg.color.saturation, cfg.color.saturation_gamut)[::-1]
    x1 = x1[::-1]
    rows = []
    for cn in ("inner", "core"):
        cr = crops[cn]
        for step, im in (("до shadows", x0), ("после shadows", x1), ("после saturate", x2),
                         ("после finish", imgs["C: + слой звезды"]), ("phase-3h, после finish", imgs["phase-3h"])):
            c_ = crop(im, cr.x, cr.y, cr.size)
            fr = [(c_[..., k] >= 0.99).mean() * 100 for k in range(3)]
            rows.append(f"{cn}\t{step}\t" + "\t".join(f"{v:.1f}" for v in fr))
            print(f"{cn} | {step}: R/G/B ≥ 0,99 = {fr[0]:.1f} / {fr[1]:.1f} / {fr[2]:.1f}%")
    (out / "доли_каналов_3i.tsv").write_text("кроп\tшаг\tR\tG\tB\n" + "\n".join(rows) + "\n", encoding="utf-8")

    # 4. три карты разности вне 150 px от центральной звезды, по маске туманности
    neb = (sd > 0)[::-1]
    yy, xx = np.indices(neb.shape)
    far = neb & (np.hypot(xx - cx, yy - cy) > 150)
    pairs = [("выборка без звёзд", "phase-3h", "A: выборка без звёзд"),
             ("тени по max, насыщенность limit, white F090W/F187N 99,98", "A: выборка без звёзд", "B: + тени по max, white 99,98"),
             ("слой звёзд с насыщенным ядром", "B: + тени по max, white 99,98", "C: + слой звезды")]
    maps, mlabels, res = [], [], {}
    for what, a, bname in pairs:
        d = np.abs(imgs[bname] - imgs[a]).mean(-1)
        res[what] = round(float(d[far].mean()), 4)
        maps.append(np.clip(np.stack([d, d * 0.55, d * 0.15], -1) * 8, 0, 1))
        mlabels.append(f"{what}: средняя вне 150 px {res[what]:.4f}")
    contact_sheet([m[::-1] for m in maps], mlabels, out / "20_разность_по_причинам.png", cols=3, tile_inches=5,
                  title="Разность showcase с phase-3h по шагам (×8): выборка → режим и white → слой звезды")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
