"""3i: итоговые листы — «phase-3h | 3i | NASA» целиком и кропы приёмки (100%, sky-кропы ещё ×5).

Запуск: uv run python scripts/final_sheet_3i.py <папка> <прогон --all> [стиль] [baseline] [фаза]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import with_resolved_crops
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402


def load(p: pathlib.Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255


def crop(img: np.ndarray, x: int, y: int, s: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0 = int(np.clip(x - s // 2, 0, w - s)), int(np.clip(y - s // 2, 0, h - s))
    return img[y0:y0 + s, x0:x0 + s]


def main() -> None:
    out, run = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    style = sys.argv[3] if len(sys.argv) > 3 else "showcase"
    tag = sys.argv[4] if len(sys.argv) > 4 else "phase-3h"
    phase = sys.argv[5] if len(sys.argv) > 5 else "3i"
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load(f"configs/ngc3132/{style}.yaml", style).model_copy(update={"bin": 1})
    now = load(run / style / f"ngc3132_{style}.png")
    base = load(P.root / "data" / "baselines" / "ngc3132" / tag / f"{style}_bin1.png")
    nasa = nasa_on_grid(now, str(P.abs(cfg.reference_image)))[::-1]
    cols = ((tag, base), (phase, now), ("NASA", nasa))
    contact_sheet([shrink(im, 900)[::-1] for _, im in cols], [n for n, _ in cols], out / f"30_{style}_{tag}_{phase}_NASA.png",
                  cols=3, tile_inches=6, title=f"{style}: {tag} | {phase} | NASA (NASA приведён к нашей сетке по трём звёздам)")
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    order = ["core", "inner", "edge", "star_left", "sky", "sky2", "sky3", "sky4"]
    tiles, labels = [], []
    for n in order:
        c = crops[n]
        for name, im in cols:
            tiles.append(crop(im, c.x, c.y, c.size))
            labels.append(f"{n} · {name} · 100%")
        if n.startswith("sky"):
            for name, im in cols:
                tiles.append(np.clip(crop(im, c.x, c.y, c.size) * 5, 0, 1))
                labels.append(f"{n} · {name} · ×5")
    stars = [("центральная", (1134, 877)), ("внизу справа", (1902, 1701)), ("слева", (767, 892)),
             ("поле 1", (1966, 1179)), ("поле 2", (276, 1412)), ("поле 3", (845, 1758))]
    for sname, (x, y) in stars:
        for name, im in cols:
            tiles.append(np.kron(im[y - 40:y + 40, x - 40:x + 40], np.ones((3, 3, 1))))
            labels.append(f"{sname} · {name} · 80 px ×3")
    contact_sheet([t[::-1] for t in tiles], labels, out / f"31_{style}_кропы_приёмки.png", cols=3, tile_inches=3.6,
                  interpolation="nearest", title=f"{style}: кропы приёмки {phase} — {tag}, {phase}, NASA")


if __name__ == "__main__":
    main()
