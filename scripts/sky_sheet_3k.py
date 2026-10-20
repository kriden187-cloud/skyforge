"""3k: лист неба — NASA и состояния showcase колонками: целиком, целиком ×8, sky-кропы ×5. Медиана яркости
кропов (уровни из 255) — в подписи.

Запуск: uv run python scripts/sky_sheet_3k.py <png листа> <заголовок> <метка>=<png или baseline:тег> ...
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
from final_sheet_3i import crop, load  # noqa: E402
from sweep_white_3i import nasa_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def main() -> None:
    sheet, title = pathlib.Path(sys.argv[1]), sys.argv[2]
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    cols = []
    for arg in sys.argv[3:]:
        name, src = arg.split("=", 1)
        if src.startswith("baseline:"):
            src = str(P.root / "data" / "baselines" / "ngc3132" / src.split(":", 1)[1] / "showcase_bin1.png")
        cols.append((name, load(pathlib.Path(src))))
    nasa = nasa_on_grid(cols[0][1], str(P.abs(cfg.reference_image)))[::-1]
    cols.insert(0, ("NASA", nasa))
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    tiles, labels = [], []
    for name, im in cols:
        tiles.append(shrink(im, 900)); labels.append(f"{name} · целиком")
    for name, im in cols:
        tiles.append(np.clip(shrink(im, 900) * 8, 0, 1)); labels.append(f"{name} · целиком ×8")
    for n in ("sky", "sky2", "sky3", "sky4"):
        c = crops[n]
        for name, im in cols:
            t = crop(im, c.x, c.y, c.size)
            tiles.append(np.clip(t * 5, 0, 1))
            labels.append(f"{n} · {name} · ×5 · небо {np.median(t @ LUMA) * 255:.1f}")
    contact_sheet([t[::-1] for t in tiles], labels, sheet, cols=len(cols), tile_inches=4.2, title=title)


if __name__ == "__main__":
    main()
