"""Фаза 4: лист 41 — обработка Тевено (Commons, CC BY-SA 4.0) против нашего итога Sh2-305 на одной сетке (bin 1):
целиком и ×8, кропы star, bubble, filaments, globule при 100%, sky, sky2, seam ×5; таблица — разброс яркости и медиана
R/G/B в кропах. Совмещение — как scripts/faint_s305.py (поворот 90° по часовой, сдвиг по корреляции).

Запуск: uv run python scripts/thevenot_compare_s305.py <png нашего итога bin 1> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import load_aligned, with_resolved_crops
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from faint_s305 import thevenot_on_grid  # noqa: E402

Image.MAX_IMAGE_PIXELS = None


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    load_aligned(c, P)
    ours = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255      # картинка: строка 0 — верх
    H, W = ours.shape[:2]
    thev, reg = thevenot_on_grid(ours @ LUMA, 1)
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    tiles, labels, rows = [], [], ["автор\tкроп\tσ яркости\tмедиана R/G/B"]
    for lab, im in (("Тевено", thev), ("наша", ours)):
        tiles.append(shrink(im, 1000)[::-1]); labels.append(f"{lab}, целиком")
    for lab, im in (("Тевено", thev), ("наша", ours)):
        tiles.append(np.clip(shrink(im, 1000) * 8, 0, 1)[::-1]); labels.append(f"{lab}, ×8")
    for name, k in (("star", 1), ("bubble", 1), ("filaments", 1), ("globule", 1), ("sky", 5), ("sky2", 5), ("seam", 5)):
        cr = crops[name]
        sl = display_crop((H, W), cr.x, cr.y, cr.size, 1)
        sl = (slice(H - sl[0].stop, H - sl[0].start), sl[1])                   # срез сетки → строки картинки
        for lab, im in (("Тевено", thev), ("наша", ours)):
            t = im[sl]
            tiles.append(np.clip(t * k, 0, 1)[::-1]); labels.append(f"{lab}, {name} " + (f"×{k}" if k > 1 else "100%"))
            rows.append(f"{lab}\t{name}\t{(t @ LUMA * 255).std():.1f}\t"
                        f"{'/'.join(f'{np.median(t[..., i]) * 255:.0f}' for i in range(3))}")
    contact_sheet(tiles, labels, out / "41_сравнение_с_Тевено.png", cols=2, tile_inches=6, interpolation="lanczos",
                  title=f"Sh2-305: Мелина Тевено (Commons, CC BY-SA 4.0, 2026-04-03) против нашей (bin 1), одна сетка; {reg}")
    (out / "41_сравнение.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
