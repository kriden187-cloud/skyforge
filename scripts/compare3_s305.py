"""Фаза 4, шаг 7: лист 42 — Тевено (Commons, CC BY-SA 4.0), Маккохрейн (PI программы 4547, Commons, CC BY-SA 4.0) и наша
(bin 1) на одной сетке; лист 71 — кроп seam: проба с F182M в маске (лист шаг6/69) против финала без F182M в маске.

Совмещение — scripts/refgrid_s305.py: Тевено — поворот 90° по часовой, Маккохрейн — без поворота; сдвиг по корреляции.
Таблица — разброс яркости и медиана R/G/B в кропах, как лист 41.

Запуск: uv run python scripts/compare3_s305.py <png нашего итога bin 1> <папка листов приёмки> <папка шага 7>
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
from refgrid_s305 import ref_on_grid  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
REF = pathlib.Path("data/reference/sh2305")
CROPS = (("star", 1), ("bubble", 1), ("filaments", 1), ("globule", 1), ("sky", 5), ("sky2", 5), ("seam", 5))


def main() -> None:
    png, out, step = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/sh2305/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    load_aligned(c, P)
    ours = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32) / 255      # картинка: строка 0 — верх
    H, W = ours.shape[:2]
    lum = ours @ LUMA
    thev, rt = ref_on_grid(REF / "thevenot_lw_commons.jpg", -1, lum)
    mcc, rm = ref_on_grid(REF / "mccaughrean_commons.jpg", 0, lum)
    print("Тевено:", rt, "| Маккохрейн:", rm, flush=True)
    cols = (("Тевено (4 LW)", thev), ("Маккохрейн (у него F182M тоже синий)", mcc), ("наша (F182M синий + 4 LW)", ours))
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    tiles, labels, rows = [], [], ["автор\tкроп\tσ яркости\tмедиана R/G/B"]
    for lab, im in cols:
        tiles.append(shrink(im, 1000)[::-1]); labels.append(f"{lab}, целиком")
    for name, k in CROPS:
        cr = crops[name]
        sl = display_crop((H, W), cr.x, cr.y, cr.size, 1)
        sl = (slice(H - sl[0].stop, H - sl[0].start), sl[1])                   # срез сетки → строки картинки
        for lab, im in cols:
            t = im[sl]
            tiles.append(np.clip(t * k, 0, 1)[::-1]); labels.append(f"{lab.split(' (')[0]}, {name} " + (f"×{k}" if k > 1 else "100%"))
            rows.append(f"{lab.split(' (')[0]}\t{name}\t{(t @ LUMA * 255).std():.1f}\t"
                        f"{'/'.join(f'{np.median(t[..., i]) * 255:.0f}' for i in range(3))}")
    contact_sheet(tiles, labels, out / "42_сравнение_PI_Тевено_наша.png", cols=3, tile_inches=5, interpolation="lanczos")
    (out / "42_сравнение.tsv").write_text("\n".join(rows) + f"\nТевено: {rt}\nМаккохрейн: {rm}\n"
                                          "у Маккохрейна F182M тоже есть (синий канал)\n", encoding="utf-8")
    print("\n".join(rows))
    # лист 71: seam — проба шага 6 (F182M строил маску) против финала (маска без F182M)
    cr = crops["seam"]
    sl = display_crop((H, W), cr.x, cr.y, cr.size, 1)
    sl = (slice(H - sl[0].stop, H - sl[0].start), sl[1])
    t = ours[sl]
    old = [np.asarray(Image.open(step.parent / "шаг6" / f"69_seam_В_пятно_маски_{s}.png").convert("RGB"),
                      dtype=np.float32) / 255 for s in ("100", "x5")]
    contact_sheet([old[0][::-1], t[::-1], old[1][::-1], np.clip(t * 5, 0, 1)[::-1]],
                  ["шаг 6: слева Б (4 фильтра), справа В (F182M в маске, 30,1% кадра) · seam 100%",
                   "финал: F182M, маска без него (21,1%) · seam 100%", "шаг 6: Б | В · seam ×5", "финал · seam ×5"],
                  step / "71_seam_маска_без_F182M.png", cols=2, tile_inches=5, interpolation="nearest")


if __name__ == "__main__":
    main()
