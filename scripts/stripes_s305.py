"""Фаза 4, шаг 6: полосы F182M после растяжки при bin 1 — лист 67.

Б (тёплый сдвиг, 4 фильтра) против В (тот же сдвиг + F182M синий) при bin 1: кропы sky, sky2, seam, gap212 при 100%
и ×5. Мера полос — по яркости итога в кропе, звёзды (маска clean с отступом 3 px) исключены: σ медиан столбцов
(вертикальные полосы) и строк (горизонтальные), делённая на ожидаемую от белого шума σ_пикселя/√n (1 — полос нет).

Запуск: uv run python scripts/stripes_s305.py <папка> <конфиг пятифильтровой пробы (папка со showcase.yaml)>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import with_resolved_crops
from skyforge.clean import clean_dir
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from palette_s305 import F182, WARM, spec, workspace  # noqa: E402
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

CROPS = ("sky", "sky2", "seam", "gap212")


def stripes(lum: np.ndarray, ok: np.ndarray) -> tuple[float, float]:
    """σ медиан столбцов и строк / (σ пикселя / √n) по пикселям ok."""
    v = np.where(ok, lum, np.nan)
    # σ пикселя — std с отсечением 4σ (не MAD: небо около 10 уровней в 8 битах почти постоянно, MAD выходит 0)
    x = v[np.isfinite(v)]
    for _ in range(5):
        m0, s0 = x.mean(), x.std()
        x = x[np.abs(x - m0) <= 4 * s0 + 1e-6]
    sig = max(float(x.std()), 1e-3)
    out = []
    for ax in (0, 1):                                    # 0 — медиана по строкам каждого столбца: вертикальные полосы
        n = np.sum(ok, axis=ax)
        med = np.nanmedian(v, axis=ax)
        good = n > 0.5 * ok.shape[ax]
        m = med[good]
        trend = ndimage.uniform_filter1d(m, 51, mode="nearest")    # плавный ход фона — не полосы
        out.append(float(np.std(m - trend) / (sig / np.sqrt(np.median(n[good])))))
    return out[0], out[1]


def main() -> None:
    out, probe = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    P = Paths(pathlib.Path(".").resolve())
    variants = [("Б · 4 фильтра", Config.load("configs/sh2305/showcase.yaml", "showcase"), spec(WARM)),
                ("В · + F182M", Config.load(probe / "showcase.yaml", "showcase"), spec(WARM | F182))]
    tiles, labels, rows = [], [], ["вариант\tкроп\tвертикальные полосы (σ столбцов / шум)\tгоризонтальные (σ строк / шум)"]
    per = {}
    for lab, c0, sp in variants:
        c = apply(c0.model_copy(update={"bin": 1}), sp)
        w = workspace(c, P)
        img = render(c, P, w["arrays"], w["ctx"])                           # картинка: строка 0 — верх
        stars = ndimage.binary_dilation(np.load(clean_dir(c, P) / "star_weight.npy") > 0.01, iterations=3)[::-1]
        crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
        H, W = img.shape[:2]
        per[lab] = []
        for n in CROPS:
            k = crops[n]
            sl = display_crop((H, W), k.x, k.y, k.size, 1)
            sl = (slice(H - sl[0].stop, H - sl[0].start), sl[1])           # срез сетки → строки картинки
            t = img[sl]
            vs, hs = stripes(t @ LUMA * 255, ~stars[sl])
            rows.append(f"{lab}\t{n}\t{vs:.2f}\t{hs:.2f}")
            per[lab].append((n, t, vs, hs))
        del img, w
        print(f"  готово: {lab}", flush=True)
    for i, n in enumerate(CROPS):
        for kk in (1, 5):
            for lab, _, _ in variants:
                _, t, vs, hs = per[lab][i]
                tiles.append(np.clip(t * kk, 0, 1)[::-1])
                labels.append(f"{lab} · {n} · {'100%' if kk == 1 else '×5'}\nполосы: верт. {vs:.2f} · гориз. {hs:.2f}")
    contact_sheet(tiles, labels, out / "67_F182M_полосы_bin1.png", cols=4, tile_inches=4.2, interpolation="nearest")
    (out / "67_полосы.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
