"""Фаза 6: опорное небо (stretch.sky_reference: darkest_blocks) до растяжки — небо это или слабая туманность.

Лист 1 (<папка>/01_опорные_блоки.png): опорный фильтр целиком (asinh) с тремя опорными блоками в рамках и для сравнения —
самые светлые блоки (туманность) и медианные.
Лист 2 (<папка>/02_блоки_x5.png): каждый опорный блок во всех фильтрах при ×5: отображение от медианы блока − 3σ шума
до медианы + 2σ — ровное зерно без рисунка — небо; волокна, градиент, прожилки — туманность.
Числа (<папка>/03_числа.tsv): по фильтру — медиана опорных блоков против процентилей кадра (1, 5, 50) и «структурность»:
σ средних по ячейкам 16×16 px в блоке, делённая на σ шума / 16 (у белого шума ≈ 1; у шума с корреляцией дизеринга —
до ~2–3; туманность — десятки). Шум σ — по разностям соседних пикселей (MAD), он не видит плавного рисунка.

Запуск: uv run python scripts/skyref_blocks.py <конфиг> <стиль> <папка>   (bin — из конфига)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from scipy import ndimage

from skyforge.clean import clean_dir
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet
from skyforge.stretch import darkest_blocks

CELL = 16


def noise_sigma(a: np.ndarray) -> float:
    d = np.diff(a, axis=1)
    d = d[np.isfinite(d)]
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2)) if d.size else float("nan")


def structure(a: np.ndarray, sigma: float) -> float:
    h, w = (a.shape[0] // CELL) * CELL, (a.shape[1] // CELL) * CELL
    cells = np.nanmean(a[:h, :w].reshape(h // CELL, CELL, w // CELL, CELL), axis=(1, 3))
    return float(np.nanstd(cells) / (sigma / CELL)) if sigma > 0 else float("nan")


def asinh_show(a: np.ndarray) -> np.ndarray:
    v = a[np.isfinite(a)]
    lo, hi = np.percentile(v, [1, 99.7])
    x = np.clip((np.nan_to_num(a, nan=lo) - lo) / max(hi - lo, 1e-12), 0, None)
    return np.arcsinh(x * 20) / np.arcsinh(20)


def main() -> None:
    cfg_path, style, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load(cfg_path, style)
    d = clean_dir(c, P)
    arrays = {f: np.load(d / f"{f}.npy") for f in c.filters}
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    stars = ndimage.binary_dilation(np.load(d / "star_weight.npy") > 0.01, iterations=3)
    s = c.stretch
    block = max(16, s.sky_reference_block_px // c.bin)
    ref = arrays[c.reference]
    ok = full & ~stars
    mask, info = darkest_blocks(ref, ok, block, s.sky_reference_n_blocks, s.sky_reference_min_free)
    H, W = ref.shape
    allb = []
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            m = ok[y:y + block, x:x + block]
            if m.mean() >= s.sky_reference_min_free:
                allb.append((float(np.median(ref[y:y + block, x:x + block][m])), y, x))
    allb.sort()
    dark = allb[:s.sky_reference_n_blocks]
    mid = [allb[len(allb) // 2]]
    bright = [allb[-1]]
    print(f"блоков {block} px (bin {c.bin}): в выборке {len(allb)} из {(H // block) * (W // block)}; "
          f"опорные (строка, столбец сетки; медиана {c.reference}): " +
          "; ".join(f"({y}, {x}) {v:.4g}" for v, y, x in dark), flush=True)

    img = np.repeat(asinh_show(ref)[..., None], 3, axis=2)
    for group, col in ((dark, (0.2, 1, 0.2)), (mid, (1, 1, 0.2)), (bright, (1, 0.3, 0.3))):
        for _, y, x in group:
            t = max(1, block // 40)
            img[y:y + t, x:x + block] = img[y + block - t:y + block, x:x + block] = col
            img[y:y + block, x:x + t] = img[y:y + block, x + block - t:x + block] = col
    contact_sheet([img], [f"{c.reference} целиком (asinh) · зелёные — опорные блоки, жёлтый — медианный, "
                          f"красный — самый светлый"], out / "01_опорные_блоки.png", cols=1, tile_inches=11,
                  title=f"{c.target} ({c.slug}): опорное небо darkest_blocks до растяжки, bin {c.bin}")

    tiles, labels, rows = [], [], ["фильтр\tблок\tмедиана блока\tp1 кадра\tp5 кадра\tp50 кадра\tσ шума\t"
                                   "структурность (σ ячеек 16 px / (σ/16))"]
    for f in c.filters:
        a = arrays[f]
        fv = a[full]
        p1, p5, p50 = np.percentile(fv, [1, 5, 50])
        for k, (_, y, x) in enumerate(dark + mid + bright):
            sub = a[y:y + block, x:x + block].copy()
            m = ok[y:y + block, x:x + block]
            sub[~m] = np.nan
            med, sig = float(np.nanmedian(sub)), noise_sigma(sub)
            st = structure(sub, sig)
            lab = ("опорный " + str(k + 1)) if k < len(dark) else ("медианный" if k == len(dark) else "светлый")
            rows.append(f"{f}\t{lab}\t{med:.4g}\t{p1:.4g}\t{p5:.4g}\t{p50:.4g}\t{sig:.3g}\t{st:.1f}")
            if k < len(dark) + 1:
                show = np.clip((np.nan_to_num(sub, nan=med) - (med - 3 * sig)) / (5 * sig), 0, 1)
                tiles.append(show[::-1])
                labels.append(f"{f} · {lab} · ×5\nструктурность {st:.1f}")
    contact_sheet(tiles, labels, out / "02_блоки_x5.png", cols=len(dark) + 1, tile_inches=3.2, interpolation="nearest",
                  title=f"{c.target}: опорные блоки и медианный блок во всех фильтрах, ±σ шума (до растяжки, bin {c.bin})")
    (out / "03_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
