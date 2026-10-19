"""3j: радиальный профиль яркости от центральной звезды по секторам между лучами — наши картинки и NASA
на одной сетке. Показывает, где и во сколько раз внешний ореол темнее, чем у NASA.

Запуск: uv run python scripts/halo_profile_3j.py <папка> <имя> <png> [<png> ...]
Первая кривая — NASA (приведён к сетке первой картинки по двум звёздам), дальше переданные PNG.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722])
CENTER = (1133, 876)                                   # центральная звезда на итоговой картинке
SPIKES = (28, 89, 151, 212, 271, 329)                  # лучи ярких звёзд в F090W, градусы против часовой от +x
HALF = 8                                               # полуширина сектора между лучами, градусы


def load(p: str) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float32) / 255


def profile(img: np.ndarray, step: int = 10) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    yy, xx = np.indices((h, w))
    dx, dy = xx - CENTER[0], CENTER[1] - yy
    r = np.hypot(dx, dy)
    ang = np.degrees(np.arctan2(dy, dx)) % 360
    mids = [((a + b) / 2) % 360 for a, b in zip(SPIKES, SPIKES[1:] + (SPIKES[0] + 360,))]
    sector = np.zeros((h, w), bool)
    for m in mids:
        sector |= np.abs((ang - m + 180) % 360 - 180) <= HALF
    lum = img @ LUMA
    edges = np.arange(0, r[sector].max(), step)
    prof = np.array([np.median(lum[sector & (r >= a) & (r < a + step)]) if (sector & (r >= a) & (r < a + step)).any()
                     else np.nan for a in edges])
    return edges + step / 2, prof


def main() -> None:
    out, name, pngs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    first = load(pngs[0])
    nasa = nasa_on_grid(first, "data/reference/ngc3132_nasa.jpg", [(1133, 876), (1901, 1701)],
                        [(1491, 1313), (2500, 2388)])[::-1]
    curves = [("NASA", *profile(nasa))] + [(pathlib.Path(p).parent.parent.name[-15:] + "/" + pathlib.Path(p).stem.split("_")[-1],
                                            *profile(load(p))) for p in pngs]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for lab, r, pr in curves:
        ax[0].plot(r, pr * 255, label=lab)
    ax[0].set_yscale("log"); ax[0].set_xlabel("радиус от центральной звезды, px"); ax[0].set_ylabel("яркость, уровни 0–255")
    ax[0].set_title("Профиль между лучами (медиана по секторам)"); ax[0].legend(); ax[0].grid(alpha=0.3)
    rn, pn = curves[0][1], curves[0][2]
    for lab, r, pr in curves[1:]:
        n = min(len(pn), len(pr))
        ax[1].plot(r[:n], pr[:n] / np.maximum(pn[:n], 1e-4), label=lab)
    ax[1].axhline(1, color="k", lw=0.8); ax[1].set_yscale("log")
    ax[1].set_xlabel("радиус, px"); ax[1].set_title("Наш / NASA"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=90)
    sky_n = np.nanmedian(pn[-8:])
    print(f"NASA: небо (последние 80 px) {sky_n * 255:.2f} уровня")
    for lab, r, pr in curves:
        rows = []
        for rad in (300, 400, 500, 600, 700, 800, 900, 1000):
            i = int(rad // 10)
            if i < len(pr):
                rows.append(f"{rad}: {pr[i] * 255:.1f}")
        print(f"{lab}: " + ", ".join(rows))


if __name__ == "__main__":
    main()
