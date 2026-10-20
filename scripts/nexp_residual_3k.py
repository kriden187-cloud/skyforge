"""3k, п.2: карта остатков фона по фильтрам — небо, сглаженное по пустому небу, минус его медиана, в σ неба;
до сшивки уровней NEXP (совмещённые кадры) и после (очищенные), с границами NEXP своего фильтра.

Запуск: uv run python scripts/nexp_residual_3k.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import load_input, load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import nexp_maps, smooth_on  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SMOOTH_PX = 24


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    raw, wcs, _ = load_aligned(cfg, P)
    clean, _ = load_input(cfg, P)
    sky = load_sky_mask(cfg, P)
    ne = {f: ndimage.median_filter(n, 15) for f, n in nexp_maps(P, wcs, sky.shape).items()}
    fig, ax = plt.subplots(2, len(clean), figsize=(6 * len(clean), 11))
    rows = ["фильтр\tσ неба\tразброс сглаженного неба (5–95%), σ: до / после"]
    for i, (f, a) in enumerate(clean.items()):
        sig = None
        spread = []
        for r, (lab, arr) in enumerate((("до сшивки", np.asarray(raw[f], dtype=np.float32)), ("после", a))):
            ok = sky & np.isfinite(arr)
            v = arr[ok]
            med = float(np.median(v))
            if sig is None:
                sig = float(1.4826 * np.median(np.abs(v - med)))
            m = smooth_on(ok, np.where(ok, arr, 0) - med, SMOOTH_PX) / sig
            spread.append(float(np.nanpercentile(m, 95) - np.nanpercentile(m, 5)))
            im = ax[r, i].imshow(m[::-1], cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            ax[r, i].contour(ne[f][::-1], levels=np.arange(0.5, 9, 1), colors="k", linewidths=0.4)
            ax[r, i].set_title(f"{f}: {lab}, σ неба {sig:.3g}"); ax[r, i].axis("off")
        fig.colorbar(im, ax=ax[:, i], fraction=0.03, label="небо − медиана, σ (сглажено 24 px)")
        rows.append(f"{f}\t{sig:.4g}\t{spread[0]:.2f} / {spread[1]:.2f}")
    fig.suptitle("Остатки фона по пустому небу: до сшивки уровней NEXP (сверху) и после (снизу); чёрные — границы NEXP")
    fig.savefig(out / "12_остатки_фона_по_фильтрам.png", dpi=50); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / "12_остатки_фона.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
