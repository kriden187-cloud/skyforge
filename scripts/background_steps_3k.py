"""3k, шаг 1: карта цвета неба крупно с границами NEXP и локальные скачки поперёк границ NEXP
(медиана неба в полосе 30 px по обе стороны границы — отделяет ступеньку от плавного хода фона и ореола).

Запуск: uv run python scripts/background_steps_3k.py <папка> <итоговый png showcase>
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import load_input, load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import nexp_maps, smooth_on  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    out, final = pathlib.Path(sys.argv[1]), sys.argv[2]
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    raw, wcs, meta = load_aligned(cfg, P)
    clean, _ = load_input(cfg, P)
    sky = load_sky_mask(cfg, P)
    ne = {f: ndimage.median_filter(n, 15) for f, n in nexp_maps(P, wcs, sky.shape).items()}
    rgb = np.asarray(Image.open(final).convert("RGB")).astype(np.float32)[::-1]
    g_s = smooth_on(sky, rgb[..., 1] - (rgb[..., 0] + rgb[..., 2]) / 2, 32)
    fig, ax = plt.subplots(1, 2, figsize=(22, 10))
    im = ax[0].imshow(g_s[::-1], cmap="RdBu_r", vmin=-2, vmax=2)
    fig.colorbar(im, ax=ax[0], fraction=0.04, label="G − (R+B)/2, уровни из 255 (красное — зеленее)")
    ax[0].set_title("Цвет неба showcase (сглажено 32 px по пустому небу) и границы NEXP: чёрные F090W, синие F356W, зелёные F470N")
    for a_ in ax:
        for f, col in (("F090W", "k"), ("F356W", "b"), ("F470N", "g")):
            a_.contour(ne[f][::-1], levels=np.arange(0.5, 9, 1), colors=col, linewidths=0.7)
        a_.axis("off")
    lum_s = smooth_on(sky, rgb @ np.array([0.2126, 0.7152, 0.0722]), 32)
    im = ax[1].imshow(lum_s[::-1], cmap="magma", vmin=np.nanpercentile(lum_s, 2), vmax=np.nanpercentile(lum_s, 98))
    fig.colorbar(im, ax=ax[1], fraction=0.04, label="яркость неба, уровни")
    ax[1].set_title("Яркость неба showcase (сглажено 32 px) и те же границы")
    fig.tight_layout(); fig.savefig(out / "05_цвет_неба_и_границы_NEXP.png", dpi=55); plt.close(fig)

    # локальные скачки поперёк границ NEXP по каждому фильтру (очищенные кадры, в σ неба области с большим NEXP)
    rows = ["фильтр\tграница NEXP\tдлина, px\tскачок медианы (больше NEXP − меньше)\tσ неба\tскачок/σ"]
    for f, a in clean.items():
        n = ne[f]
        for lo in range(1, 8):
            for hi in range(lo + 1, 9):
                A = n == lo
                B = n == hi
                edge = ndimage.binary_dilation(A, iterations=2) & B
                if edge.sum() < 300:
                    continue
                dist_edge = ndimage.distance_transform_edt(~edge)
                band = (dist_edge <= 30) & sky & np.isfinite(a)
                va, vb = a[band & A], a[band & B]
                if va.size < 500 or vb.size < 500:
                    continue
                sig = 1.4826 * np.median(np.abs(vb - np.median(vb)))
                step = np.median(vb) - np.median(va)
                rows.append(f"{f}\t{lo}→{hi}\t{int(edge.sum())}\t{step:+.4g}\t{sig:.4g}\t{step / sig:+.2f}")
    (out / "06_скачки_на_границах_NEXP.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
