"""3k, п.1: цвет неба showcase по областям NEXP при равном уровне.

Цвет пикселя — канал минус среднее трёх каналов (уровни из 255): от уровня неба не зависит, ступеньки уровня
из п.2 в него не входят. Два замера:
1. По областям NEXP (F090W/F356W/F470N): медиана и среднее цвета; разница с областью наибольшего NEXP.
2. Локальные пары у границ областей: полоса 30 px по обе стороны, разница медиан цвета.
Итог берётся из 16-битного TIFF (если есть), иначе из PNG.

Запуск: uv run python scripts/sky_color_3k.py <папка> <итог: tif или png> [метка]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import load_sky_mask
from skyforge.config import Config
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import nexp_maps  # noqa: E402


def load_levels(path: str) -> np.ndarray:
    """Итог в уровнях из 255, ориентация сетки (строка 0 — низ)."""
    p = pathlib.Path(path)
    if p.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(p).astype(np.float32)
        a = a / (65535 if a.max() > 255 else 255) * 255
    else:
        a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)
    return a[..., :3][::-1]


def main() -> None:
    out, final = pathlib.Path(sys.argv[1]), sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else ""
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    _, wcs, _ = load_aligned(cfg, P)
    sky = load_sky_mask(cfg, P)
    ne = {f: ndimage.median_filter(n, 15) for f, n in nexp_maps(P, wcs, sky.shape).items()}
    rgb = load_levels(final)
    chroma = rgb - rgb.mean(axis=-1, keepdims=True)
    combo = ne["F090W"] * 100 + ne["F356W"] * 10 + np.minimum(ne["F470N"], 9)
    name = lambda k: f"{k // 100}/{(k // 10) % 10}/{k % 10}"

    rows = [f"# {label}", "область (NEXP F090W/F356W/F470N)\tпикселей\tуровень (медиана яркости)"
            "\tцвет R, G, B: медиана\tцвет R, G, B: среднее"]
    stats = {}
    for k in np.unique(combo[sky]):
        m = sky & (combo == k)
        if m.sum() < 5000:
            continue
        c = chroma[m]
        stats[k] = (np.median(c, axis=0), c.mean(axis=0), int(m.sum()))
        rows.append(f"{name(k)}\t{int(m.sum())}\t{np.median(rgb[m].mean(-1)):.2f}\t"
                    + ", ".join(f"{v:+.2f}" for v in stats[k][0]) + "\t" + ", ".join(f"{v:+.2f}" for v in stats[k][1]))
    ref = max(stats, key=lambda k: stats[k][2])           # самая большая область — полное покрытие
    rows.append("")
    rows.append(f"разница цвета с областью {name(ref)}: медиана / среднее, по каналам R, G, B")
    worst_med, worst_mean = np.zeros(3), np.zeros(3)
    for k, (md, mn, n) in stats.items():
        dm, dn = md - stats[ref][0], mn - stats[ref][1]
        worst_med, worst_mean = np.maximum(worst_med, np.abs(dm)), np.maximum(worst_mean, np.abs(dn))
        rows.append(f"{name(k)}\t" + ", ".join(f"{v:+.2f}" for v in dm) + "\t" + ", ".join(f"{v:+.2f}" for v in dn))
    rows.append(f"наибольшая |разница| по областям: медиана {', '.join(f'{v:.2f}' for v in worst_med)}; "
                f"среднее {', '.join(f'{v:.2f}' for v in worst_mean)}")
    for k in stats:
        if k // 100 == 1:
            rows.append(f"NEXP1 ({name(k)}) против {name(ref)}: медиана "
                        + ", ".join(f"{v:+.2f}" for v in stats[k][0] - stats[ref][0]) + "; среднее "
                        + ", ".join(f"{v:+.2f}" for v in stats[k][1] - stats[ref][1]))

    # локальные пары у границ: полоса 30 px по обе стороны
    rows.append("")
    rows.append("граница областей\tдлина, px\tскачок уровня\tскачок цвета R, G, B (медианы)")
    worst = np.zeros(3)
    keys = list(stats)
    for i, ka in enumerate(keys):
        A = combo == ka
        dA = ndimage.distance_transform_edt(~A)
        for kb in keys[i + 1:]:
            B = combo == kb
            edge = ndimage.binary_dilation(A, iterations=2) & B
            if edge.sum() < 300:
                continue
            band = (ndimage.distance_transform_edt(~edge) <= 30) & sky
            ia, ib = band & A & (dA <= 32), band & B
            if ia.sum() < 500 or ib.sum() < 500:
                continue
            dc = np.median(chroma[ib], axis=0) - np.median(chroma[ia], axis=0)
            dl = np.median(rgb[ib].mean(-1)) - np.median(rgb[ia].mean(-1))
            worst = np.maximum(worst, np.abs(dc))
            rows.append(f"{name(ka)} → {name(kb)}\t{int(edge.sum())}\t{dl:+.2f}\t" + ", ".join(f"{v:+.2f}" for v in dc))
    rows.append(f"наибольший |скачок цвета| у границ: {', '.join(f'{v:.2f}' for v in worst)}")
    text = "\n".join(rows) + "\n"
    (out / "09_цвет_неба_при_равном_уровне.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
