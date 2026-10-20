"""3k, п.2: постоянен ли скачок уровня вдоль границы областей NEXP. Для каждой длинной границы (пара связных
областей, как в clean) — скачок по отрезкам SEG_PX вдоль границы: медиана неба в полосе 30 px с одной стороны
минус с другой, в σ неба. Если вдоль одной границы скачок меняется сильнее, чем его среднее, константа на
область его не снимет. До сшивки (совмещённые кадры) и после (очищенные).

Запуск: uv run python scripts/nexp_along_3k.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.background import nexp_regions
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask, nexp_on_grid
from skyforge.config import Config
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEG_PX, BAND_PX, MIN_BOUNDARY_PX = 150, 30, 600
FAR_PX = 300  # «вдали от туманности»: медиана расстояния пикселей отрезка до маски туманности больше FAR_PX


def segments(reg: np.ndarray, sky: np.ndarray, i: int, j: int, sd: np.ndarray) -> list:
    """Отрезки границы i|j: (середина, пиксели неба со стороны i, со стороны j, окно, расстояние до туманности)."""
    A, B = reg == i, reg == j
    edge = ndimage.binary_dilation(A, iterations=2) & B
    ys, xs = np.nonzero(edge)
    vertical = np.ptp(ys) >= np.ptp(xs)
    pos = ys if vertical else xs
    y0, y1 = max(0, ys.min() - BAND_PX - 2), ys.max() + BAND_PX + 3
    x0, x1 = max(0, xs.min() - BAND_PX - 2), xs.max() + BAND_PX + 3
    sub = (slice(y0, y1), slice(x0, x1))
    d_edge, idx = ndimage.distance_transform_edt(~edge[sub], return_indices=True)
    near = (d_edge <= BAND_PX) & sky[sub]
    p_of = (idx[0] + y0) if vertical else (idx[1] + x0)
    out = []
    for p0 in range(int(pos.min()), int(pos.max()) + 1, SEG_PX):
        seg = near & (p_of >= p0) & (p_of < p0 + SEG_PX)
        dist = float(-np.median(sd[sub][seg])) if seg.any() else 0.0
        out.append((p0 + SEG_PX / 2, seg & A[sub], seg & B[sub], sub, dist))
    return out


def along(a: np.ndarray, segs, sig: float):
    out = []
    for p, ma, mb, sub, dist in segs:
        va, vb = a[sub][ma], a[sub][mb]
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if va.size >= 300 and vb.size >= 300:
            out.append((p, (np.median(vb) - np.median(va)) / sig, dist))
    return out


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    raw, wcs, _ = load_aligned(cfg, P)
    clean, _ = load_input(cfg, P)
    sky = load_sky_mask(cfg, P)
    sd = load_nebula_sd(cfg, P)
    bc = cfg.background
    rows = ["фильтр\tграница (NEXP i→j)\tдлина, px\tотрезков\tскачок до: среднее, разброс вдоль (σ)"
            "\tпосле: среднее, разброс вдоль, мин…макс (σ)"
            "\tпосле, дальше 300 px от туманности: среднее, разброс, наибольший |скачок| (отрезков)"]
    worst = []
    for f, a in clean.items():
        ne = nexp_on_grid(cfg, P, f, wcs, a.shape)
        reg = nexp_regions(ne, np.isfinite(a), bc.nexp_smooth_px, bc.nexp_min_area_px)
        nem = ndimage.median_filter(ne, bc.nexp_smooth_px)
        ok = sky & np.isfinite(a)
        sig = float(1.4826 * np.median(np.abs(a[ok] - np.median(a[ok]))))
        n = int(reg.max()) + 1
        pairs = {}
        for ia, ib in ((reg[:-1, :], reg[1:, :]), (reg[:, :-1], reg[:, 1:])):
            m = (ia != ib) & (ia >= 0) & (ib >= 0)
            k, c = np.unique(np.minimum(ia[m], ib[m]).astype(np.int64) * n + np.maximum(ia[m], ib[m]), return_counts=True)
            for kk, cc in zip(k, c):
                pairs[int(kk)] = pairs.get(int(kk), 0) + int(cc)
        for kk, length in sorted(pairs.items(), key=lambda t: -t[1]):
            if length < MIN_BOUNDARY_PX:
                continue
            i, j = divmod(kk, n)
            ni, nj = int(np.median(nem[reg == i])), int(np.median(nem[reg == j]))
            if ni > nj:
                i, j, ni, nj = j, i, nj, ni
            segs = segments(reg, sky, i, j, sd)
            s0 = along(np.asarray(raw[f], dtype=np.float32), segs, sig)
            s1 = along(a, segs, sig)
            if len(s1) < 3:
                continue
            v0, v1 = np.array([v for _, v, _ in s0]), np.array([v for _, v, _ in s1])
            far = np.array([v for _, v, d in s1 if d > FAR_PX])
            far_s = f"{far.mean():+.2f}, {far.std():.2f}, {np.abs(far).max():.2f} ({far.size})" if far.size >= 2 else "—"
            rows.append(f"{f}\t{ni}→{nj} ({i}|{j})\t{length}\t{len(s1)}\t{v0.mean():+.2f}, {v0.std():.2f}"
                        f"\t{v1.mean():+.2f}, {v1.std():.2f}, {v1.min():+.2f}…{v1.max():+.2f}\t{far_s}")
            worst.append((float(np.abs(v1).max()), f, f"{ni}→{nj}", s0, s1))
    worst.sort(key=lambda t: -t[0])
    fig, ax = plt.subplots(2, 4, figsize=(22, 9))
    for a_, (_, f, lab, s0, s1) in zip(ax.ravel(), worst[:8]):
        a_.plot([p for p, _, _ in s0], [v for _, v, _ in s0], "o-", color="0.6", label="до сшивки")
        a_.plot([p for p, _, _ in s1], [v for _, v, _ in s1], "o-", color="tab:red", label="после (а)")
        for p, v, d in s1:
            if d <= FAR_PX:
                a_.plot(p, v, "s", mfc="none", mec="tab:blue", ms=11)
        a_.axhline(0, color="k", lw=0.5)
        for y in (0.2, -0.2):
            a_.axhline(y, color="tab:green", lw=0.6, ls="--")
        a_.set_title(f"{f}, граница NEXP {lab}"); a_.set_xlabel("положение вдоль границы, px"); a_.set_ylabel("скачок, σ неба")
        a_.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Скачок уровня вдоль границ NEXP по отрезкам {SEG_PX} px (полоса {BAND_PX} px); зелёные — ±0,2σ. "
                 "Восемь границ с наибольшим остатком после (а); синие квадраты — отрезки ближе 300 px к туманности")
    fig.tight_layout(); fig.savefig(out / "13_скачок_вдоль_границ.png", dpi=60); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / "13_скачок_вдоль_границ.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
