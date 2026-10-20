"""3б, закрытие: следы лучей на небе итога против звёзд неба — отношение пика в фильтре пятна к пику F658N в том же
месте, в counts/s (экспозиции F502N 400 с, F656N 400 с, F658N 700 с; BUNIT drz — counts). Пик — наибольшее превышение
над медианой 9×9 в окне 3×3 до чистки. Пятна — одноцветные точки неба итога (один канал > 2× двух других, > 15 уровней
над фоном); звёзды — пики F658N > 8σ на тихом небе, видные и в F656N. Вопрос: есть ли зазор ≥ 3× между ними.

Запуск: uv run python scripts/cosmic_ratio_3b.py <png итога bin 2> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned
from skyforge.clean import load_nebula_sd, load_sky_mask
from skyforge.cosmics import quiet_sky
from skyforge.config import Config
from skyforge.paths import Paths

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EXPTIME = {"F502N": 400.0, "F656N": 400.0, "F658N": 700.0}
CHANNEL = {0: "F658N", 1: "F656N", 2: "F502N"}          # R, G, B итога


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/ngc3132hst/showcase.yaml", "showcase")
    raw, _, _ = load_aligned(c, P)
    sd = load_nebula_sd(c, P)
    H, W = sd.shape
    exc = {}
    for f, a in raw.items():
        a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
        exc[f] = (a - ndimage.median_filter(a, 9)) / EXPTIME[f]          # counts/s над местным фоном
    peak = {f: ndimage.maximum_filter(e, 3) for f, e in exc.items()}

    # пятна: одноцветные точки неба итога
    im = np.asarray(Image.open(png).convert("RGB")).astype(np.float32)[::-1]
    ex = im - ndimage.median_filter(im, size=(15, 15, 1))
    mx, srt = ex.max(-1), np.sort(ex, -1)
    col = (mx > 15) & (srt[..., 2] > 2.0 * np.maximum(srt[..., 1], 1)) & (sd < 0)
    lab, n = ndimage.label(ndimage.binary_dilation(col, iterations=1))
    spots = {"F502N": [], "F656N": [], "F658N": []}
    for p in ndimage.maximum_position(mx, lab, np.arange(1, n + 1)):
        spots[CHANNEL[int(np.argmax(ex[p]))]].append(p)

    # звёзды неба: пики F658N > 8σ на тихом небе вне маски туманности (отбор шага 2, лист 06) и видные в F656N > 4σ —
    # у звезды Hα и [N II] близки (отношение ~1); так отсекаются артефакты F658N и следы, совпавшие с бугорком F658N
    def smooth_exc(a):
        a = np.nan_to_num(a, nan=float(np.nanmedian(a)))
        e = ndimage.gaussian_filter(a, 1.0) - ndimage.median_filter(a, 25)
        return e, 1.4826 * np.median(np.abs(e[::4, ::4] - np.median(e[::4, ::4])))
    E = {f: smooth_exc(a) for f, a in raw.items()}
    e8, s8 = E["F658N"]
    quiet = quiet_sky(raw["F658N"], load_sky_mask(c, P), c.clean.cosmics.sky_trusted_quiet_sigma)
    pk = (e8 == ndimage.maximum_filter(e8, 15)) & (e8 > 8 * s8) & (sd < 0) & quiet
    ys, xs = np.nonzero(pk)
    stars = [(y, x) for y, x in zip(ys, xs) if 4 <= y < H - 4 and 4 <= x < W - 4
             and E["F656N"][0][y - 1:y + 2, x - 1:x + 2].max() > 4 * E["F656N"][1]]

    floor = 1e-12
    rows = ["фильтр\tчто\t(y, x)\tпик фильтра, counts/s\tпик F658N, counts/s\tотношение"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    verdict = []
    for ax, f in zip(axes, ("F502N", "F656N")):
        r_st = [(peak[f][p] / max(peak["F658N"][p], floor), p) for p in stars]
        r_sp = [(peak[f][p] / max(peak["F658N"][p], floor), p) for p in spots[f]]
        for r, p in r_st:
            rows.append(f"{f}\tзвезда\t{p}\t{peak[f][p]:.4g}\t{peak['F658N'][p]:.4g}\t{r:.3g}")
        for r, p in r_sp:
            rows.append(f"{f}\tпятно\t{p}\t{peak[f][p]:.4g}\t{peak['F658N'][p]:.4g}\t{r:.3g}")
        st = np.array([r for r, _ in r_st])
        sp = np.array([r for r, _ in r_sp])
        cap = 1e4
        ax.scatter(np.clip(st, 1e-2, cap), np.full(len(st), 1.0) + np.linspace(-0.08, 0.08, len(st)), c="tab:blue",
                   s=40, label=f"звёзды неба ({len(st)})")
        ax.scatter(np.clip(sp, 1e-2, cap), np.full(len(sp), 0.0) + np.linspace(-0.08, 0.08, len(sp)), c="tab:red",
                   s=40, label=f"пятна итога ({len(sp)})")
        ax.set_xscale("log"); ax.set_yticks([0, 1]); ax.set_yticklabels(["пятна", "звёзды"]); ax.set_ylim(-0.6, 1.6)
        ax.set_xlabel(f"пик {f} / пик F658N, counts/s (≤ 0 у F658N — на правом краю)")
        ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left")
        gap = sp.min() / st.max() if len(sp) and len(st) else float("nan")
        ax.set_title(f"{f}: звёзды {st.min():.2g}…{st.max():.2g}, пятна {sp.min():.2g}…{np.clip(sp, None, cap).max():.2g}; "
                     f"зазор {gap:.2g}×")
        verdict.append(f"{f}: звёзды {st.min():.3g}…{st.max():.3g}, пятна {sp.min():.3g}…{sp.max():.3g}; "
                       f"зазор (мин. пятно / макс. звезда) {gap:.3g}×; пятен ниже макс. звезды × 3: {(sp < 3 * st.max()).sum()}")
    fig.suptitle("Следы лучей на небе итога против звёзд неба: пик фильтра / пик F658N в том же месте (counts/s)")
    fig.tight_layout(); fig.savefig(out / "50_следы_против_звёзд.png", dpi=80); plt.close(fig)
    text = "\n".join(verdict) + "\n\n" + "\n".join(rows) + "\n"
    (out / "50_следы_против_звёзд.tsv").write_text(text, encoding="utf-8")
    print("\n".join(verdict))
    print(f"пятен: F502N {len(spots['F502N'])}, F656N {len(spots['F656N'])}, F658N {len(spots['F658N'])}; звёзд {len(stars)}")


if __name__ == "__main__":
    main()
