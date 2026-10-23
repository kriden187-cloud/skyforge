"""Фаза 8: совмещение двух фильтров по плиткам — локальный сдвиг и наложение (листы 43–44 для i2d MAST, 46–47 для
своей сборки).

Плитки 300 px на сетке второго файла (длинные волны, 0,063″/px): первый перепроецируется на плитку, у обоих снимается
крупный масштаб (медиана 15 px), сдвиг — по пику взаимной корреляции с параболической подгонкой (доли px). Плитки с
корреляцией ≤ MIN_CORR не оцениваются (пустое небо, мало деталей).

Запуск: uv run python scripts/registration_check.py <файл SW> <файл LW> <папка> <номер листа> <подпись>
"""

from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp
from scipy import ndimage, signal

warnings.filterwarnings("ignore")
T = 300
MIN_CORR = 0.3
TEXT = {
    "ru": {"line": "{label}: плиток {n}, оценено {m}; сдвиг медиана {med:.2f} px, 95% {p95:.2f}, макс {mx:.2f} px "
                   "(0,063″/px); > 1 px: {over}",
           "legend": "зелёный ≤ 1 px, жёлтый 1–7, красный ≥ 7 px (1,2″ ≈ 19 px)", "shift": "сдвиг",
           "overlay_title": "красный — длинные волны, голубой — короткие; плитки 300 px (0,063″/px), рельеф без "
                            "крупного масштаба. Шесть худших плиток и две лучшие",
           "map": "сдвиг", "overlay": "наложение"},
    "en": {"line": "{label}: {n} tiles, {m} measured; shift median {med:.2f} px, 95% {p95:.2f}, max {mx:.2f} px "
                   "(0.063\"/px); > 1 px: {over}",
           "legend": "green ≤ 1 px, yellow 1–7, red ≥ 7 px (1.2\" ≈ 19 px); number = local shift of the short-wavelength "
                     "image relative to the long-wavelength one",
           "shift": "shift",
           "overlay_title": "red — long wavelength, cyan — short wavelength; 300 px tiles (0.063\"/px), large scale "
                            "removed. Six worst tiles and two best",
           "map": "shift", "overlay": "overlay"},
}


def _hp(a: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(a)
    return np.clip(a - ndimage.uniform_filter(a, 15), -2, 50)


def _sub(v: np.ndarray, i: int) -> float:
    """Параболическая поправка пика по трём точкам."""
    if 0 < i < len(v) - 1:
        den = v[i - 1] - 2 * v[i] + v[i + 1]
        if den < 0:
            return float(0.5 * (v[i - 1] - v[i + 1]) / den)
    return 0.0


def load(path: str) -> tuple[np.ndarray, WCS]:
    with fits.open(path) as h:
        a = np.array(h["SCI"].data, dtype=np.float32)
        if "WHT" in h:
            a[np.asarray(h["WHT"].data) <= 0] = np.nan
        return a, WCS(h["SCI"].header)


def tiles(S: np.ndarray, ws: WCS, R: np.ndarray, wr: WCS) -> np.ndarray:
    H, W = R.shape
    rows = []
    for y0 in range(0, H - T + 1, T):
        for x0 in range(0, W - T + 1, T):
            r = R[y0:y0 + T, x0:x0 + T]
            if np.isfinite(r).mean() < 0.9:
                continue
            s, _ = reproject_interp((S, ws), wr.slice((slice(y0, y0 + T), slice(x0, x0 + T))), shape_out=(T, T),
                                    order="bilinear")
            if np.isfinite(s).mean() < 0.9:
                continue
            # пустые пиксели (насыщенные ядра разного размера в двух фильтрах) с запасом — вне сравнения: иначе пик
            # корреляции ловит края дыр, а не звёзды
            holes = ndimage.binary_dilation(~np.isfinite(r) | ~np.isfinite(s), iterations=5)
            A, B = np.where(holes, 0, _hp(r)), np.where(holes, 0, _hp(s))
            cc = signal.fftconvolve(B, A[::-1, ::-1], mode="same")
            c, win = T // 2, 30
            z = cc[c - win:c + win + 1, c - win:c + win + 1]
            my, mx = np.unravel_index(np.argmax(z), z.shape)
            q = z[my, mx] / np.sqrt((A * A).sum() * (B * B).sum())
            dx = mx - win + _sub(z[my], mx)
            dy = my - win + _sub(z[:, mx], my)
            rows.append((y0 + T / 2, x0 + T / 2, dx, dy, float(q)))
    return np.array(rows)


def main() -> None:
    sw, lw, out, num, label = sys.argv[1:6]
    t = TEXT["en" if len(sys.argv) > 6 and sys.argv[6] == "en" else "ru"]
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    S, ws = load(sw)
    R, wr = load(lw)
    rows = tiles(S, ws, R, wr)
    good = rows[rows[:, 4] > MIN_CORR]
    d = np.hypot(good[:, 2], good[:, 3])
    line = t["line"].format(label=label, n=len(rows), m=len(good), med=np.median(d), p95=np.percentile(d, 95),
                            mx=d.max(), over=int((d > 1).sum()))
    print(line)
    np.save(out / f"{num}_сдвиги.npy", rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = 6
    Rs = np.nan_to_num(R[::k, ::k])
    lo, hi = np.percentile(Rs[Rs != 0], [5, 99.5])
    img = np.arcsinh(np.clip((Rs - lo) / (hi - lo), 0, None) * 10) / np.arcsinh(10)
    fig, ax = plt.subplots(figsize=(16, 16 * R.shape[0] / R.shape[1] + 1.5), dpi=110)
    ax.imshow(img, cmap="gray", origin="lower")
    for yc, xc, dx, dy, q in good:
        dd = np.hypot(dx, dy)
        col = "lime" if dd <= 1 else ("yellow" if dd < 7 else "red")
        ax.add_patch(plt.Rectangle(((xc - T / 2) / k, (yc - T / 2) / k), T / k, T / k, fill=False, ec=col, lw=1.5))
        ax.text(xc / k, yc / k, f"{dd:.1f}", color=col, ha="center", va="center", fontsize=7)
    ax.set_title(f"OMC3-N: {line}\n{t['legend']}", fontsize=10)
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(0, img.shape[0])
    fig.savefig(out / f"{num}_{t['map']}_{label}.png", bbox_inches="tight")
    plt.close(fig)
    # наложение: худшие плитки и две лучшие
    order = good[np.argsort(-np.hypot(good[:, 2], good[:, 3]))]
    pick = list(order[:6]) + list(order[-2:])
    fig, axs = plt.subplots(2, 4, figsize=(20, 10.6), dpi=100)

    def norm(a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a)
        a = a - ndimage.median_filter(a, 25)
        lo, hi = np.percentile(a, [5, 99.7])
        return np.clip((a - lo) / (hi - lo), 0, 1)

    for ax, (yc, xc, dx, dy, q) in zip(axs.ravel(), pick):
        y0, x0 = int(yc - T / 2), int(xc - T / 2)
        s, _ = reproject_interp((S, ws), wr.slice((slice(y0, y0 + T), slice(x0, x0 + T))), shape_out=(T, T),
                                order="bilinear")
        a, b = norm(R[y0:y0 + T, x0:x0 + T]), norm(s)
        ax.imshow(np.dstack([a, b, b]), origin="lower", interpolation="nearest")
        ra, dec = wr.pixel_to_world_values(xc, yc)
        ax.set_title(f"ra {float(ra):.4f} dec {float(dec):.4f}: {t['shift']} {np.hypot(dx, dy):.1f} px", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"OMC3-N, {label}: {t['overlay_title']}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / f"{int(num) + 1}_{t['overlay']}_{label}.png")
    (out / f"{num}_{t['map']}_{label}.txt").write_text(line + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
