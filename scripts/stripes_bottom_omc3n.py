"""OMC3-N: вертикальные линии у нижнего края и полосы 1/f — лист 03 (проверка владельца и консультанта 2026-10-22).

Верх листа: нижние 400 строк финала при 100% и ×3 и те же строки растянутых слоёв F150W, F200W, F356W, F480M.
Середина: профиль по столбцам итога (медиана по нижним 400 строкам после вычитания сглаживания вдоль строки) с пиками
и их разбором — что на этом столбце ниже края кадра (звезда в мозаике i2d за краем прямоугольника), в каких фильтрах
линия. Низ: горизонтальные полосы 1/f (строки детектора в этом кадре горизонтальны, угол по WCS _cal) — тёмные кропы
итога при 100% и ×3 и мера: σ медиан строк / шум пикселя (как в листе 34).

Запуск: uv run python scripts/stripes_bottom_omc3n.py <png итога> <папка> [угол строк детектора, °]
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402

from skyforge.align import with_resolved_crops  # noqa: E402
from skyforge.config import Config  # noqa: E402
from skyforge.paths import Paths  # noqa: E402
from skyforge.preview import display_crop  # noqa: E402
from skyforge.stretch import stretched_dir  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
RAW = {"F150W": "jw07433-o001_t001_nircam_clear-f150w", "F356W": "jw07433-o001_t001_nircam_clear-f356w"}


def col_profile(a: np.ndarray) -> np.ndarray:
    hp = a - ndimage.uniform_filter1d(a, 31, axis=1)
    p = np.median(hp, axis=0)
    return p / (1.4826 * np.median(np.abs(p - np.median(p))) + 1e-12)


def below_edge_star(grid: WCS, f: str, c: int) -> tuple[float, int]:
    """Самый яркий пиксель i2d фильтра в столбцах c±12 на 600 строк ниже нижнего края кадра: (значение, строк ниже)."""
    path = pathlib.Path("data/raw/jwst") / RAW[f] / f"{RAW[f]}_i2d.fits"
    with fits.open(path, memmap=True) as h:
        sci, w = h["SCI"].data, WCS(h["SCI"].header)
        ks = np.arange(-600, 0)
        best, at = -np.inf, 0
        for dc in range(-12, 13):
            ra, dec = grid.pixel_to_world_values(np.full(ks.size, float(c + dc)), ks.astype(float))
            x, y = w.world_to_pixel_values(ra, dec)
            xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
            ok = (xi >= 0) & (yi >= 0) & (xi < sci.shape[1]) & (yi < sci.shape[0])
            v = np.full(ks.size, np.nan)
            v[ok] = sci[yi[ok], xi[ok]]
            if np.isfinite(v).any() and np.nanmax(v) > best:
                best, at = float(np.nanmax(v)), int(-ks[np.nanargmax(v)])
    return best, at


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    angle = float(sys.argv[3]) if len(sys.argv) > 3 else float("nan")
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/omc3n/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    meta = json.loads((P.work("omc3n", 1) / "aligned" / "meta.json").read_text(encoding="utf-8"))
    grid = WCS(fits.Header.fromstring(meta["wcs"], sep="\n"))
    img = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255          # строка 0 — верх
    H, W = img.shape[:2]
    band = img[H - 400:]
    z = col_profile(band.mean(-1) * 255)
    peaks = np.nonzero(z > 6)[0]
    groups = np.split(peaks, np.nonzero(np.diff(peaks) > 3)[0] + 1) if peaks.size else []
    sd = stretched_dir(c, P)
    layers = {f: np.flipud(np.nan_to_num(np.asarray(np.load(sd / f"{f}.npy", mmap_mode="r")[0:400])))
              for f in c.filters}                                                    # нижние 400 строк, верх — вверху
    rows = ["столбцы\tz итога\tz F150W / F200W / F356W / F480M\tF150W ниже края: max (строк ниже)\t"
            "F356W ниже края: max (строк ниже)\tразбор"]
    marks = []
    for g in groups:
        c0, c1 = int(g[0]), int(g[-1]) + 1
        zl = {f: float(col_profile(L)[c0:c1].max()) for f, L in layers.items()}
        s150, d150 = below_edge_star(grid, "F150W", (c0 + c1) // 2)
        s356, d356 = below_edge_star(grid, "F356W", (c0 + c1) // 2)
        hp = band.mean(-1) - ndimage.uniform_filter1d(band.mean(-1), 31, axis=1)
        prof = hp[:, c0:c1].mean(1)
        lit = np.nonzero(prof > 0.5 * prof.max())[0]
        if max(s150, s356) > 50:
            why = f"луч звезды за нижним краем ({d150 if s150 >= s356 else d356} строк ниже)"
        elif lit.max() - lit.min() < 40:
            why = "компактный источник у края (звезда, узел), не линия через кадр"
        elif zl["F150W"] + zl["F200W"] > 2 * (zl["F356W"] + zl["F480M"]):
            why = "вертикальная линия коротких фильтров без звезды за краем — разобрать отдельно"
        else:
            why = "линия длинных фильтров без звезды за краем — разобрать отдельно"
        rows.append(f"{c0}–{c1 - 1}\t{z[c0:c1].max():.1f}\t" + " / ".join(f"{zl[f]:.1f}" for f in c.filters)
                    + f"\t{s150:.1f} ({d150})\t{s356:.1f} ({d356})\t{why}")
        marks.append(((c0 + c1) / 2, why))
    # горизонтальные полосы: тёмные кропы итога, σ медиан строк / шум
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    hstats = []
    for name in ("sky", "sky2", "filament"):
        k = crops[name]
        sl = display_crop((H, W), k.x, k.y, k.size, 1)
        lum = img[::-1][sl].mean(-1) * 255
        rmed = np.median(lum - ndimage.uniform_filter1d(lum, 61, axis=0), axis=1)
        noise = 1.4826 * np.median(np.abs(np.diff(lum, axis=1))) / np.sqrt(2)
        hstats.append((name, float(np.std(rmed) / noise * np.sqrt(lum.shape[1])), float(np.std(rmed)), float(noise)))
        rows.append(f"горизонтальные полосы итога в {name}\tσ медиан строк {np.std(rmed):.2f} уровня, шум пикселя "
                    f"{noise:.2f}, σ·√N/шум {hstats[-1][1]:.1f}\t\t\t\t—")
    rows.append(f"угол строк детектора в кадре (WCS _cal)\t{angle:.1f}°\t\t\t\tполосы 1/f — вдоль строк детектора")
    (out / "03_полосы_при_100.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))

    fig = plt.figure(figsize=(16, 22), facecolor="#111")
    gs = fig.add_gridspec(9, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1.4, 1.1, 1.1])
    x0 = max(int(marks[0][0]) - 300, 0) if marks else 0
    tiles = [("итог 100% (нижние 400 строк)", band), ("итог ×3", np.clip(band * 3, 0, 1))]
    for f, L in layers.items():
        lo, hi = np.percentile(L, [1, 99])
        tiles.append((f"{f} после растяжки (1–99%)", np.clip((L - lo) / max(hi - lo, 1e-6), 0, 1)))
    for i, (lab, t) in enumerate(tiles):
        ax = fig.add_subplot(gs[i, :])
        ax.imshow(t, cmap="gray" if t.ndim == 2 else None, aspect="auto", interpolation="nearest")
        for x, _ in marks:
            ax.axvline(x, color="#ff4040", lw=0.6, alpha=0.6)
        ax.set_title(lab, color="w", fontsize=11)
        ax.axis("off")
    ax = fig.add_subplot(gs[6, :])
    ax.plot(z, color="#9ad", lw=0.7)
    for x, why in marks:
        ax.axvline(x, color="#ff4040", lw=0.6)
        ax.text(x, z.max() * 0.95, why.split("(")[0][:22], rotation=90, color="w", fontsize=7, va="top")
    ax.set_title("профиль по столбцам итога, нижние 400 строк (z: медиана / MAD)", color="w")
    ax.set_facecolor("#111")
    ax.tick_params(colors="w")
    for i, name in enumerate(("sky", "sky2")):
        k = crops[name]
        sl = display_crop((H, W), k.x, k.y, k.size, 1)
        t = img[::-1][sl][::-1]
        ax = fig.add_subplot(gs[7, i])
        ax.imshow(t, interpolation="nearest")
        ax.set_title(f"{name} · 100% · σ·√N/шум {hstats[i][1]:.1f}", color="w")
        ax.axis("off")
        ax = fig.add_subplot(gs[8, i])
        ax.imshow(np.clip(t * 3, 0, 1), interpolation="nearest")
        ax.set_title(f"{name} · ×3", color="w")
        ax.axis("off")
    fig.suptitle(f"OMC3-N финал: вертикальные линии у нижнего края и полосы 1/f (строки детектора в кадре: {angle:.1f}°)",
                 color="w", fontsize=14)
    fig.savefig(out / "03_полосы_при_100.png", dpi=90, facecolor="#111")
    print(out / "03_полосы_при_100.png")


if __name__ == "__main__":
    main()
