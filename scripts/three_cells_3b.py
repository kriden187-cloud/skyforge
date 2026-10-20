"""3б, шаг 4: лист трёх клеток для поста — Heritage 1998 | наш Hubble | наш JWST showcase phase-3k на сетке Hubble
(север вверх, кадр align.box), одинаковый масштаб и кадр; второй ряд — те же три ×8 (небо). Heritage приводится к сетке
по AVM, JWST — по WCS своей совмещённой сетки; затем каждая подгоняется к Hubble подобием (сдвиг, поворот, масштаб)
по сдвигам кусков вокруг опорных точек: JWST — звёзды поля и центральная звезда; Heritage — они же и узлы туманности
(те же снимки WFPC2). Невязки — в подписи и в .txt.

Запуск: uv run python scripts/three_cells_3b.py <png Hubble showcase> <папка> <имя> [bin]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from reproject import reproject_interp
from scipy import ndimage
from skimage.transform import SimilarityTransform, warp

from skyforge.align import load_aligned, with_resolved_crops
from skyforge.clean import clean_dir, load_nebula_sd
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from clean_report_3b import reference_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def load_grid(path: pathlib.Path) -> np.ndarray:
    """PNG → RGB 0–1 в координатах сетки (строка 0 — низ); прозрачное — чёрное."""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(np.float32) / 255
    rgb = im[..., :3] * im[..., 3:4]
    return rgb[::-1].copy()


def star_points(cfg: Config, P: Paths, wcs, shape) -> list[tuple[int, int]]:
    """Опорные звёзды: звёзды неба из маски clean (вес > 0,5, вне маски туманности) и центральная звезда (кроп center).
    Узлы туманности не берутся: в ИК NIRCam и в узких линиях WFPC2 это разные структуры."""
    w = np.load(clean_dir(cfg, P) / "star_weight.npy")
    sd = load_nebula_sd(cfg, P)
    lab, n = ndimage.label((w > 0.5) & (sd < 0), structure=np.ones((3, 3)))
    pts = [tuple(int(v) for v in p) for p in ndimage.center_of_mass(w > 0.5, lab, np.arange(1, n + 1))]
    c = next(c for c in cfg.report.crops if c.name == "center")
    x, y = wcs.world_to_pixel_values(c.ra, c.dec)
    pts.append((int(round(float(y))), int(round(float(x)))))
    return [(y, x) for y, x in pts if 20 <= y < shape[0] - 20 and 20 <= x < shape[1] - 20]


def nebula_points(cfg: Config, P: Paths, step: int) -> list[tuple[int, int]]:
    """Для Heritage: он собран из тех же снимков WFPC2 (программа 6221), узлы туманности в нём те же, что у нас, —
    опорные точки сеткой по маске туманности (звёзды поля на его чёрном небе почти не видны)."""
    sd = load_nebula_sd(cfg, P)
    H, W = sd.shape
    return [(y, x) for y in range(step, H - step, step) for x in range(step, W - step, step) if sd[y, x] > 10]


def _hp(a: np.ndarray) -> np.ndarray:
    return a - ndimage.median_filter(a, 9)


def match(lo: np.ndarray, lt: np.ndarray, y: int, x: int, half: int, search: int) -> tuple[float, float, float] | None:
    """Сдвиг куска other к куску ours вокруг звезды: нормированная взаимная корреляция по сдвигам ±search,
    уточнение параболой. (dy, dx, корреляция в пике)."""
    a = _hp(lo[y - half - 4:y + half + 5, x - half - 4:x + half + 5])[4:-4, 4:-4]
    a = (a - a.mean()) / (a.std() + 1e-9)
    big = _hp(lt[y - half - search - 4:y + half + search + 5, x - half - search - 4:x + half + search + 5])[4:-4, 4:-4]
    if big.shape != (2 * (half + search) + 1,) * 2:
        return None
    cc = np.full((2 * search + 1, 2 * search + 1), -1.0)
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            b = big[search + dy:search + dy + 2 * half + 1, search + dx:search + dx + 2 * half + 1]
            b = (b - b.mean()) / (b.std() + 1e-9)
            cc[dy + search, dx + search] = (a * b).mean()
    j, i = np.unravel_index(np.argmax(cc), cc.shape)
    if not (0 < j < cc.shape[0] - 1 and 0 < i < cc.shape[1] - 1):
        return None
    def par(m1, c0, p1):
        d = m1 - 2 * c0 + p1
        return 0.0 if d == 0 else 0.5 * (m1 - p1) / d
    sy = par(cc[j - 1, i], cc[j, i], cc[j + 1, i])
    sx = par(cc[j, i - 1], cc[j, i], cc[j, i + 1])
    return j - search + sy, i - search + sx, float(cc[j, i])


def fit_to(ours: np.ndarray, other: np.ndarray, pts: list, search: int, half: int = 10,
           min_cc: float = 0.5) -> tuple[np.ndarray, str]:
    """Подобие other → ours по сдвигам кусков вокруг опорных звёзд; other уже на сетке (строка 0 — низ)."""
    lo, lt = ours @ LUMA, other @ LUMA
    src, dst, used = [], [], []
    for y, x in pts:
        m = match(lo, lt, y, x, half, search)
        if m is None or m[2] < min_cc:
            continue
        src.append((x + m[1], y + m[0]))
        dst.append((x, y))
        used.append((y, x, m))
    src, dst = np.array(src, float), np.array(dst, float)
    if len(src) == 0:
        return other, "по звёздам не подогнано: совпадений нет"
    if len(src) < 3:                                       # мало звёзд — только сдвиг
        t = SimilarityTransform(translation=np.median(dst - src, axis=0))
    else:
        t = SimilarityTransform()
        for _ in range(3):
            t.estimate(src, dst)
            res = np.hypot(*(t(src) - dst).T)
            ok = res < max(1.0, 3 * np.median(res))
            if ok.all():
                break
            src, dst = src[ok], dst[ok]
        t.estimate(src, dst)
    res = np.hypot(*(t(src) - dst).T)
    out = warp(other, t.inverse, output_shape=ours.shape[:2], order=1)
    info = (f"опорных точек {len(src)} из {len(pts)}; сдвиг {t.translation[0]:+.1f}, {t.translation[1]:+.1f} px, поворот "
            f"{np.degrees(t.rotation):+.2f}°, масштаб {t.scale:.4f}; невязка медиана {np.median(res):.2f}, макс {res.max():.2f} px")
    return out.astype(np.float32), info


def main() -> None:
    hub_png, out, name = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
    bin_ = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    P = Paths(pathlib.Path(".").resolve())
    hc = Config.load("configs/ngc3132hst/showcase.yaml", "showcase").model_copy(update={"bin": bin_})
    harr, hwcs, _ = load_aligned(hc, P)
    shape = next(iter(harr.values())).shape
    hub = load_grid(hub_png)
    assert hub.shape[:2] == shape, (hub.shape, shape)
    her, _ = reference_on_grid(str(P.abs(hc.reference_image)), hwcs, shape)
    jc = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    _, jwcs, _ = load_aligned(jc, P)
    jw = load_grid(P.root / "data" / "baselines" / "ngc3132" / "phase-3k" / "showcase_bin1.png")
    jwst = np.stack([np.nan_to_num(reproject_interp((jw[..., k], jwcs), hwcs, shape_out=shape, order="bilinear")[0])
                     for k in range(3)], -1).astype(np.float32)
    pts = star_points(with_resolved_crops(hc, P), P, hwcs, shape)
    s = 0.0455 * bin_
    neb_pts = nebula_points(hc, P, max(20, int(round(3.6 / s))))
    her, i_her = fit_to(hub, her, pts + neb_pts, int(round(1.5 / s)), half=int(round(1.8 / s)), min_cc=0.6)   # ±1,5″
    jwst, i_jw = fit_to(hub, jwst, pts, int(round(0.8 / s)))
    cols = [("Hubble Heritage 1998 (STScI/AURA/NASA/ESA)", her), ("наш Hubble (WFPC2, 1995)", hub),
            ("наш JWST showcase phase-3k (NIRCam, 2022)", jwst)]
    tiles, labels = [], []
    for lab, im in cols:
        tiles.append(shrink(im, 1100)); labels.append(lab)
    for lab, im in cols:
        tiles.append(np.clip(shrink(im, 1100) * 8, 0, 1)); labels.append(f"{lab} · ×8")
    contact_sheet(tiles, labels, out / f"{name}.png", cols=3, tile_inches=6.0, interpolation="lanczos",
                  title=f"NGC 3132: один кадр и масштаб (сетка Hubble, север вверх, {s:.4f}″/px)")
    text = f"Heritage → Hubble: {i_her}\nJWST phase-3k → Hubble: {i_jw}\nопорных звёзд Hubble: {len(pts)}; точек по туманности для Heritage: {len(neb_pts)}\n"
    (out / f"{name}.txt").write_text(text, encoding="utf-8")
    print(text)
    for lab, im in (("heritage", her), ("hubble", hub), ("jwst", jwst)):     # клетки по отдельности, для поста
        Image.fromarray((np.clip(im[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)).save(out / f"{name}_{lab}.png")


if __name__ == "__main__":
    main()
