"""3б, шаг 2: отчёт после clean — кадр-рамка, космические лучи, швы чипов, звёзды против референса.

Кропы (ra/dec из конфига): center, edge при 100%; sky, sky2 ×5; gap ×3 — до (совмещённые кадры) и после
(clean + заполнение швов showcase). Цвет — быстрый, по палитре референса (F658N R, F656N G, F502N B), одни и те же
точки до и после. Референс Heritage приводится к нашей сетке по его AVM (масштаб, центр, поворот). Числа: удалённые
следы, заполненные пиксели, поле статистики, сдвиги фильтров по звёздам вне туманности.

Запуск: uv run python scripts/clean_report_3b.py <папка>
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
from astropy.wcs import WCS
from PIL import Image
from reproject import reproject_interp
from scipy import ndimage

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.cosmics import fill_gaps_linear
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

ORDER = ("F658N", "F656N", "F502N")  # R, G, B — палитра референса


def color(arrs: dict, lims: dict) -> np.ndarray:
    ch = []
    for f in ORDER:
        lo, hi = lims[f]
        x = np.clip((np.nan_to_num(arrs[f], nan=lo) - lo) / (hi - lo), 0, 1)
        ch.append(np.arcsinh(x / 0.05) / np.arcsinh(1 / 0.05))
    return np.stack(ch, -1).astype(np.float32)


def reference_on_grid(path: str, wcs: WCS, shape) -> np.ndarray:
    """Heritage 935×700 → наша сетка по AVM: центр 468,5;351 → (151,757259; −40,4350252), 0,1″/px, поворот 109°.
    Знак шкалы x выбран так, чтобы центральная звезда попала в пиксель (518, 330) референса (сверху)."""
    img = np.asarray(Image.open(path).convert("RGB")).astype(np.float32)[::-1] / 255   # строка 0 — низ, как FITS
    best = None
    for sx in (-1, 1):
        w = WCS(naxis=2)
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crpix = [468.5, 351.0]
        w.wcs.crval = [151.757259, -40.4350252]
        w.wcs.cdelt = [sx * 2.78304294934e-05, 2.77026382011e-05]
        w.wcs.crota = [109.0, 109.0]
        x, y = w.world_to_pixel_values(151.757305, -40.43642)
        err = math.hypot(float(x) - 517, float(y) - (699 - 330))
        if best is None or err < best[0]:
            best = (err, w)
    err, w = best
    out = np.stack([reproject_interp((img[..., k], w), wcs, shape_out=shape, order="bilinear")[0] for k in range(3)], -1)
    return np.nan_to_num(out).astype(np.float32), err


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132hst/showcase.yaml", "showcase")
    raw, wcs, meta = load_aligned(cfg, P)
    clean, _ = load_input(cfg, P)
    fill = {f: fill_gaps_linear(a, cfg.clean.fill_gaps_px) for f, a in clean.items()}
    after = {f: v[0] for f, v in fill.items()}
    filled_any = np.logical_or.reduce([v[1] for v in fill.values()])
    sky, sd = load_sky_mask(cfg, P), load_nebula_sd(cfg, P)
    lims = {f: tuple(float(v) for v in np.nanpercentile(clean[f][sky], [5.0])) + (float(np.nanpercentile(clean[f][sd > 0], 99.5)),)
            for f in clean}
    before_c, after_c = color(raw, lims), color(after, lims)
    ref, ref_err = reference_on_grid(str(P.abs(cfg.reference_image)), wcs, next(iter(raw.values())).shape)
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    H, W = before_c.shape[:2]

    def cut(img, c):
        """img — строка 0 сверху; c.x, c.y — на итоговой картинке при bin 1 (y вниз)."""
        s = max(16, c.size // cfg.bin)
        cx, cy = int(round(c.x / cfg.bin)), int(round(c.y / cfg.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    tiles, labels = [], []
    for name, k in (("center", 1), ("edge", 1), ("sky", 5), ("sky2", 5), ("gap", 3)):
        c = crops[name]
        for lab, im in (("до", before_c), ("после", after_c), ("Heritage 1998", ref)):
            t = cut(im[::-1], c)                          # кропы — в координатах картинки (строка 0 — верх)
            if name.startswith("sky"):
                t = np.clip(t * k, 0, 1)                  # небо ×5 — усиление яркости
            elif name == "gap":
                t = np.kron(t, np.ones((k, k, 1)))         # шов ×3 — увеличение
            tiles.append(t[::-1])                         # contact_sheet рисует строку 0 снизу
            labels.append(f"{name} · {lab}" + (f" · ×{k}" if k > 1 else " · 100%"))
    contact_sheet(tiles, labels, out / "02_clean_до_после.png", cols=3, tile_inches=4.0, interpolation="nearest",
                  title="3б clean: лучи и швы — до, после, референс Heritage (на нашей сетке по AVM)")
    ov = shrink(after_c, 900).copy()
    fm = shrink(filled_any.astype(np.float32), 900) > 0
    ov[fm] = [1.0, 0.1, 0.1]
    contact_sheet([ov[::-1], shrink(before_c, 900)[::-1]], ["после + заполненные швы (красные)", "до"],
                  out / "03_швы_заполненные.png", cols=2, tile_inches=7, interpolation="nearest",
                  title=f"Швы чипов и чёрточки: заполнено {int(filled_any.sum())} px (полуширина ≤ {cfg.clean.fill_gaps_px} px)")
    rows = [f"референс на сетке: невязка центральной звезды {ref_err:.1f} px референса"]
    rows.append("удалено следов лучей, px (с кольцом 1 px): " + ", ".join(
        f"{f} {int(np.isfinite(raw[f]).sum() and (np.abs(np.nan_to_num(raw[f]) - np.nan_to_num(clean[f])) > 0).sum())}" for f in raw))
    rows.append("заполнено швов и чёрточек, px: " + ", ".join(f"{f} {int(v[1].sum())}" for f, v in fill.items()))
    field = stats_field(cfg, P, list(clean.values()))
    full = field.full
    neb = sd > 0
    rows.append(f"поле статистики: прямоугольник {field.rows.start}:{field.rows.stop} × {field.cols.start}:{field.cols.stop} "
                f"из {H}×{W}; все фильтры в {full.mean() * 100:.1f}% кадра; туманности в поле {field.mask(neb).sum() / neb.sum() * 100:.1f}%, "
                f"неба в поле {field.mask(sky).sum() / sky.sum() * 100:.1f}%")
    # сдвиги фильтров по звёздам вне туманности (после очистки лучей)
    exc = {f: ndimage.gaussian_filter(np.nan_to_num(a, nan=float(np.nanmedian(a))), 1.0) -
           ndimage.median_filter(np.nan_to_num(a, nan=float(np.nanmedian(a))), 25) for f, a in clean.items()}
    sig = {f: 1.4826 * np.median(np.abs(e[::5, ::5] - np.median(e[::5, ::5]))) for f, e in exc.items()}
    e0 = exc["F656N"]
    far = ndimage.distance_transform_edt(~neb) > 30
    pk = (e0 == ndimage.maximum_filter(e0, 21)) & (e0 > 20 * sig["F656N"]) & far
    shifts = {f: [] for f in ("F502N", "F658N")}
    for y, x in zip(*np.nonzero(pk)):
        if not all(exc[f][y - 2:y + 3, x - 2:x + 3].max() > 10 * sig[f] for f in exc):
            continue
        cs = {}
        for f in exc:
            win = exc[f][y - 3:y + 4, x - 3:x + 4].copy()
            if win.shape != (7, 7):
                break
            win[win < 0] = 0
            yy, xx = np.mgrid[-3:4, -3:4]
            cs[f] = ((yy * win).sum() / win.sum(), (xx * win).sum() / win.sum())
        if len(cs) == 3:
            for f in shifts:
                shifts[f].append((cs[f][0] - cs["F656N"][0], cs[f][1] - cs["F656N"][1]))
    for f, v in shifts.items():
        v = np.array(v)
        if v.size:
            rows.append(f"{f} к F656N по {len(v)} звёздам вне туманности: dy {np.median(v[:, 0]):+.3f}, dx {np.median(v[:, 1]):+.3f} px "
                        f"(bin {cfg.bin}); разброс {1.4826 * np.median(np.abs(v[:, 0] - np.median(v[:, 0]))):.3f} / "
                        f"{1.4826 * np.median(np.abs(v[:, 1] - np.median(v[:, 1]))):.3f}")
    text = "\n".join(rows) + "\n"
    (out / "02_clean_числа.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
