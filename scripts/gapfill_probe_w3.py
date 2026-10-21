"""Фаза 6, W3-2: проба заполнения щелей F182M/F210M (color.fill_gaps) — лист 02 и числа.

Два кропа: «gap» из конфига (перекрестье щелей, правый модуль) и такой же по размеру в левом модуле — в месте, где
заполненного больше всего (щели через тёмные прожилки). Колонки: без заполнения 100%, с заполнением 100%, без ×3,
с ×3, карта заполненного (белое — заполнено).
Числа: разность итогов вне заполненного (уровни 0–255): доля пикселей с разностью, максимум — у щели и дальше
радиуса local_contrast (80 px сетки при bin 1 → /bin); заметность полосы — средний цвет в щели против полос той же
ширины по обе стороны (по каждой вертикальной и горизонтальной полосе в кропе), до и после, без пикселей звёзд
(маска clean с отступом 3 px: лучи яркой звезды в полосе дают синий сдвиг сами по себе); для сравнения — тот же контраст
у случайных полос той же ширины вне щелей (естественный разброс туманности).

Запуск: uv run python scripts/gapfill_probe_w3.py <png без заполнения> <png с заполнением> <папка>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import aligned_dir, with_resolved_crops
from skyforge.clean import clean_dir
from skyforge.compose import composite_path
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop


def bands(filled: np.ndarray) -> list[tuple[str, slice]]:
    """Полосы щелей в кропе: столбцы и строки, где заполнено > 60%."""
    out = []
    for axis, name in ((0, "столбцы"), (1, "строки")):
        frac = filled.mean(axis)
        idx = np.flatnonzero(frac > 0.6)
        if idx.size == 0:
            continue
        runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
        out += [(name, slice(int(r[0]), int(r[-1]) + 1)) for r in runs]
    return out


def band_contrast(img: np.ndarray, name: str, sl: slice, keep: np.ndarray) -> np.ndarray:
    """Средний RGB в полосе минус среднее двух соседних полос той же ширины (уровни), по пикселям keep."""
    w = sl.stop - sl.start
    lo, hi = slice(max(0, sl.start - w), sl.start), slice(sl.stop, sl.stop + w)
    take = (lambda a, s: a[:, s]) if name == "столбцы" else (lambda a, s: a[s, :])
    inner = take(img, sl)[take(keep, sl)].mean(0)
    side = np.concatenate([take(img, lo)[take(keep, lo)], take(img, hi)[take(keep, hi)]]).mean(0)
    return (inner - side) * 255


def main() -> None:
    before, after, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/w3_2/showcase.yaml", "showcase")
    A = np.asarray(Image.open(before).convert("RGB"), dtype=np.float32)[::-1] / 255   # сетка: строка 0 — низ
    B = np.asarray(Image.open(after).convert("RGB"), dtype=np.float32)[::-1] / 255
    filled = np.load(composite_path(c, P).with_name("filled.npy"))
    nostar = ~ndimage.binary_dilation(np.load(clean_dir(c, P) / "star_weight.npy") > 0.01, iterations=3)
    data = np.load(aligned_dir(c, P) / "coverage.npy") > 0
    H, W = filled.shape
    k = next(x for x in with_resolved_crops(c, P).report.crops if x.name == "gap")
    sl_gap = display_crop((H, W), k.x, k.y, k.size, c.bin)
    s = sl_gap[0].stop - sl_gap[0].start
    left = filled.copy()
    left[:, W // 2:] = False
    dens = ndimage.uniform_filter(left.astype(np.float32), s)
    cy, cx = np.unravel_index(np.argmax(dens), dens.shape)
    r0, c0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
    sl_left = (slice(r0, r0 + s), slice(c0, c0 + s))
    diff = np.abs(B - A).max(-1) * 255
    lc = c.finish.local_contrast.radius / c.bin
    far = ndimage.distance_transform_edt(~filled) > lc
    valid = np.isfinite(A).all(-1) & (A.sum(-1) > 0)
    rows = ["мера\tзначение",
            f"заполнено, % кадра\t{filled.mean() * 100:.2f}",
            f"вне заполненного: пикселей с разностью ≥ 1 уровня, %\t{((diff >= 1) & ~filled & valid).mean() * 100:.3f}",
            f"вне заполненного: максимум разности, уровней\t{diff[~filled & valid].max():.0f}",
            f"дальше {lc:.0f} px от щели (радиус local_contrast): максимум разности, уровней\t{diff[far & valid].max():.0f}"]
    tiles, labels = [], []
    for name, sl in (("gap (правый модуль)", sl_gap), ("левый модуль", sl_left)):
        fa, fb, fm = A[sl], B[sl], filled[sl]
        for lab, t in (("без заполнения · 100%", fa), ("с заполнением · 100%", fb),
                       ("без · ×3", np.clip(fa * 3, 0, 1)), ("с · ×3", np.clip(fb * 3, 0, 1))):
            tiles.append(t[::-1]); labels.append(f"{name} · {lab}")
        tiles.append(np.repeat(fm[::-1, :, None].astype(np.float32), 3, axis=2)); labels.append(f"{name} · заполнено")
        for bname, bsl in bands(fm):
            cb, ca = band_contrast(fa, bname, bsl, nostar[sl]), band_contrast(fb, bname, bsl, nostar[sl])
            rows.append(f"{name}: полоса ({bname} {bsl.start}–{bsl.stop}) минус соседи R/G/B, уровни: без / с\t"
                        f"{' / '.join(f'{v:+.1f}' for v in cb)}  →  {' / '.join(f'{v:+.1f}' for v in ca)}")
    rng, nat, w = np.random.default_rng(0), [], 17
    for _ in range(200000):
        if len(nat) >= 300:
            break
        y0, x0 = int(rng.integers(0, H - 150)), int(rng.integers(w, W - 2 * w))
        reg = (slice(y0, y0 + 150), slice(x0 - w, x0 + 2 * w))
        if filled[reg].any() or not data[reg].all():
            continue
        nat.append(np.abs(band_contrast(B[reg], "столбцы", slice(w, 2 * w), nostar[reg])))
    if nat:
        q = np.percentile(np.array(nat), [50, 90, 95], axis=0)
        rows.append(f"естественный разброс: |контраст| случайных полос {w}×150 px вне щелей ({len(nat)} шт.) "
                    f"R/G/B p50 / p90 / p95\t" + "  ·  ".join(" / ".join(f"{v:.1f}" for v in r) for r in q))
    contact_sheet(tiles, labels, out / "02_заполнение_щелей.png", cols=5, tile_inches=3.6, interpolation="nearest",
                  title=f"W3-2: щели F182M/F210M — без и с заполнением (bin {c.bin}; 100% и ×3; белое — заполнено)")
    (out / "02_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
