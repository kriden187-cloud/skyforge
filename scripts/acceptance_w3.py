"""Фаза 6, W3-2: листы прогона по итоговому PNG showcase (bin из конфига или --bin).

30 — целиком при обычной яркости и ×5.
31 — восемь кропов из конфига при 100%.
32 — sky, sky2, gap при ×5 и рядом карта областей CON F444W и F182M (границы дизеринга).
33 — числа: опорное небо R/G/B, нули неба, выжжено в туманности всего и по модулям, зелень звёзд, прозрачные пиксели.
34 — полосы 1/f: растянутые слои F182M и F210M и итог при 100% в кропах sky, sky2 и cloud (как F182M у Sh2-305:
     видны ли полосы при 100% после растяжки); мера — σ медиан строк и столбцов / шум (scripts/stripes_s305.py).

Запуск: uv run python scripts/acceptance_w3.py <png итога> <папка> [bin]
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import aligned_dir, load_aligned, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.compose import LUMA
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop, shrink
from skyforge.stretch import reference_sky, stretched_dir

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from stripes_s305 import stripes  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
SCALE = 0.0629


def main() -> None:
    png, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load("configs/w3_2/showcase.yaml", "showcase")
    if len(sys.argv) > 3:
        c = c.model_copy(update={"bin": int(sys.argv[3])})
    b = c.bin
    arrays, _ = load_input(c, P)
    load_aligned(c, P)
    sd, sky = load_nebula_sd(c, P), load_sky_mask(c, P)
    H, W = sd.shape
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    img = rgba[..., :3].astype(np.float32)[::-1] / 255          # сетка: строка 0 — низ
    assert img.shape[:2] == (H, W), (img.shape, (H, W))
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    meta = json.loads((aligned_dir(c, P) / "meta.json").read_text(encoding="utf-8"))
    regions = meta.get("regions") or [[0, H, 0, W]]

    def cut(name: str) -> tuple[slice, slice]:
        k = crops[name]
        return display_crop((H, W), k.x, k.y, k.size, b)

    contact_sheet([shrink(img, 1600)[::-1], np.clip(shrink(img, 1600) * 5, 0, 1)[::-1]], ["целиком", "целиком ×5"],
                  out / "30_целиком.png", cols=1, tile_inches=12, interpolation="lanczos",
                  title=f"W3-2 showcase bin {b}: целиком и ×5")
    tiles, labels = [], []
    for name, k in crops.items():
        tiles.append(img[cut(name)][::-1]); labels.append(f"{name} · 100% · {k.size * SCALE:.0f}″")
    contact_sheet(tiles, labels, out / "31_кропы_100.png", cols=4, tile_inches=4.2, interpolation="nearest",
                  title=f"W3-2 showcase bin {b}: восемь кропов при 100%")
    tl = {f: np.load(aligned_dir(c, P) / f"{f}.tiles.npy") for f in ("F444W", "F182M")}
    pal = np.random.default_rng(0).uniform(0.15, 1, (4096, 3)).astype(np.float32)
    tiles, labels = [], []
    for name in ("sky", "sky2", "gap"):
        sl = cut(name)
        tiles.append(np.clip(img[sl] * 5, 0, 1)[::-1]); labels.append(f"{name} · ×5")
        for f, t_all in tl.items():
            t = t_all[sl]
            m = pal[np.where(t >= 0, t, -1).astype(int) % 4096]
            m[t < 0] = 0
            tiles.append(m[::-1]); labels.append(f"{name} · области CON {f}")
    contact_sheet(tiles, labels, out / "32_небо_щели_CON.png", cols=3, tile_inches=4.2, interpolation="nearest",
                  title=f"W3-2 showcase bin {b}: небо и щели ×5 с картами CON (F444W, F182M)")

    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    ref, info = reference_sky(c, P, sd.shape)
    rk = img[ref & full] * 255
    S = sky & full
    sk = img[S] * 255
    sw = np.load(clean_dir(c, P) / "star_weight.npy")
    st = img[(sw > 0.5) & (sd < 0)] * 255
    gx = st[:, 1] - np.maximum(st[:, 0], st[:, 2])
    green = [float(np.percentile(gx, q)) for q in (50, 75, 95)] if st.size else [float("nan")] * 3
    neb = sd > 0

    def burned(m: np.ndarray) -> float:
        v = img[m & neb]
        return float((v.max(-1) >= 254 / 255).mean() * 100) if v.size else float("nan")

    whole = np.zeros((H, W), bool)
    per = []
    for i, (r0, r1, c0, c1) in enumerate(regions):
        m = np.zeros((H, W), bool)
        m[r0:r1, c0:c1] = True
        whole |= m
        per.append(f"{'левый' if c0 < W // 2 else 'правый'} модуль {burned(m):.2f}")
    rows = ["критерий\tзначение\tпорог",
            f"опорное небо ({c.stretch.sky_reference}) R / G / B, медиана\t"
            f"{' / '.join(f'{np.median(rk[:, k]):.0f}' for k in range(3))}\t{c.stretch.sky_level:g} (stretch.sky_level)",
            f"маска неба clean R / G / B (медиана, уровни)\t{' / '.join(f'{np.median(sk[:, k]):.0f}' for k in range(3))}\t—",
            f"нулей на опорном небе R / G / B, %\t{' / '.join(f'{(rk[:, k] <= 0).mean() * 100:.2f}' for k in range(3))}\t< 0,5",
            f"нулей в маске неба R / G / B, %\t{' / '.join(f'{(sk[:, k] <= 0).mean() * 100:.2f}' for k in range(3))}\t< 0,5",
            f"выжжено в туманности (канал ≥ 254), %: всего / по модулям\t{burned(whole):.2f} · {' · '.join(per)}\t≤ 0,5",
            f"зелень звёзд неба: G − max(R, B), p50 / p75 / p95, уровни\t{' / '.join(f'{g:+.0f}' for g in green)}\tp50 ≤ 0",
            f"прозрачных пикселей (нет данных)\t{int((rgba[..., 3] == 0).sum())} ({(rgba[..., 3] == 0).mean() * 100:.1f}%)\t—",
            f"опорное небо: {json.dumps(info, ensure_ascii=False)}\t\t"]

    sdir = stretched_dir(c, P)
    layers = {f: np.load(sdir / f"{f}.npy") for f in ("F182M", "F210M") if (sdir / f"{f}.npy").exists()}
    stars = ndimage.binary_dilation(sw > 0.01, iterations=3)
    tiles, labels = [], []
    for name in ("sky", "sky2", "cloud"):
        sl = cut(name)
        for f, L in layers.items():
            t = np.nan_to_num(L[sl])
            lo, hi = np.percentile(t, [1, 99])
            tiles.append(np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1)[::-1])
            v, h = stripes(t * 255, ~stars[sl])
            labels.append(f"{name} · {f} после растяжки (1–99%)\nполосы верт./гориз. {v:.1f} / {h:.1f}")
            rows.append(f"полосы {f} в {name} после растяжки: верт. / гориз. (σ медиан / шум)\t{v:.1f} / {h:.1f}\t—")
        lum = img[sl] @ LUMA * 255
        v, h = stripes(lum, ~stars[sl])
        tiles.append(img[sl][::-1]); labels.append(f"{name} · итог 100%\nполосы верт./гориз. {v:.1f} / {h:.1f}")
        rows.append(f"полосы итога (яркость) в {name}: верт. / гориз.\t{v:.1f} / {h:.1f}\t—")
    contact_sheet(tiles, labels, out / "34_полосы_F182M_F210M.png", cols=len(layers) + 1, tile_inches=4.2,
                  interpolation="nearest", title=f"W3-2 bin {b}: полосы 1/f коротковолновых при 100% после растяжки")
    (out / "33_числа.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
