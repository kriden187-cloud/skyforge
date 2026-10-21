"""Фаза 4, Sh2-305: кропы по фильтрам после align или clean — asinh по общим процентилям фильтра (все кропы одного
фильтра в одной растяжке), NaN (насыщенные ядра, пустоты) — красным. Для неба: ×k и рядом карта областей CON.

Запуск: uv run python scripts/s305_crops.py <align|clean> <лист.png> <кроп>[:k] [<кроп>[:k] ...] [--con F212N] [--cfg путь] [--bin 1]
  k — усиление яркости (небо ×5); --con F — добавить к каждому кропу карту областей CON фильтра F;
  --cfg — другой конфиг (по умолчанию configs/sh2305/base.yaml).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

from skyforge.align import aligned_dir, load_aligned, with_resolved_crops
from skyforge.clean import load_input
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop


def main() -> None:
    stage, sheet = sys.argv[1], pathlib.Path(sys.argv[2])
    args = sys.argv[3:]
    con_f, cfg_path = None, "configs/sh2305/base.yaml"
    if "--con" in args:
        i = args.index("--con")
        con_f, args = args[i + 1], args[:i] + args[i + 2:]
    if "--cfg" in args:                                   # другой конфиг (проба в отдельной папке)
        i = args.index("--cfg")
        cfg_path, args = args[i + 1], args[:i] + args[i + 2:]
    bin_ = None
    if "--bin" in args:                                   # шаг 4d: кропы при bin 1 (по умолчанию — bin из конфига)
        i = args.index("--bin")
        bin_, args = int(args[i + 1]), args[:i] + args[i + 2:]
    P = Paths(pathlib.Path(".").resolve())
    c = Config.load(cfg_path)
    if bin_:
        c = c.model_copy(update={"bin": bin_})
    arrays = load_input(c, P)[0] if stage == "clean" else load_aligned(c, P)[0]
    crops = {k.name: k for k in with_resolved_crops(c, P).report.crops}
    shape = next(iter(arrays.values())).shape
    lims = {}
    for f, a in arrays.items():
        v = a[np.isfinite(a)][::7]
        lims[f] = tuple(np.percentile(v, [5, 99.8]))
    tiles, labels = [], []
    for spec in args:
        name, _, k = spec.partition(":")
        k = float(k) if k else 1.0
        cr = crops[name]
        sl = display_crop(shape, cr.x, cr.y, cr.size, c.bin)
        for f, a in arrays.items():
            lo, hi = lims[f]
            t = a[sl]
            s = np.arcsinh(np.clip((np.nan_to_num(t, nan=lo) - lo) / (hi - lo), 0, None) * 10) / np.arcsinh(10)
            rgb = np.repeat(np.clip(s * k, 0, 1)[..., None], 3, -1)
            rgb[~np.isfinite(t)] = [1.0, 0.1, 0.1]
            tiles.append(rgb)
            nan = (~np.isfinite(t)).sum()
            labels.append(f"{name} · {f}" + (f" · ×{k:g}" if k != 1 else "") + (f" · NaN {nan} px" if nan else ""))
        if con_f:
            tl = np.load(aligned_dir(c, P) / f"{con_f}.tiles.npy")[sl]
            rng = np.random.default_rng(0)
            pal = rng.uniform(0.15, 1, (int(np.nanmax(tl)) + 2, 3))
            img = pal[np.where(tl >= 0, tl, -1).astype(int)]
            img[tl < 0] = 0
            tiles.append(img.astype(np.float32))
            labels.append(f"{name} · области CON {con_f}")
    cols = len(arrays) + (1 if con_f else 0)
    contact_sheet(tiles, labels, sheet, cols=cols, tile_inches=3.2, interpolation="nearest",
                  title=f"Sh2-305 [{c.slug}] после {stage} (bin {c.bin}): кропы по фильтрам, asinh по процентилям фильтра; "
                        f"красное — NaN")


if __name__ == "__main__":
    main()
