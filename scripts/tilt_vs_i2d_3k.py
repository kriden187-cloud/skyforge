"""3k, п.2(б): чей общий наклон по x в коротких волнах — сведения по усилителям или общий с i2d STScI.

Собирает F187N ещё раз с одной константой на экспозицию (как skymatch STScI: без усилителей и без наклона)
в data/work/ngc3132/compare_3k/ и сравнивает с i2d на сетке i2d три сборки: phase-3j (константа на экспозицию
и усилитель, копия до пересборки), одна константа на экспозицию, текущая (усилители + наклон по y).
Разность «сборка − i2d» по свободному небу (маска источников сборки): плоскость (наклон по x и y, σ неба на 1000 px)
и карта, сглаженная 48 px. Лист 15 и таблица 15.

Запуск: uv run python scripts/tilt_vs_i2d_3k.py <папка> [фильтр]
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
import numpy as np
from astropy.io import fits
from astropy.wcs.utils import proj_plane_pixel_scales

from skyforge.background import fit_plane
from skyforge.config import Config
from skyforge.fetch import level2_files, require_products
from skyforge.level2 import _build_filter, _wcs, source_mask_grid
from skyforge.paths import Paths

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from background_diag_3k import smooth_on  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    f = sys.argv[2] if len(sys.argv) > 2 else "F187N"
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase")
    cmp_dir = P.work_root / "ngc3132" / "compare_3k"
    single = cmp_dir / f"{f}_single_const.fits"
    l3 = require_products(cfg, P, level3=True)
    if not single.exists():
        c1 = cfg.model_copy(update={"level2": cfg.level2.model_copy(update={"match_per_amp": False, "match_curve_y": False})})
        _build_filter(c1, f, P.abs(l3[f].path), level2_files(cfg, P)[f], single)
    meta = json.loads((P.work_root / "ngc3132" / "level2" / "meta.json").read_text(encoding="utf-8"))
    builds = {"phase-3j: константа на усилитель": cmp_dir / f"{f}_per_amp_phase3j.fits",
              "одна константа на экспозицию (как STScI)": single}
    if cfg.level2.match_curve_y:
        builds["п.2: усилители + кривая по y"] = P.work_root / "ngc3132" / "level2" / meta["filters"][f]["file"]
    with fits.open(P.abs(l3[f].path), memmap=True) as h:
        i2d = np.array(h["SCI"].data, dtype=np.float32)
        if "WHT" in h:
            i2d[np.asarray(h["WHT"].data) <= 0] = np.nan
        gw = _wcs(h["SCI"].header)
    scale = float(np.mean(proj_plane_pixel_scales(gw))) * 3600
    free = np.isfinite(i2d) & ~source_mask_grid(i2d, scale, cfg)
    rows = ["сборка\tσ неба\tнаклон (сборка − i2d) по x, σ/1000 px\tпо y\tразброс разности 5–95%, σ"]
    fig, ax = plt.subplots(1, len(builds), figsize=(7 * len(builds), 7))
    for a_, (lab, p) in zip(ax, builds.items()):
        with fits.open(p, memmap=True) as h:
            S = np.array(h["SCI"].data, dtype=np.float32)
        ok = free & np.isfinite(S)
        sig = float(1.4826 * np.median(np.abs(S[ok] - np.median(S[ok]))))
        d = S - i2d
        d = d - np.median(d[ok])
        yy, xx = np.nonzero(ok)
        step = max(1, yy.size // 300_000)
        coef = fit_plane(xx[::step].astype(float), yy[::step].astype(float), d[yy[::step], xx[::step]].astype(float))
        m = smooth_on(ok, np.where(ok, d, 0), 48) / sig
        sp = float(np.nanpercentile(m, 95) - np.nanpercentile(m, 5))
        rows.append(f"{lab}\t{sig:.4g}\t{coef[1] * 1000 / sig:+.3f}\t{coef[2] * 1000 / sig:+.3f}\t{sp:.2f}")
        a_.imshow(m[::-1], cmap="PuOr_r", vmin=-0.8, vmax=0.8)
        a_.set_title(f"{f}: {lab} − i2d\nнаклон x {coef[1] * 1000 / sig:+.2f}, y {coef[2] * 1000 / sig:+.2f} σ/1000 px", fontsize=11)
        a_.axis("off")
    fig.suptitle(f"{f}: разность сборки и i2d STScI по свободному небу (сглажено 48 px, σ неба сборки; шкала ±0,8σ)")
    fig.tight_layout(); fig.savefig(out / f"15_{f}_наклон_x_к_i2d.png", dpi=55); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / f"15_{f}_наклон_x_к_i2d.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
