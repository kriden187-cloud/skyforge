"""3k, п.3: гладкая модель крупного хода неба для показа (только стиль, где включена) — полином низкого порядка
по x и y на фильтр, подгонка по пустому небу вне файла исключения (туманность с кольцом ореола и круг у центра),
внутри исключённой зоны — продолжение полинома. Стек не меняется: поправка — до растяжки, после clean."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from skyforge.config import Config
from skyforge.paths import Paths

SKY_POLY_VERSION = 3  # поднять при смене алгоритма — входит в ключ кеша растяжки
POLY_MAX_POINTS = 300_000
POLY_ITERS, POLY_CLIP = 6, 3.0


def poly_terms(u: np.ndarray, v: np.ndarray, order: int) -> np.ndarray:
    """Столбцы u^i·v^j, i + j ≤ order (u, v — координаты в −1…1)."""
    return np.column_stack([u ** i * v ** (k - i) for k in range(order + 1) for i in range(k, -1, -1)])


def fit_poly(a: np.ndarray, ok: np.ndarray, order: int, zero_on: np.ndarray | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Робастный полином по пикселям ok (отсечка POLY_CLIP·σ по MAD от медианы остатков). Возвращает модель на всём
    кадре без своего среднего — по zero_on (зона исключения: туманность не сдвигается, внутри только форма) или,
    без него, по точкам подгонки."""
    H, W = a.shape
    ys, xs = np.nonzero(ok)
    step = max(1, ys.size // POLY_MAX_POINTS)
    ys, xs = ys[::step], xs[::step]
    u, v = (xs - W / 2) / (W / 2), (ys - H / 2) / (H / 2)
    A = poly_terms(u, v, order)
    b = a[ys, xs].astype(np.float64)
    sel = np.ones(b.size, dtype=bool)
    coef = np.zeros(A.shape[1])
    for _ in range(POLY_ITERS):
        coef, *_ = np.linalg.lstsq(A[sel], b[sel], rcond=None)
        r = b - A @ coef
        mid = float(np.median(r[sel]))
        mad = 1.4826 * float(np.median(np.abs(r[sel] - mid)))
        new = np.abs(r - mid) <= POLY_CLIP * max(mad, 1e-12)
        if np.array_equal(new, sel):
            break
        sel = new
    yy, xx = np.mgrid[0:H, 0:W]
    model = (poly_terms(((xx - W / 2) / (W / 2)).ravel(), ((yy - H / 2) / (H / 2)).ravel(), order) @ coef).reshape(H, W)
    model -= float(np.mean(model[zero_on])) if zero_on is not None and zero_on.any() else float(np.mean(A[sel] @ coef))
    return model.astype(np.float32), {"points": int(sel.sum()) * step, "coef": [round(float(x), 8) for x in coef]}


def exclusion_on_grid(cfg: Config, paths: Paths, wcs: Any, shape: tuple[int, int]) -> np.ndarray:
    from skyforge.level2 import reproject_mask

    return reproject_mask(paths.abs(cfg.stretch.sky_poly_exclude), wcs, shape)


def apply_sky_poly(cfg: Config, paths: Paths, arrays: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict, dict]:
    """Вычесть полином неба из каждого фильтра (stretch.sky_poly_order > 0). Возвращает (кадры, модели, сведения)."""
    if cfg.stretch.sky_poly_order <= 0:
        return arrays, {}, {}
    from skyforge.align import load_aligned
    from skyforge.clean import load_sky_mask
    from skyforge.log import SkyforgeError

    if not cfg.stretch.sky_poly_exclude:
        raise SkyforgeError("stretch.sky_poly_order > 0: задай stretch.sky_poly_exclude — файл исключения (туманность, кольцо, круг)")
    sky = load_sky_mask(cfg, paths)
    if sky is None:
        raise SkyforgeError("stretch.sky_poly_order > 0: нет маски неба — запусти skyforge clean")
    _, wcs, _ = load_aligned(cfg, paths)
    excl = exclusion_on_grid(cfg, paths, wcs, sky.shape)
    taper = None
    if cfg.stretch.sky_poly_taper_px > 0:
        from skyforge.clean import load_nebula_sd

        sd = load_nebula_sd(cfg, paths)
        taper = np.clip(-sd / (cfg.stretch.sky_poly_taper_px / cfg.bin), 0, 1).astype(np.float32)
    out, models, info = {}, {}, {}
    for f, a in arrays.items():
        ok = sky & ~excl & np.isfinite(a)
        m, fi = fit_poly(np.asarray(a, dtype=np.float32), ok, cfg.stretch.sky_poly_order, excl & np.isfinite(a))
        v = a[ok]
        sig = float(1.4826 * np.median(np.abs(v - np.median(v))))
        fi.update(sigma_sky=round(sig, 6), range_sigma=round(float(np.ptp(m[np.isfinite(a)])) / sig, 3) if sig else None,
                  in_excluded_range_sigma=round(float(np.ptp(m[excl & np.isfinite(a)])) / sig, 3) if sig else None)
        if taper is not None:
            m = (m * taper).astype(np.float32)
        out[f] = (np.asarray(a, dtype=np.float32) - m).astype(np.float32)
        models[f] = m
        info[f] = fi
    info["_excluded_percent"] = round(float(excl.mean() * 100), 1)
    models["_excl"] = excl
    return out, models, info


def sky_poly_sheet(models: dict, info: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filters = [f for f in models if not f.startswith("_")]
    excl = models["_excl"]
    fig, ax = plt.subplots(2, (len(filters) + 1) // 2, figsize=(6 * ((len(filters) + 1) // 2), 10), squeeze=False)
    for a_, f in zip(ax.ravel(), filters):
        m = models[f] / info[f]["sigma_sky"]
        lim = max(0.2, float(np.nanmax(np.abs(m))))
        im = a_.imshow(m[::-1], cmap="RdBu_r", vmin=-lim, vmax=lim)
        a_.contour(excl[::-1].astype(float), levels=[0.5], colors="k", linewidths=0.6)
        a_.set_title(f"{f}: полином неба, σ неба; в зоне исключения размах {info[f]['in_excluded_range_sigma']}σ", fontsize=10)
        a_.axis("off")
        fig.colorbar(im, ax=a_, fraction=0.04)
    fig.suptitle("Гладкая модель крупного хода неба (вычитается до растяжки); чёрный контур — зона, где полином продолжен")
    fig.tight_layout(); fig.savefig(path, dpi=55); plt.close(fig)
