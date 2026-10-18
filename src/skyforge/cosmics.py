"""Следы космических лучей и тонкие зазоры между чипами.

Луч. У Hubble часть фильтров снята одной экспозицией, и сложение следы не убрало. След — резкий
выброс над мелкой структурой (как в L.A.Cosmic: остаток после медианы 3×3, который заметно больше
перепада между медианами 3×3 и 7×7). Звёзды в WFPC2 тоже резкие, поэтому решает сверка фильтров:
настоящий источник есть во всех фильтрах на том же месте неба, а луч — только в одном.

Зазор. Между чипами WFPC2 в одиночной экспозиции остаётся пустая полоса в несколько пикселей.
Это NaN-пиксели, которые исчезают при морфологическом замыкании маски данных окном
ширины зазора; заполняются средним известных соседей снаружи внутрь.
"""

from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy import ndimage

from skyforge.config import CosmicsConfig
from skyforge.stretch import sample


def _fine_structure(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    smp = sample(a)
    fill = float(np.nanmedian(smp))
    f = np.where(np.isfinite(a), a, fill).astype(np.float32)
    med3 = ndimage.median_filter(f, 3)
    med7 = ndimage.median_filter(med3, 7)
    hp = f - med3
    hs = sample(np.where(np.isfinite(a), hp, np.nan))
    _, _, noise = sigma_clipped_stats(hs[np.isfinite(hs)], sigma=3.0)
    return f, hp, med3 - med7, float(noise)


def find_cosmics(layers: dict[str, np.ndarray], c: CosmicsConfig, scale_arcsec: float) -> dict[str, np.ndarray]:
    """Маска следов для каждого фильтра (сверка по всем остальным)."""
    parts = {f: _fine_structure(a) for f, a in layers.items()}
    protect = max(0, int(round(c.protect_arcsec / scale_arcsec)))
    seen = {}
    for f, (_, hp, _, noise) in parts.items():
        s = hp > c.others_sigma * noise
        if protect:
            s = ndimage.maximum_filter(s.astype(np.uint8), size=2 * protect + 1) > 0
        seen[f] = s
    masks = {}
    for f, (_, hp, fine, noise) in parts.items():
        cand = (hp > c.sigma * noise) & (hp > c.sharpness * np.maximum(fine, noise))
        others = [seen[g] for g in layers if g != f]
        if others:
            cand &= ~np.logical_or.reduce(others)
        if c.grow_px:
            cand = ndimage.binary_dilation(cand, iterations=c.grow_px)
        masks[f] = cand & np.isfinite(layers[f])
    return masks


def remove_cosmics(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return a
    f = np.where(np.isfinite(a), a, np.nanmedian(sample(a))).astype(np.float32)
    out = np.array(a, dtype=np.float32, copy=True)
    out[mask] = ndimage.median_filter(f, 7)[mask]
    return out


def find_gaps(a: np.ndarray, max_width_px: int) -> np.ndarray:
    valid = np.isfinite(a)
    k = 2 * max(1, max_width_px) + 1
    closed = ndimage.binary_closing(valid, structure=np.ones((k, k), dtype=bool))
    return closed & ~valid


def fill_mean(a: np.ndarray, holes: np.ndarray) -> np.ndarray:
    """Заполнить дыры средним известных соседей, кольцами снаружи внутрь."""
    out = np.array(a, dtype=np.float32, copy=True)
    todo = holes.copy()
    kernel = np.ones((3, 3), dtype=np.float32)
    while todo.any():
        known = np.isfinite(out) & ~todo
        vals = np.where(known, out, 0).astype(np.float32)
        s = ndimage.convolve(vals, kernel, mode="constant")
        n = ndimage.convolve(known.astype(np.float32), kernel, mode="constant")
        ring = todo & (n > 0)
        if not ring.any():
            break
        out[ring] = s[ring] / n[ring]
        todo &= ~ring
    return out
