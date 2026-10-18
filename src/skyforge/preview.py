"""Превью: уменьшение, 8-битные PNG, листы с подписями. Превью не участвуют в итоговом файле."""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


def nan_block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Среднее по блокам factor×factor без учёта NaN. Работает и для HxW, и для HxWx3."""
    if factor <= 1:
        return np.asarray(a, dtype=np.float32)
    h, w = a.shape[:2]
    h2, w2 = h // factor * factor, w // factor * factor
    a = np.asarray(a[:h2, :w2], dtype=np.float32)
    shape = (h2 // factor, factor, w2 // factor, factor) + a.shape[2:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a.reshape(shape), axis=(1, 3)).astype(np.float32)


def shrink(a: np.ndarray, max_size: int) -> np.ndarray:
    factor = math.ceil(max(a.shape[:2]) / max_size)
    return nan_block_mean(a, factor)


def to_uint8(a01: np.ndarray) -> np.ndarray:
    a = np.nan_to_num(np.asarray(a01, dtype=np.float32), nan=0.0)
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_png(a01: np.ndarray, path: Path, max_size: int | None = None) -> Path:
    if max_size:
        a01 = shrink(a01, max_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(a01)).save(path)
    return path


def quick_norm(a: np.ndarray, lo_pct: float, hi_pct: float, a_param: float) -> np.ndarray:
    """Быстрая asinh-нормализация только для превью (параметры берутся из конфига)."""
    sample = a[:: max(1, a.shape[0] // 1000), :: max(1, a.shape[1] // 1000)]
    lo, hi = np.nanpercentile(sample, [lo_pct, hi_pct])
    if not np.isfinite(hi - lo) or hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    x = np.clip((a - lo) / (hi - lo), 0, 1)
    return (np.arcsinh(x / a_param) / np.arcsinh(1 / a_param)).astype(np.float32)


def contact_sheet(
    images: list[np.ndarray],
    labels: list[str],
    path: Path,
    cols: int | None = None,
    title: str | None = None,
    tile_inches: float = 4.0,
    interpolation: str = "lanczos",
) -> Path:
    """Лист картинок с подписями (серые или цветные, значения 0–1)."""
    n = len(images)
    cols = cols or min(n, 3)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(tile_inches * cols, tile_inches * rows + (0.5 if title else 0)),
                             squeeze=False, facecolor="#111")
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, label in zip(axes.flat, images, labels):
        img = np.nan_to_num(img, nan=0.0)
        ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1,
                  interpolation=interpolation)
        ax.set_title(label, color="#ddd", fontsize=11)
    if title:
        fig.suptitle(title, color="#fff", fontsize=13)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def checkerboard(a: np.ndarray, b: np.ndarray, cell: int) -> np.ndarray:
    """Шахматка из двух одинаковых по размеру картинок: чётные клетки из a, нечётные из b."""
    h, w = a.shape[:2]
    yy, xx = np.indices((h, w))
    mask = ((yy // cell) + (xx // cell)) % 2 == 0
    return np.where(mask, a, b)
