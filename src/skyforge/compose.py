"""Раскраска: каждый фильтр получает оттенок по длине волны, слои смешиваются в RGB.

Короткие волны — синий конец hue_range, длинные — красный. Оттенки раздаются
по порядку фильтров, а не по абсолютной длине волны, как в пресс-релизах:
так соседние фильтры не сливаются в один цвет.
"""

from __future__ import annotations

import colorsys
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skyforge.align import load_aligned
from skyforge.config import ColorConfig, Config
from skyforge.filters import sort_by_wavelength, wavelength_um
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, save_png, shrink
from skyforge.stretch import sample, stretch_key, stretched_dir

# Яркость по Rec. 709 — для насыщенности и нейтрального фона.
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def hex_to_rgb(h: str) -> tuple[float, float, float]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def assign_colors(filters: list[str], mission: str, instrument: str, c: ColorConfig) -> dict[str, tuple[float, float, float]]:
    ordered = sort_by_wavelength(filters, mission, instrument)
    n = len(ordered)
    colors = {}
    for i, f in enumerate(ordered):
        if f in c.overrides:
            colors[f] = hex_to_rgb(c.overrides[f])
        elif n == 1:
            colors[f] = (1.0, 1.0, 1.0)
        else:
            h0, h1 = c.hue_range
            hue = (h0 + (h1 - h0) * i / (n - 1)) % 360
            colors[f] = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
    return colors


def blend(layers: dict[str, np.ndarray], colors: dict[str, tuple[float, float, float]], c: ColorConfig) -> np.ndarray:
    first = next(iter(layers.values()))
    h, w = first.shape
    if c.blend == "screen":
        # 1 − Π(1 − слой·цвет): одиночный слой сохраняет яркость, совпадения светлеют к белому.
        inv = np.ones((h, w, 3), dtype=np.float32)
        for f, L in layers.items():
            col = np.asarray(colors[f], dtype=np.float32) * np.float32(c.weights.get(f, 1.0))
            Lf = np.nan_to_num(np.asarray(L, dtype=np.float32), nan=0.0)
            inv *= 1 - np.clip(Lf[..., None] * col, 0, 1)
        rgb = 1 - inv
    else:
        # Взвешенное среднее по каналам: все слои по 1 дают белый.
        num = np.zeros((h, w, 3), dtype=np.float32)
        den = np.zeros(3, dtype=np.float32)
        for f, L in layers.items():
            col = np.asarray(colors[f], dtype=np.float32) * np.float32(c.weights.get(f, 1.0))
            num += np.nan_to_num(np.asarray(L, dtype=np.float32), nan=0.0)[..., None] * col
            den += col
        rgb = num / np.where(den > 0, den, 1)
    # Где нет ни одного фильтра — честный чёрный, а не случайный цвет.
    nodata = np.all([~np.isfinite(np.asarray(L)) for L in layers.values()], axis=0)
    rgb[nodata] = 0
    return rgb


def neutralize(rgb: np.ndarray, pct: float) -> tuple[np.ndarray, list[float]]:
    """Фон каждого канала в ноль: по самым тёмным пикселям, чтобы небо было серым, а не зелёным."""
    luma = rgb @ LUMA
    dark = sample(luma) <= np.nanpercentile(sample(luma), pct)
    bg = []
    for k in range(3):
        ch = sample(rgb[..., k])
        bg.append(float(np.nanmedian(ch[dark])))
    bgv = np.asarray(bg, dtype=np.float32)
    out = (rgb - bgv) / np.where(bgv < 1, 1 - bgv, 1)
    return np.clip(out, 0, 1), bg


def saturate(rgb: np.ndarray, s: float) -> np.ndarray:
    if s == 1.0:
        return rgb
    luma = (rgb @ LUMA)[..., None]
    return np.clip(luma + s * (rgb - luma), 0, 1)


def compose_layers(cfg: Config, layers: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, Any]]:
    colors = assign_colors(list(layers), cfg.mission, cfg.instrument, cfg.color)
    rgb = blend(layers, colors, cfg.color)
    bg = None
    if cfg.color.neutral_background:
        rgb, bg = neutralize(rgb, cfg.color.background_percentile)
    rgb = saturate(rgb, cfg.color.saturation).astype(np.float32)
    info = {"colors": {f: "#%02x%02x%02x" % tuple(round(v * 255) for v in colors[f]) for f in colors},
            "background_removed": bg}
    return rgb, info


def composite_path(cfg: Config, paths: Paths) -> Path:
    return paths.work(cfg.slug, cfg.bin) / "composite.npy"


def compose_key(cfg: Config, align_meta: dict[str, Any]) -> dict[str, Any]:
    return {**stretch_key(cfg, align_meta), "filters": cfg.filters, "color": cfg.color.model_dump(mode="json")}


def load_stretched(cfg: Config, paths: Paths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    _, _, align_meta = load_aligned(cfg, paths)
    d = stretched_dir(cfg, paths)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        raise SkyforgeError("нет растянутых кадров: сначала skyforge stretch")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["key"] != stretch_key(cfg, align_meta):
        raise SkyforgeError("растяжка сделана с другими параметрами или кадрами: перезапусти skyforge stretch")
    want = [f for f in meta["filters"] if not cfg.filters or f in cfg.filters]
    return {f: np.load(d / f"{f}.npy") for f in want}, align_meta


def compose(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    layers, align_meta = load_stretched(cfg, paths)
    rgb, info = compose_layers(cfg, layers)
    np.save(composite_path(cfg, paths), rgb)
    composite_path(cfg, paths).with_suffix(".json").write_text(
        json.dumps({"key": compose_key(cfg, align_meta), **info}, ensure_ascii=False, indent=2), encoding="utf-8")
    for f, col in info["colors"].items():
        console.print(f"  {f} ({wavelength_um(f, cfg.mission, cfg.instrument):.2f} мкм) → {col}")

    pv = cfg.preview.max_size
    save_png(rgb, out_dir / "compose.png", pv)
    colored = []
    for f, L in layers.items():
        col = np.asarray(hex_to_rgb(info["colors"][f]), dtype=np.float32)
        colored.append(np.nan_to_num(shrink(np.asarray(L), pv // 3))[..., None] * col)
    contact_sheet(colored, [f"{f} · {info['colors'][f]}" for f in layers], out_dir / "compose_layers.png",
                  title="Слои с назначенными цветами")
    _legend(info["colors"], cfg, out_dir / "compose_legend.png")
    info["previews"] = ["compose.png", "compose_layers.png", "compose_legend.png"]
    return info


def _legend(colors: dict[str, str], cfg: Config, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.4 * len(colors) + 1, 1.4), facecolor="#111")
    for i, (f, col) in enumerate(colors.items()):
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 0.6, color=col))
        ax.text(i + 0.45, 0.75, f, ha="center", color="#eee", fontsize=10)
        ax.text(i + 0.45, -0.25, f"{wavelength_um(f, cfg.mission, cfg.instrument):.2f} мкм", ha="center",
                color="#aaa", fontsize=8)
    ax.set_xlim(-0.1, len(colors))
    ax.set_ylim(-0.5, 1.0)
    ax.axis("off")
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
