"""Сравнение с принятым результатом (предыдущим тегом): карта разности по туманности, фактура, кропы рядом.

Средний сдвиг яркости не ловит выжигание: оно поднимает яркие пиксели и гасит тёмные, а среднее
почти не меняется. Поэтому кроме средней разности считается фактура — разброс яркости в кольце
вокруг центральной звезды — и кропы приёмки ставятся «было | стало».
Принятые картинки лежат в data/baselines/<slug>/<имя>/<стиль>_bin<N>.png (команда skyforge baseline).
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
DIFF_GAIN = 10  # карта разности усиливается, чтобы сдвиг в 0.01 был виден


def baseline_dir(paths: Paths, slug: str, name: str) -> Path:
    return paths.root / "data" / "baselines" / slug / name


def baseline_file(paths: Paths, cfg: Config) -> Path | None:
    if not cfg.report.baseline:
        return None
    return baseline_dir(paths, cfg.slug, cfg.report.baseline) / f"{cfg.style or 'default'}_bin{cfg.bin}.png"


def save_baseline(paths: Paths, slug: str, name: str, run_dir: Path) -> list[Path]:
    """Итоговые PNG прогона (папка стиля или папка --all со стилями) → data/baselines/<slug>/<name>/."""
    pat = re.compile(rf"^{re.escape(slug)}(?:_(?P<style>[a-z0-9-]+?))?(?:_bin(?P<bin>\d+))?\.png$")
    found = []
    for p in sorted(list(run_dir.glob("*.png")) + list(run_dir.glob("*/*.png"))):
        m = pat.match(p.name)
        if m:
            found.append((p, m.group("style") or "default", int(m.group("bin") or 1)))
    if not found:
        raise SkyforgeError(f"в {run_dir} нет итоговых {slug}*.png")
    dst = baseline_dir(paths, slug, name)
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p, style, bin_ in found:
        target = dst / f"{style}_bin{bin_}.png"
        shutil.copy2(p, target)
        out.append(target)
    return out


def _load(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255


def texture_sigma(img: np.ndarray, cx: float, cy: float, r0: float, r1: float) -> float:
    lum = img @ LUMA
    yy, xx = np.indices(lum.shape)
    r = np.hypot(xx - cx, yy - cy)
    ring = (r > r0) & (r < r1)
    return float(lum[ring].std()) if ring.any() else float("nan")


def compare(cfg: Config, paths: Paths, out_dir: Path, final_png: Path) -> dict[str, Any]:
    p = baseline_file(paths, cfg)
    if p is None:
        return {"skipped": "report.baseline не задан"}
    if not p.exists():
        return {"skipped": f"нет принятой картинки {paths.rel(p)} (сохраняется: skyforge baseline)"}
    from skyforge.clean import load_nebula_sd

    cur, base = _load(final_png), _load(p)
    if cur.shape != base.shape:
        return {"skipped": f"размер {cur.shape[1]}×{cur.shape[0]} не совпадает с принятым {base.shape[1]}×{base.shape[0]}"}
    sd = load_nebula_sd(cfg, paths)
    neb = np.flipud(sd) > 0 if sd is not None and sd.shape == cur.shape[:2] else np.ones(cur.shape[:2], bool)
    diff = np.abs(cur - base).mean(axis=-1)

    b = cfg.bin
    core = next((c for c in cfg.report.crops if c.name == "core"), None)
    cx, cy = (core.x / b, core.y / b) if core else (cur.shape[1] / 2, cur.shape[0] / 2)
    r0, r1 = (v / b for v in cfg.report.texture_ring)
    t_cur, t_base = texture_sigma(cur, cx, cy, r0, r1), texture_sigma(base, cx, cy, r0, r1)
    change = (t_cur - t_base) / t_base * 100 if t_base else float("nan")
    info = {
        "baseline": cfg.report.baseline,
        "nebula_diff_mean": round(float(diff[neb].mean()), 5),
        "nebula_diff_p99": round(float(np.percentile(diff[neb], 99)), 4),
        "texture_sigma": round(t_cur, 5),
        "texture_sigma_baseline": round(t_base, 5),
        "texture_change_percent": round(change, 1),
        "texture_ring_px": [cfg.report.texture_ring[0], cfg.report.texture_ring[1]],
    }
    console.print(f"  сравнение с {cfg.report.baseline}: средняя разность по туманности {info['nebula_diff_mean']:.4f}, "
                  f"фактура σ {t_base:.4f} → {t_cur:.4f} ({change:+.1f}%)")

    # Картинки рисуются в ориентации экрана: contact_sheet ждёт сетку (строка 0 — низ), поэтому flipud.
    heat = np.clip(diff * DIFF_GAIN, 0, 1)
    heat_rgb = np.stack([heat, heat * 0.55, heat * 0.15], axis=-1)
    heat_rgb[~neb] *= 0.25  # вне маски туманности — приглушено
    size = cfg.preview.max_size // 2
    panels = [np.flipud(shrink(base, size)), np.flipud(shrink(cur, size)), np.flipud(shrink(heat_rgb, size))]
    contact_sheet(panels, [f"принято: {cfg.report.baseline}", "сейчас",
                           f"|разность| ×{DIFF_GAIN}, по туманности средняя {info['nebula_diff_mean']:.4f}"],
                  out_dir / "baseline_diff.png", cols=3, tile_inches=5,
                  title=f"Фактура в кольце {cfg.report.texture_ring[0]}–{cfg.report.texture_ring[1]} px вокруг "
                        f"центральной звезды: σ {t_base:.4f} → {t_cur:.4f} ({change:+.1f}%)")
    previews = ["baseline_diff.png"]
    if cfg.report.crops:
        imgs, labels = [], []
        for c in cfg.report.crops:
            s = max(8, c.size // b)
            x0, y0 = int(np.clip(c.x // b - s // 2, 0, cur.shape[1] - s)), int(np.clip(c.y // b - s // 2, 0, cur.shape[0] - s))
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            imgs += [np.flipud(base[sl]), np.flipud(cur[sl])]
            labels += [f"{c.name} · {cfg.report.baseline}", f"{c.name} · сейчас"]
        contact_sheet(imgs, labels, out_dir / "crops_compare.png", cols=2, tile_inches=4.5, interpolation="nearest",
                      title="Кропы приёмки 100%: слева принятое, справа сейчас")
        previews.append("crops_compare.png")
    info["previews"] = previews
    return info
