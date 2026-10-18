"""Финиш: денойз, резкость, экспорт 16-битного TIFF и PNG с метаданными.

В файлы не пишется время: тот же конфиг даёт байт-в-байт тот же файл.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle

from skyforge import __version__
from skyforge.align import load_aligned
from skyforge.compose import composite_path, compose_key
from skyforge.config import Config, FinishConfig
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, save_png, to_uint8

# denoise: 1.0 соответствует весу TV-регуляризации 0.1 на шкале яркости 0–1 — это уже сильное сглаживание.
DENOISE_WEIGHT_AT_MAX = 0.1


def unsharp(rgb: np.ndarray, radius: float, amount: float) -> np.ndarray:
    """Нерезкая маска: картинка + amount·(картинка − размытая). Своя, потому что
    skimage.filters.unsharp_mask 0.26 на RGB float32 возвращал тёмный кадр (максимум 0.49 вместо 1)."""
    blurred = np.stack([gaussian_filter(rgb[..., k], sigma=radius) for k in range(rgb.shape[-1])], axis=-1)
    return (rgb + np.float32(amount) * (rgb - blurred)).astype(np.float32)


def finish_array(rgb: np.ndarray, f: FinishConfig) -> np.ndarray:
    out = np.asarray(rgb, dtype=np.float32)
    if f.denoise > 0:
        out = denoise_tv_chambolle(out, weight=f.denoise * DENOISE_WEIGHT_AT_MAX, channel_axis=-1,
                                   max_num_iter=f.denoise_iters).astype(np.float32)
    if f.sharpen > 0:
        out = unsharp(out, f.sharpen_radius, f.sharpen)
    return np.clip(np.nan_to_num(out, nan=0.0), 0, 1)


def export_metadata(cfg: Config, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "title": f"{cfg.target} — {cfg.mission.upper()} {cfg.instrument.upper()}",
        "credit": f"{cfg.credit}; обработка: {cfg.author}",
        "software": f"skyforge {__version__}",
        "filters": extra.get("colors"),
        "config_digest": cfg.digest(),
    }


def export(rgb: np.ndarray, base: Path, meta: dict[str, Any]) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    tif = base.with_suffix(".tif")
    png = base.with_suffix(".png")
    u16 = (np.clip(rgb, 0, 1) * 65535 + 0.5).astype(np.uint16)
    desc = json.dumps(meta, ensure_ascii=False, sort_keys=True)
    # Строки TIFF — только ASCII, поэтому кириллица там в виде \\uXXXX; в PNG — как есть.
    tifffile.imwrite(tif, u16, photometric="rgb", compression="zlib",
                     description=json.dumps(meta, ensure_ascii=True, sort_keys=True),
                     software=meta["software"], metadata=None)
    info = PngInfo()
    info.add_itxt("Title", meta["title"])
    info.add_itxt("Author", meta["credit"])
    info.add_itxt("Software", meta["software"])
    info.add_itxt("Description", desc)
    Image.fromarray(to_uint8(rgb)).save(png, pnginfo=info)
    return {"tif": tif.name, "png": png.name}


def output_base(cfg: Config, out_dir: Path) -> Path:
    suffix = f"_bin{cfg.bin}" if cfg.bin > 1 else ""
    return out_dir / f"{cfg.slug}{suffix}"


def finish(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    cp = composite_path(cfg, paths)
    meta_path = cp.with_suffix(".json")
    if not cp.exists() or not meta_path.exists():
        raise SkyforgeError("нет цветной композиции: сначала skyforge compose")
    _, _, align_meta = load_aligned(cfg, paths)
    cmeta = json.loads(meta_path.read_text(encoding="utf-8"))
    if cmeta["key"] != compose_key(cfg, align_meta):
        raise SkyforgeError("композиция сделана с другими параметрами: перезапусти skyforge stretch и compose")

    rgb = np.load(cp)
    done = finish_array(rgb, cfg.finish)
    files = export(done, output_base(cfg, out_dir), export_metadata(cfg, cmeta))
    console.print(f"  итог: {files['png']}, {files['tif']} ({done.shape[1]}×{done.shape[0]})")

    pv = cfg.preview
    save_png(done, out_dir / "finish_preview.png", pv.max_size)
    # Кроп 100% из центра до и после финиша: видно, что сделали денойз и резкость.
    z = min(pv.zoom, *done.shape[:2])
    cy, cx = done.shape[0] // 2, done.shape[1] // 2
    sl = (slice(cy - z // 2, cy - z // 2 + z), slice(cx - z // 2, cx - z // 2 + z))
    contact_sheet([np.clip(np.nan_to_num(rgb[sl]), 0, 1), done[sl]],
                  ["до финиша, 100%", f"denoise {cfg.finish.denoise}, sharpen {cfg.finish.sharpen}, 100%"],
                  out_dir / "finish_zoom.png", cols=2)
    return {"files": files, "shape": list(done.shape[:2]),
            "previews": ["finish_preview.png", "finish_zoom.png"]}
