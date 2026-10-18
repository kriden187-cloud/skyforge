"""Этап clean: между совмещением и растяжкой чинит то, что видно на картинке, но не в коде.

Сначала заполняются насыщенные ядра звёзд (stars.py), затем вычитается
крупномасштабная модель фона (background.py). Результат кешируется в
data/work/<slug>/bin<N>/clean/. Если оба механизма выключены, этап ничего не пишет,
и растяжка берёт совмещённые кадры как есть.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from skyforge.align import load_aligned, load_tiles, read_aligned_meta
from skyforge.background import background_model, tile_pedestal
from skyforge.compose import compose_layers
from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, quick_norm, shrink
from skyforge.stars import core_area_px, fill_rings, find_core_holes
from skyforge.stretch import stretch_array

# Превью фона тянется жёстче рабочей растяжки: стыки плиток — это проценты от шума,
# при обычной растяжке их не видно, а на итоге с lift они вылезают.
REVEAL_BLACK, REVEAL_WHITE, REVEAL_A = 1.0, 90.0, 0.02


def enabled(cfg: Config) -> bool:
    return cfg.stars.fill_cores or cfg.background.subtract


def clean_dir(cfg: Config, paths: Paths) -> Path:
    return paths.work(cfg.slug, cfg.bin) / "clean"


def input_key(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Ключ данных, которые идут в растяжку: совмещённые кадры плюс параметры clean."""
    key: dict[str, Any] = {"align": read_aligned_meta(cfg, paths)["key"]}
    if enabled(cfg):
        key["stars"] = cfg.stars.model_dump(mode="json")
        key["background"] = cfg.background.model_dump(mode="json")
    return key


def load_input(cfg: Config, paths: Paths) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays, _, _ = load_aligned(cfg, paths)
    key = input_key(cfg, paths)
    if not enabled(cfg):
        return arrays, key
    d = clean_dir(cfg, paths)
    meta_path = d / "meta.json"
    if not meta_path.exists() or json.loads(meta_path.read_text(encoding="utf-8"))["key"] != key:
        raise SkyforgeError("очистка не сделана или сделана с другими параметрами: сначала skyforge clean")
    return {f: np.load(d / f"{f}.npy") for f in arrays}, key


def clean(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    if not enabled(cfg):
        console.print("clean: stars.fill_cores и background.subtract выключены — растяжка возьмёт совмещённые кадры")
        return {"enabled": False}
    arrays, _, meta = load_aligned(cfg, paths)
    scale = meta["pixel_scale_arcsec"]
    d = clean_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    before = {f: np.asarray(a, dtype=np.float32) for f, a in arrays.items()}
    after: dict[str, np.ndarray] = {}
    holes_all = np.zeros(next(iter(before.values())).shape, dtype=bool)
    info: dict[str, Any] = {"enabled": True, "filters": {}}
    models, masks = {}, {}
    for f, a in before.items():
        out = a
        fi: dict[str, Any] = {}
        if cfg.stars.fill_cores:
            holes = find_core_holes(out, core_area_px(cfg.stars.max_core_arcsec, scale))
            out = fill_rings(out, holes)
            holes_all |= holes
            fi["core_pixels"] = int(holes.sum())
            fi["cores"] = int(ndimage.label(holes, structure=np.ones((3, 3)))[1])
        if cfg.background.subtract:
            bc = cfg.background
            model = np.zeros_like(out)
            tiles = load_tiles(cfg, paths, f) if bc.tiles else None
            if bc.tiles and tiles is None:
                console.print(f"  [yellow]{f}: в файле нет слоя CON/CTX — плитки не выравниваю[/]")
            if tiles is not None:
                ped, ti = tile_pedestal(out, tiles, bc, scale)
                out = out - np.nan_to_num(ped)
                model += np.nan_to_num(ped)
                fi.update(tiles=ti["tiles"], tiles_measured=ti["measured"])
            mask = np.zeros(out.shape, dtype=bool)
            if bc.smooth:
                smooth, mask, box = background_model(out, bc, scale)
                out = out - smooth
                model += np.nan_to_num(smooth)
                fi["box_px"] = box
                fi["masked_percent"] = round(float(mask.mean() * 100), 1)
            models[f], masks[f] = model, mask
        after[f] = out.astype(np.float32)
        np.save(d / f"{f}.npy", after[f])
        info["filters"][f] = fi
        parts = []
        if "cores" in fi:
            parts.append(f"ядер {fi['cores']} ({fi['core_pixels']} px)")
        if "tiles" in fi:
            parts.append(f"плиток {fi['tiles']}, измерено {fi['tiles_measured']}")
        if "box_px" in fi:
            parts.append(f"гладкий фон: блок {fi['box_px']} px, маска {fi['masked_percent']}%")
        console.print(f"  {f}: " + ", ".join(parts))

    (d / "meta.json").write_text(json.dumps({"key": input_key(cfg, paths), **info}, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    previews = []
    if cfg.background.subtract:
        _background_sheet(cfg, before, after, models, masks, out_dir / "background_model.png")
        previews.append("background_model.png")
    if cfg.stars.fill_cores and holes_all.any():
        _core_sheet(cfg, before, after, holes_all, out_dir / "star_cores.png")
        previews.append("star_cores.png")
    info["previews"] = previews
    return info


def _background_sheet(cfg: Config, before: dict, after: dict, models: dict, masks: dict, path: Path) -> None:
    size = cfg.preview.max_size // 4
    images, labels = [], []
    for f in models:
        b = shrink(before[f], size)
        a = shrink(after[f], size)
        m = shrink(models[f], size)
        lo, hi = np.nanpercentile(m, [1, 99])
        images += [quick_norm(b, REVEAL_BLACK, REVEAL_WHITE, REVEAL_A),
                   shrink(masks[f].astype(np.float32), size),
                   np.clip((m - lo) / (hi - lo if hi > lo else 1), 0, 1),
                   quick_norm(a, REVEAL_BLACK, REVEAL_WHITE, REVEAL_A)]
        labels += [f"{f}: до", "маска источников", "модель фона", "после вычитания"]
    contact_sheet(images, labels, path, cols=4, tile_inches=3.0,
                  title="Фон: стыки плиток и градиенты должны уйти в «модель», туманность — остаться")


def _core_sheet(cfg: Config, before: dict, after: dict, holes: np.ndarray, path: Path) -> None:
    """Кроп 100% вокруг самой большой дыры: цветной предпросмотр до и после заполнения."""
    labels, n = ndimage.label(holes, structure=np.ones((3, 3)))
    sizes = ndimage.sum_labels(holes, labels, index=np.arange(1, n + 1))
    cy, cx = (int(round(v)) for v in ndimage.center_of_mass(holes, labels, int(np.argmax(sizes)) + 1))
    z = cfg.preview.core_zoom
    h, w = holes.shape
    y0, x0 = int(np.clip(cy - z // 2, 0, max(h - z, 0))), int(np.clip(cx - z // 2, 0, max(w - z, 0)))
    sl = (slice(y0, y0 + z), slice(x0, x0 + z))
    # Предпросмотр без нейтрализации фона: её статистика по крошечному кропу была бы случайной.
    c = cfg.model_copy(update={"color": cfg.color.model_copy(update={"neutral_background": False})})
    renders = []
    for src in (before, after):
        layers = {f: stretch_array(a[sl], cfg.stretch.for_filter(f), stats_from=a)[0] for f, a in src.items()}
        renders.append(compose_layers(c, layers)[0])
    contact_sheet(renders, ["до: дыры насыщения пустые", "после: ядро заполнено кольцами"], path, cols=2,
                  title="Ядро самой насыщенной звезды, 100%: должно быть белым, без цветной точки",
                  interpolation="nearest")
