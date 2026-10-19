"""Растяжка: процентиль фона → 0, процентиль ярких деталей → 1, затем нелинейная кривая.

Каждый фильтр нормируется отдельно, поэтому разный уровень фона и единицы
(MJy/sr у JWST, e⁻/s у Hubble) не перекрашивают картинку.
NaN сохраняются до экспорта; вся статистика считается через nan-функции.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from skyforge.config import Config, StretchConfig
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, save_png, shrink

# Статистика по прореженной сетке пикселей: детерминированно и в сотни раз быстрее сортировки всей мозаики.
STATS_SAMPLE_PIXELS = 4_000_000
HIST_BINS = 256


def sample(a: np.ndarray) -> np.ndarray:
    step = max(1, math.ceil(math.sqrt(a.size / STATS_SAMPLE_PIXELS)))
    return np.asarray(a[::step, ::step], dtype=np.float32)


def curve(x: np.ndarray, method: str, a: float) -> np.ndarray:
    if method == "asinh":
        return np.arcsinh(x / a) / np.arcsinh(1 / a)
    if method == "log":
        return np.log1p(x / a) / np.log1p(1 / a)
    if method == "sqrt":
        return np.sqrt(x)
    return x


def masked_sample(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Выборка на той же решётке, что и sample(): шаг зависит от размера кадра, а не от числа пикселей маски —
    # иначе изменение маски на краю сдвигало бы всю выборку и процентили по всему кадру (3h, п.4).
    v = sample(a)[sample(np.asarray(mask, dtype=np.float32)) > 0.5]
    return v[np.isfinite(v)]


def stretch_array(arr: np.ndarray, s: StretchConfig, stats_from: np.ndarray | None = None,
                  masks: tuple[np.ndarray, np.ndarray] | None = None,
                  field: Any = None) -> tuple[np.ndarray, dict[str, float]]:
    """stats_from — полный кадр, если arr его уменьшенная копия (для sweep).
    masks — (пустое небо, туманность) на сетке stats_from: при anchor masks чёрная точка берётся
    по небу, белая — по туманности."""
    src = arr if stats_from is None else stats_from
    if field is not None:  # поле статистики (skyforge.field): процентили — только там, где есть все фильтры
        if masks is not None:
            masks = (field.mask(masks[0]), field.mask(masks[1]))
        src = field.values(src)
    smp = sample(src)
    if not np.isfinite(smp).any():
        raise SkyforgeError("в кадре нет ни одного числа — только NaN")
    anchor = "frame"
    if s.anchor == "masks" and masks is not None:
        sky_v, neb_v = masked_sample(src, masks[0]), masked_sample(src, masks[1])
        if sky_v.size > 100 and neb_v.size > 100:
            lo = float(np.percentile(sky_v, s.black))
            hi = float(np.percentile(neb_v, s.white))
            anchor = "masks"
    if anchor == "frame":
        lo, hi = (float(v) for v in np.nanpercentile(smp, [s.black, s.white]))
    if not hi > lo:
        raise SkyforgeError(f"процентили black={s.black} и white={s.white} дают одно значение {lo}: кадр пустой?")
    x = (np.asarray(arr, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo)
    np.clip(x, 0, 1, out=x)  # NaN остаются NaN
    y = curve(x, s.method, s.a).astype(np.float32)
    stats = {
        "black_value": lo,
        "white_value": hi,
        "white_clipped_percent": round(float(np.nanmean(smp >= hi) * 100), 3),
        "median_after": round(float(np.nanmedian(sample(y))), 4),
        "anchor": anchor,
    }
    return y, stats


def stretched_dir(cfg: Config, paths: Paths) -> Path:
    return paths.style_work(cfg.slug, cfg.bin, cfg.style) / "stretched"


def stretch_key(cfg: Config, in_key: dict[str, Any]) -> dict[str, Any]:
    key = {"input": in_key, "stretch": cfg.stretch.model_dump(mode="json")}
    if cfg.stretch.anchor == "masks":
        key["sky_grow_px"] = cfg.color.sky_grow_px  # маска неба зависит от отступа
    return key


def load_masks(cfg: Config, paths: Paths) -> tuple[np.ndarray, np.ndarray] | None:
    """(пустое небо, туманность) с этапа clean; None, если масок нет."""
    from skyforge.clean import load_nebula_sd, load_sky_mask

    sky, sd = load_sky_mask(cfg, paths), load_nebula_sd(cfg, paths)
    if sky is None or sd is None:
        return None
    return sky, sd > 0


def stretch_all(cfg: Config, arrays: dict[str, np.ndarray],
                masks: tuple[np.ndarray, np.ndarray] | None = None,
                field: Any = None) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    out, stats = {}, {}
    for f, a in arrays.items():
        out[f], stats[f] = stretch_array(a, cfg.stretch.for_filter(f), masks=masks, field=field)
    return out, stats


def stretch(cfg: Config, paths: Paths, out_dir: Path) -> dict[str, Any]:
    from skyforge.clean import load_input  # clean сам использует растяжку для превью

    arrays, in_key = load_input(cfg, paths)
    if not arrays:
        raise SkyforgeError("среди совмещённых кадров нет фильтров из конфига")
    d = stretched_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    masks = load_masks(cfg, paths) if cfg.stretch.anchor == "masks" else None
    if cfg.stretch.anchor == "masks" and masks is None:
        console.print("[yellow]  stretch.anchor: masks — масок нет (clean выключен), растяжка по всему кадру[/]")
    from skyforge.align import stats_field

    stretched, stats = stretch_all(cfg, arrays, masks, stats_field(cfg, paths, list(arrays.values())))
    for f, y in stretched.items():
        np.save(d / f"{f}.npy", y)
        s = cfg.stretch.for_filter(f)
        console.print(f"  {f}: фон {stats[f]['black_value']:.4g} → 0, белая точка {stats[f]['white_value']:.4g} → 1, "
                      f"{s.method} a={s.a}, по {'маскам' if stats[f]['anchor'] == 'masks' else 'кадру'}, "
                      f"выжжено {stats[f]['white_clipped_percent']}%")
    (d / "meta.json").write_text(json.dumps({"key": stretch_key(cfg, in_key), "filters": list(stretched), "stats": stats},
                                            ensure_ascii=False, indent=2), encoding="utf-8")

    pv = cfg.preview.max_size
    thumbs = {f: shrink(y, pv) for f, y in stretched.items()}
    for f, t in thumbs.items():
        save_png(t, out_dir / f"stretch_{f}.png")
    contact_sheet(list(thumbs.values()), list(thumbs), out_dir / "stretch_sheet.png",
                  title=f"После растяжки ({cfg.stretch.method})")
    _hist_sheet(arrays, stretched, stats, out_dir / "stretch_hist.png")
    return {"stats": stats, "previews": ["stretch_hist.png", "stretch_sheet.png"]}


def _hist_sheet(raw: dict[str, np.ndarray], done: dict[str, np.ndarray], stats: dict[str, dict], path: Path) -> None:
    n = len(raw)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.3 * n), squeeze=False)
    for i, f in enumerate(raw):
        r = sample(raw[f])
        r = r[np.isfinite(r)]
        lo, hi = stats[f]["black_value"], stats[f]["white_value"]
        span = hi - lo
        ax = axes[i, 0]
        ax.hist(r, bins=HIST_BINS, range=(lo - 0.5 * span, hi + 0.5 * span), color="#4a7fd1", log=True)
        ax.axvline(lo, color="k", lw=1, ls="--")
        ax.axvline(hi, color="#d14a4a", lw=1, ls="--")
        ax.set_ylabel(f, fontsize=11, fontweight="bold")
        if i == 0:
            ax.set_title("до: исходные значения, пунктир — чёрная и белая точки")
        s = sample(done[f])
        ax = axes[i, 1]
        ax.hist(s[np.isfinite(s)], bins=HIST_BINS, range=(0, 1), color="#d18a4a", log=True)
        if i == 0:
            ax.set_title("после: 0–1, пик фона у нуля, хвост вправо")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
