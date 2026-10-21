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


def curve_inv(y: float, method: str, a: float) -> float:
    """Обратная к curve: какое значение до растяжки даёт y после."""
    if method == "asinh":
        return float(a * math.sinh(y * math.asinh(1 / a)))
    if method == "log":
        return float(a * math.expm1(y * math.log1p(1 / a)))
    if method == "sqrt":
        return float(y * y)
    return float(y)


def sky_toe(x: np.ndarray, u_sky: float, u_target: float, max_p: float | None = None) -> tuple[np.ndarray, float]:
    """Носок до растяжки (x в 0–1): u_sky → u_target. Выше u_sky — прямая через (u_sky, u_target) и (1, 1),
    то есть линейно по данным, как чёрная точка; ниже — u_target·(x/u_sky)^p, p подобрано так, что наклон
    в u_sky совпадает с прямой: кривая гладкая, монотонная, 0 → 0, без обрезки. Возвращает (x, p).
    max_p (фаза 4, stretch.sky_toe_max_p): при сильной растяжке и небе высоко по шкале p ~ 100–340 — «стенка»: шум
    чуть ниже неба уходит в 0. С ограничением значение у неба то же, ниже неба кривая положе (излом наклона на уровне
    неба); совпадение и значения, и наклона при крутом наклоне всё равно дало бы ноль в u_target/q ниже неба."""
    if not u_sky > u_target > 0:
        return x, 1.0
    q = (1 - u_target) / (1 - u_sky)
    p = q * u_sky / u_target
    if max_p is not None:
        p = min(p, max_p)
    low = u_target * np.power(np.maximum(x, 0) / np.float32(u_sky), np.float32(p))
    out = np.where(x >= u_sky, u_target + (x - u_sky) * np.float32(q), low)
    return out.astype(np.float32), float(p)


def masked_sample(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # Выборка на той же решётке, что и sample(): шаг зависит от размера кадра, а не от числа пикселей маски —
    # иначе изменение маски на краю сдвигало бы всю выборку и процентили по всему кадру (3h, п.4).
    v = sample(a)[sample(np.asarray(mask, dtype=np.float32)) > 0.5]
    return v[np.isfinite(v)]


def noisy_sigma(a: np.ndarray, sky: np.ndarray, block: int) -> float:
    """σ неба самых шумных мест: робастный шум (MAD) в блоках block×block, где неба не меньше половины;
    98-й процентиль по блокам."""
    h, w = a.shape
    sig = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            m = sky[y:y + block, x:x + block]
            if m.mean() < 0.5:
                continue
            v = a[y:y + block, x:x + block][m]
            v = v[np.isfinite(v)]
            if v.size > 100:
                sig.append(1.4826 * float(np.median(np.abs(v - np.median(v)))))
    return float(np.percentile(sig, 98)) if sig else 0.0


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
            sky_med = float(np.median(sky_v))
            if s.black_mode == "pedestal":
                lo = sky_med - s.black_k * noisy_sigma(src, masks[0], s.noise_block_px)
                anchor = "masks, пьедестал"
    if anchor == "frame":
        lo, hi = (float(v) for v in np.nanpercentile(smp, [s.black, s.white]))
    if not hi > lo:
        raise SkyforgeError(f"процентили black={s.black} и white={s.white} дают одно значение {lo}: кадр пустой?")
    x = (np.asarray(arr, dtype=np.float32) - np.float32(lo)) / np.float32(hi - lo)
    np.clip(x, 0, 1, out=x)  # NaN остаются NaN
    toe = None
    if s.sky_level is not None and anchor != "frame":
        u_sky = (sky_med - lo) / (hi - lo)
        u_t = curve_inv(s.sky_level / 255, s.method, s.a)
        x, p = sky_toe(x, u_sky, u_t, s.sky_toe_max_p)
        toe = {"u_sky": round(u_sky, 6), "u_target": round(u_t, 6), "p": round(p, 3)}
    y = curve(x, s.method, s.a).astype(np.float32)
    stats = {
        "black_value": lo,
        "white_value": hi,
        "white_clipped_percent": round(float(np.nanmean(smp >= hi) * 100), 3),
        "median_after": round(float(np.nanmedian(sample(y))), 4),
        "anchor": anchor,
    }
    if toe is not None:
        stats["sky_toe"] = toe
    return y, stats


def stretched_dir(cfg: Config, paths: Paths) -> Path:
    return paths.style_work(cfg.slug, cfg.bin, cfg.style) / "stretched"


def stretch_key(cfg: Config, in_key: dict[str, Any]) -> dict[str, Any]:
    exclude = set() if cfg.stretch.sky_toe_max_p is not None else {"sky_toe_max_p"}
    if cfg.stretch.sky_reference == "mask":   # шаг 4d: поля опоры — только когда она не по умолчанию
        exclude |= {"sky_reference", "sky_reference_block_px", "sky_reference_n_blocks", "sky_reference_min_free"}
    if cfg.stretch.sky_reference_crop is None:
        exclude.add("sky_reference_crop")
    st = cfg.stretch.model_dump(mode="json", exclude=exclude)
    key = {"input": in_key, "stretch": st}   # фаза 4: новые поля в ключе только когда заданы — кеши прежних объектов те же
    if cfg.stretch.sky_reference == "crop":   # фаза 6: координаты кропа опоры (report в хеш не входит)
        key["sky_reference_crop_spec"] = next(c.model_dump(mode="json") for c in cfg.report.crops
                                              if c.name == cfg.stretch.sky_reference_crop)
    if cfg.clean.fill_gaps_px > 0:
        key["fill_gaps_px"] = cfg.clean.fill_gaps_px
    if cfg.stretch.sky_poly_order > 0:
        from skyforge.skymodel import SKY_POLY_VERSION

        key["sky_poly_version"] = SKY_POLY_VERSION
    if cfg.stretch.anchor == "masks":
        key["sky_grow_px"] = cfg.color.sky_grow_px  # маска неба зависит от отступа
    return key


def darkest_blocks(a: np.ndarray, ok: np.ndarray, block: int, n: int, min_free: float) -> tuple[np.ndarray, dict]:
    """Шаг 4d: n блоков block×block с самой низкой медианой a по пикселям ok (доля ok в блоке ≥ min_free).
    Возвращает маску опорного неба (ok внутри выбранных блоков) и {blocks: [[строка, столбец, медиана]], eligible, total}."""
    H, W = a.shape
    cand, total = [], 0
    for y in range(0, H - block + 1, block):
        for x in range(0, W - block + 1, block):
            total += 1
            m = ok[y:y + block, x:x + block]
            if m.mean() >= min_free:
                cand.append((float(np.median(a[y:y + block, x:x + block][m])), y, x))
    if len(cand) < n:
        raise SkyforgeError(f"stretch.sky_reference: блоков с долей чистых пикселей ≥ {min_free} — {len(cand)} из {total}, "
                            f"нужно {n}: уменьши sky_reference_block_px или sky_reference_min_free")
    cand.sort()
    ref = np.zeros((H, W), dtype=bool)
    for _, y, x in cand[:n]:
        ref[y:y + block, x:x + block] = ok[y:y + block, x:x + block]
    return ref, {"blocks": [[y, x, round(v, 6)] for v, y, x in cand[:n]], "eligible": len(cand), "total": total}


def reference_sky(cfg: Config, paths: Paths, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Опорное небо по самым тёмным блокам опорного фильтра (cfg.reference) после clean: чистые пиксели — поле всех
    фильтров вне маски звёзд clean (с отступом 3 px)."""
    from scipy import ndimage

    from skyforge.clean import clean_dir

    d = clean_dir(cfg, paths)
    if not cfg.reference:
        raise SkyforgeError("stretch.sky_reference: darkest_blocks — нужен reference (опорный фильтр) в конфиге")
    full = np.ones(shape, dtype=bool)
    for f in cfg.filters:
        full &= np.isfinite(np.load(d / f"{f}.npy", mmap_mode="r"))
    stars = ndimage.binary_dilation(np.load(d / "star_weight.npy") > 0.01, iterations=3)
    s = cfg.stretch
    if s.sky_reference == "crop":
        from skyforge.align import with_resolved_crops
        from skyforge.preview import display_crop

        k = next(c for c in with_resolved_crops(cfg, paths).report.crops if c.name == s.sky_reference_crop)
        sl = display_crop(shape, k.x, k.y, k.size, cfg.bin)
        ref = np.zeros(shape, dtype=bool)
        ref[sl] = (full & ~stars)[sl]
        if ref.sum() < 0.25 * ref[sl].size:
            console.print(f"[yellow]  опорное небо: в кропе {k.name} чистых пикселей {ref.sum()} из {ref[sl].size}[/]")
        return ref, {"crop": k.name, "rows": [sl[0].start, sl[0].stop], "cols": [sl[1].start, sl[1].stop],
                     "pixels": int(ref.sum())}
    ref, info = darkest_blocks(np.load(d / f"{cfg.reference}.npy", mmap_mode="r"), full & ~stars,
                               max(16, s.sky_reference_block_px // cfg.bin), s.sky_reference_n_blocks, s.sky_reference_min_free)
    if info["eligible"] < 0.25 * info["total"]:
        console.print(f"[yellow]  опорное небо: блоков без звёзд {info['eligible']} из {info['total']} — опора может уйти "
                      f"в туманность (шаг 4d); уменьши sky_reference_min_free[/]")
    return ref, info


def load_masks(cfg: Config, paths: Paths) -> tuple[np.ndarray, np.ndarray] | None:
    """(пустое небо, туманность) с этапа clean; None, если масок нет. stretch.sky_reference: darkest_blocks —
    вместо маски неба опора по самым тёмным блокам (шаг 4d)."""
    from skyforge.clean import load_nebula_sd, load_sky_mask

    sky, sd = load_sky_mask(cfg, paths), load_nebula_sd(cfg, paths)
    if sky is None or sd is None:
        return None
    if cfg.stretch.sky_reference in ("darkest_blocks", "crop"):
        sky, _ = reference_sky(cfg, paths, sd.shape)
    neb = sd > 0
    if cfg.stretch.exclude_stars:
        from skyforge.clean import star_exclusion

        neb &= ~star_exclusion(cfg, paths, neb.shape)  # белая точка — по туманности, а не по звёздам в ней
    return sky, neb


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
    from skyforge.skymodel import apply_sky_poly, sky_poly_sheet

    extra_pv = []
    gaps_info: dict[str, int] = {}
    if cfg.clean.fill_gaps_px > 0:  # 3б: швы чипов — в стиле, не в общем кеше clean
        from skyforge.cosmics import fill_gaps_linear

        filled_maps = {}
        for f in list(arrays):
            arrays[f], filled_maps[f] = fill_gaps_linear(arrays[f], cfg.clean.fill_gaps_px)
            gaps_info[f] = int(filled_maps[f].sum())
        console.print("  швы заполнены (полуширина ≤ " + str(cfg.clean.fill_gaps_px) + " px): "
                      + ", ".join(f"{f} {n} px" for f, n in gaps_info.items()))
        _gaps_sheet(arrays, filled_maps, out_dir / "gaps_filled.png")
        extra_pv.append("gaps_filled.png")
    arrays, poly_models, poly_info = apply_sky_poly(cfg, paths, arrays)
    if poly_models:
        sky_poly_sheet(poly_models, poly_info, out_dir / "sky_poly.png")
        extra_pv.append("sky_poly.png")
        for f in arrays:
            console.print(f"  {f}: полином неба {cfg.stretch.sky_poly_order}-го порядка — размах {poly_info[f]['range_sigma']}σ, "
                          f"в зоне исключения {poly_info[f]['in_excluded_range_sigma']}σ")
    d = stretched_dir(cfg, paths)
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.npy"):
        old.unlink()

    masks = load_masks(cfg, paths) if cfg.stretch.anchor == "masks" else None
    if cfg.stretch.anchor == "masks" and masks is None:
        console.print("[yellow]  stretch.anchor: masks — масок нет (clean выключен), растяжка по всему кадру[/]")
    from skyforge.align import stats_field

    field = stats_field(cfg, paths, list(arrays.values()))
    stretched, stats = stretch_all(cfg, arrays, masks, field)
    ss = cfg.stretch.saturated_stars
    sinfo: list = []
    if ss.enabled:
        from skyforge.clean import load_saturated_stars
        from skyforge.satstars import screen, star_layers

        stars = load_saturated_stars(cfg, paths)
        if stars:
            lin, disp, sinfo = star_layers(arrays, {f: stats[f]["white_value"] for f in arrays}, stars, ss, cfg.bin)
            for f, a in arrays.items():
                # туманность без звезды, точки растяжки — по исходному кадру (не меняются)
                neb, _ = stretch_array(a - lin[f], cfg.stretch.for_filter(f), stats_from=a, masks=masks, field=field)
                stretched[f] = screen(neb, disp[f])
            console.print(f"  звёзды с насыщенным ядром отдельно от туманности: {len(sinfo)}")
    for f, y in stretched.items():
        np.save(d / f"{f}.npy", y)
    for f in stretched:
        s = cfg.stretch.for_filter(f)
        console.print(f"  {f}: фон {stats[f]['black_value']:.4g} → 0, белая точка {stats[f]['white_value']:.4g} → 1, "
                      f"{s.method} a={s.a}, по {'маскам' if stats[f]['anchor'].startswith('masks') else 'кадру'}, "
                      f"выжжено {stats[f]['white_clipped_percent']}%")
    (d / "meta.json").write_text(json.dumps({"key": stretch_key(cfg, in_key), "filters": list(stretched), "stats": stats,
                                             "saturated_stars": sinfo},
                                            ensure_ascii=False, indent=2), encoding="utf-8")

    pv = cfg.preview.max_size
    thumbs = {f: shrink(y, pv) for f, y in stretched.items()}
    for f, t in thumbs.items():
        save_png(t, out_dir / f"stretch_{f}.png")
    contact_sheet(list(thumbs.values()), list(thumbs), out_dir / "stretch_sheet.png",
                  title=f"После растяжки ({cfg.stretch.method})")
    _hist_sheet(arrays, stretched, stats, out_dir / "stretch_hist.png")
    return {"stats": stats, "saturated_stars": sinfo, "sky_poly": poly_info, "gaps_filled": gaps_info,
            "previews": ["stretch_hist.png", "stretch_sheet.png"] + extra_pv}


def _gaps_sheet(arrays: dict[str, np.ndarray], filled: dict[str, np.ndarray], path: Path) -> None:
    """3б: карта заполненных пикселей (красные) поверх кадра опорного фильтра."""
    from skyforge.preview import quick_norm

    f0 = next(iter(arrays))
    base = quick_norm(arrays[f0], 1.0, 99.5, 0.05)
    any_f = np.logical_or.reduce(list(filled.values()))
    rgb = np.stack([base] * 3, -1)
    rgb[any_f] = [1.0, 0.1, 0.1]
    contact_sheet([rgb], [f"заполненные швы и чёрточки (красные): {int(any_f.sum())} px"], path, cols=1, tile_inches=10,
                  interpolation="nearest", title="Швы чипов: линейно поперёк шва, только где данные с обеих сторон")


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
