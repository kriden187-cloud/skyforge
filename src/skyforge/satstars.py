"""Звёзды с насыщенным ядром отдельно от туманности до растяжки (3i).

В F090W центральная звезда NGC 3132 в сотни раз ярче туманности рядом: растяжка туманности делает её
белым пятном на сотни пикселей. Здесь из линейных кадров (после clean) вычитается слой звезды — одна
радиальная форма на все фильтры и один цвет на звезду, — туманность растягивается как обычно (точки
растяжки считаются по исходным кадрам, они не меняются), а звезда — своей кривой с белой точкой по ядру.
Слои складываются после растяжки «экраном»: 1 − (1 − туманность)·(1 − звезда).

Форма: радиальный профиль (нижний процентиль по кольцам — туманность и лучи его не завышают) суммы
фильтров, каждый в единицах белой точки туманности своего фильтра, над уровнем на внешнем кольце;
до r_in целиком, к r_out плавно в ноль. Цвет: отношение избытка над туманностью по фильтрам в кольце
color_ring (там нет заполненных ядер).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from skyforge.stretch import curve


def _taper(r: np.ndarray, r_in: float, r_out: float) -> np.ndarray:
    t = np.where(r < r_in, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((r - r_in) / (r_out - r_in), 0, 1)))
    return np.where(r >= r_out, 0.0, t)


def star_layers(arrays: dict[str, np.ndarray], whites: dict[str, float], stars: list[list[int]],
                c: Any, bin_: int = 1) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    """Линейный слой звёзд по фильтрам (для вычитания) и показ звёзд (0–1) по фильтрам после растяжки.
    stars — [строка, столбец, площадь, фильтров] на сетке; c — StretchConfig.saturated_stars."""
    shape = next(iter(arrays.values())).shape
    lin = {f: np.zeros(shape, np.float32) for f in arrays}
    disp = {f: np.zeros(shape, np.float32) for f in arrays}
    r_in, r_out = c.r_in_px / bin_, c.r_out_px / bin_
    rc, rf = c.core_r_px / bin_, c.core_feather_px / bin_
    ring0, ring1 = c.color_ring_px[0] / bin_, c.color_ring_px[1] / bin_
    info = []
    R = int(np.ceil(r_out))
    for row, col, _, _ in stars:
        sl = (slice(max(row - R, 0), min(row + R + 1, shape[0])), slice(max(col - R, 0), min(col + R + 1, shape[1])))
        yy, xx = np.indices((sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))
        r = np.hypot(yy + sl[0].start - row, xx + sl[1].start - col)
        ri = np.floor(r).astype(int)
        outer = (r >= r_out - 20 / bin_) & (r < r_out)
        norm, core_level = {}, {}
        core_ring = (r >= rc) & (r < rc + rf)
        for f, a in arrays.items():
            sub = a[sl]
            ok = np.isfinite(sub)
            level = float(np.median(sub[ok & outer])) if (ok & outer).any() else 0.0
            norm[f] = np.where(ok, (sub - level) / whites[f], np.nan)
            # туманность внутри ядра — ровная, уровень кольца сразу снаружи ядра
            core_level[f] = float(np.median(sub[ok & core_ring])) if rc > 0 and (ok & core_ring).any() else level
        comb = np.nanmean(np.stack(list(norm.values())), axis=0)
        prof = np.zeros(R + 1)
        for k in range(R + 1):
            v = comb[(ri == k) & np.isfinite(comb)]
            prof[k] = np.percentile(v, c.profile_percentile) if v.size else 0.0
        prof = np.clip(ndimage.median_filter(prof, size=3, mode="nearest"), 0, None)
        shape_s = prof[np.minimum(ri, R)] * _taper(r, r_in, r_out)          # форма, в единицах белых точек
        beta = (np.where(r < rc, 1.0, 0.5 + 0.5 * np.cos(np.pi * np.clip((r - rc) / rf, 0, 1)))
                * (r < rc + rf)) if rc > 0 else np.zeros_like(r)
        core_s = np.nanmean(np.stack([(arrays[f][sl] - core_level[f]) / whites[f] for f in arrays]), axis=0)
        shape_s = beta * np.clip(np.nan_to_num(core_s), 0, None) + (1 - beta) * shape_s
        ring = (r >= ring0) & (r < ring1)
        e = {f: max(float(np.nanmedian(norm[f][ring])), 0.0) for f in arrays}
        tot = sum(e.values()) or 1.0
        w = {f: e[f] / tot * len(arrays) for f in arrays}                    # доля фильтра в форме (среднее = 1)
        peak = float(shape_s.max()) or 1.0
        tone = curve(np.clip(shape_s / peak, 0, 1), "asinh", c.a).astype(np.float32)
        kmax = max(w.values()) or 1.0
        radial = prof[np.minimum(ri, R)] * _taper(r, r_in, r_out)
        for f in arrays:
            # вычитаемое: в ядре — всё выше ровной туманности, снаружи — радиальная модель с цветом звезды
            inside = np.nan_to_num(arrays[f][sl] - core_level[f])
            lin[f][sl] += (beta * inside + (1 - beta) * radial * w[f] * whites[f]).astype(np.float32)
            disp[f][sl] = 1 - (1 - disp[f][sl]) * (1 - tone * np.float32(w[f] / kmax))
        info.append({"row": row, "col": col, "peak_in_whites": round(peak, 1),
                     "color": {f: round(w[f] / kmax, 3) for f in arrays}})
    return lin, disp, info


def screen(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = (1 - (1 - np.nan_to_num(a)) * (1 - b)).astype(np.float32)
    out[~np.isfinite(a)] = np.nan
    return out
