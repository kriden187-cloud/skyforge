"""Астрометрия экспозиций уровня 2 по Gaia DR3 — фаза 8 (level2.astrometry: gaia), вместо подстройки по звёздам i2d.

Зачем: у OMC3-N (программа 7433) визит 004 наведён по другой гидирующей звезде и сдвинут на 1,2″; tweakreg STScI это
не исправил (в i2d зона «только 004» на 1,2″ от Gaia, в перекрытии визитов фильтры двоятся). Подстройка по звёздам i2d
(фаза 3g) переносит ошибку i2d в нашу сборку.

Как (аналог tweakreg с абсолютной привязкой):
1. Звёзды в каждом кадре детектора: острые изолированные максимумы без пустых пикселей рядом (не насыщены),
   центроид — гауссом; ширина как у остальных находок кадра (узлы туманности шире).
2. Экспозиция — одно наведение: все её детекторы (короткие и длинные волны снимаются одновременно) получают одну
   поправку на небе — сдвиг, при достаточном числе звёзд и заметном угле — поворот вокруг центра экспозиции.
3. Грубый сдвиг каждой экспозиции — голосованием разностей «Gaia − звезда» в радиусе COARSE_RADIUS; экспозиция без
   голосов берёт медиану своего визита.
4. Точно — один МНК на все экспозиции всех фильтров: пары одной звезды в разных экспозициях (относительная точность)
   и звёзды Gaia DR3 на эпоху наблюдения с их ошибками (абсолютная привязка). Ошибки центроидов масштабируются по
   разбросу остатков; выбросы > CLIP σ отбрасываются.
5. Правило 3g: поправка ставится, если точность сдвига экспозиции относительно остальных (совмещение между собой —
   двоение и цвет) < level2.tweak_max_precision (px коротких волн, 0,031″); иначе — грубый сдвиг визита (статус это
   называет). Точность привязки к Gaia (общая для всех экспозиций, сдвигает только координаты) — отдельной колонкой.
Решение одно на объект (все фильтры уровня 2) и кешируется в папке сборки: astrometry.json.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import warnings

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS, FITSFixedWarning
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths

ASTRO_VERSION = 2   # 2: правило 3g — по относительной точности; Gaia без маскированных решений
SW_SCALE = 0.031          # ″/px коротких волн NIRCam — единица точности (правило 3g)
MIN_SNR = 10.0            # пик над местным фоном, в ошибках пикселя
MAX_STARS = 300           # звёзд на кадр детектора
COARSE_RADIUS = 3.0       # ″: окно грубого голосования «Gaia − звезда»
COARSE_BIN = 0.04         # ″: ячейка голосования
COARSE_MIN_VOTES = 6
MATCH_RADIUS = 0.1        # ″: пары одной звезды после грубого сдвига
GAIA_RADIUS = 0.15        # ″: звезда — Gaia после грубого сдвига
GAIA_MAX_ERR_MAS = 10.0   # ошибка положения Gaia на эпоху наблюдения (с собственным движением)
GAIA_MAX_RUWE = 1.4       # двойные и плохие решения Gaia
CENTROID_FLOOR_PX = 0.01  # нижняя граница ошибки центроида (фаза пикселя, дисторсия)
SAT_ERR_PX = 0.05         # ошибка центра насыщенной звезды по симметрии крыльев
SAT_MAX_HALF = 30         # окно симметрии насыщенной звезды, px
CLIP = 4.0
I2D_OK_ARCSEC = 0.1       # i2d годится для сравнения там, где поправки всех экспозиций не больше этого
GAIA_QUERY = ("SELECT source_id, ra, dec, ra_error, dec_error, pmra, pmdec, pmra_error, pmdec_error, ruwe, "
              "phot_g_mean_mag, ref_epoch FROM gaiadr3.gaia_source WHERE "
              "CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {r})) = 1")


def exposure_of(name: str) -> str:
    """Экспозиция без детектора: jw07433001004_03101_00001 — у коротких и длинных волн одна."""
    return re.sub(r"_nrc[ab]\w*?_cal\.fits$", "", name)


def visit_of(name: str) -> str:
    return name[:13]   # jw + программа (5) + наблюдение (3) + визит (3)


def field_center(cfg: Config) -> tuple[float, float]:
    if not cfg.coords:
        raise SkyforgeError("level2.astrometry: gaia — нужны coords (центр поля для запроса Gaia)")
    ra, dec = (float(v) for v in cfg.coords.replace(",", " ").split()[:2])
    return ra, dec


def gaia_cache(cfg: Config, paths: Paths) -> Path:
    return paths.root / "data" / "cache" / "gaia" / f"{cfg.slug}_dr3.ecsv"


def gaia_catalog(cfg: Config, paths: Paths) -> Table:
    """Gaia DR3 в круге cfg.radius вокруг coords; запрос один раз, дальше — кеш (только чтение каталога)."""
    p = gaia_cache(cfg, paths)
    if p.exists():
        return Table.read(p)
    from astropy.coordinates import Angle
    from astroquery.gaia import Gaia

    ra, dec = field_center(cfg)
    r = float(Angle(cfg.radius).deg)
    console.print(f"  Gaia DR3: запрос в круге {r * 60:.1f}′ вокруг {ra:.5f} {dec:.5f}")
    t = Gaia.launch_job(GAIA_QUERY.format(ra=ra, dec=dec, r=r)).get_results()
    p.parent.mkdir(parents=True, exist_ok=True)
    t.write(p, overwrite=True)
    return t


def gaia_at(t: Table, year: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Положения Gaia на эпоху year с собственным движением и их ошибки (″). Без собственного движения, с RUWE выше
    GAIA_MAX_RUWE и с ошибкой больше GAIA_MAX_ERR_MAS — не берутся."""
    # маскированные значения (2-параметрические решения без собственного движения, без RUWE) — NaN: в кеше ECSV
    # под маской лежат нули, и без filled такие звёзды проходили бы как «движение 0, ошибка 0»
    col = lambda k: np.ma.filled(np.ma.asarray(t[k], dtype=np.float64), np.nan)  # noqa: E731
    ra, dec, dt = col("ra"), col("dec"), year - col("ref_epoch")
    pmra, pmdec = col("pmra"), col("pmdec")
    ok = np.isfinite(pmra) & np.isfinite(pmdec) & (np.nan_to_num(col("ruwe"), nan=99) <= GAIA_MAX_RUWE)
    err = np.sqrt(col("ra_error") ** 2 + col("dec_error") ** 2
                  + (np.hypot(col("pmra_error"), col("pmdec_error")) * dt) ** 2)   # mas
    ok &= np.isfinite(err) & (err <= GAIA_MAX_ERR_MAS)
    ra2 = ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    dec2 = dec + pmdec * dt / 3.6e6
    return ra2[ok], dec2[ok], err[ok] / 1000 / math.sqrt(2)   # на ось


def _gauss_fit(img: np.ndarray, x: float, y: float, half: int) -> tuple[float, float, float] | None:
    """(x, y, σ) подгонкой 2D-Гаусса с постоянным фоном; None — не вышло или центр у края окна."""
    from scipy.optimize import curve_fit

    xi, yi = int(round(x)), int(round(y))
    w = img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)
    fin = np.isfinite(w)
    # отдельные плохие пиксели (DQ) в окне не мешают: подгонка по остальным, но не больше 10% окна
    if w.shape != (2 * half + 1, 2 * half + 1) or fin.mean() < 0.9 or not fin[half - 1:half + 2, half - 1:half + 2].all():
        return None
    yy, xx = np.indices(w.shape)
    border = np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]])
    z = w - float(np.nanmedian(border))
    yy, xx, z = yy[fin], xx[fin], z[fin]

    def g(c, a, x0, y0, sg, b):
        return a * np.exp(-((c[1] - x0) ** 2 + (c[0] - y0) ** 2) / (2 * sg**2)) + b

    try:
        p, _ = curve_fit(g, (yy, xx), z, p0=(z.max(), half, half, 1.0, 0.0), maxfev=2000)
    except Exception:
        return None
    if not (half - 1.5 <= p[1] <= half + 1.5 and half - 1.5 <= p[2] <= half + 1.5 and 0.3 < abs(p[3]) < half / 1.5):
        return None
    return xi - half + float(p[1]), yi - half + float(p[2]), abs(float(p[3]))


def _symmetric_center(img: np.ndarray, x0: float, y0: float, half: int) -> tuple[float, float] | None:
    """Центр насыщенной звезды: точка центральной симметрии крыльев PSF (ядро пустое, PSF NIRCam симметрична при
    повороте на 180°). Минимум χ² между окном и его отражением относительно центра; пустые пиксели не участвуют."""
    from scipy.optimize import minimize

    xi, yi = int(round(x0)), int(round(y0))
    w = img[yi - half:yi + half + 1, xi - half:xi + half + 1].astype(np.float64)
    if w.shape != (2 * half + 1, 2 * half + 1):
        return None
    fin = np.isfinite(w)
    if fin.mean() < 0.3:
        return None
    border = np.concatenate([w[0], w[-1], w[:, 0], w[:, -1]])
    base = float(np.nanmedian(border))
    noise = 1.4826 * float(np.nanmedian(np.abs(border - base))) + 1e-12
    z = np.where(fin, w - base, 0.0)
    m = fin.astype(np.float64)
    yy, xx = np.indices(w.shape)
    use = fin & (np.hypot(yy - half, xx - half) <= half - 2)
    zu, yu, xu = z[use], yy[use], xx[use]

    def cost(c: np.ndarray) -> float:
        ry, rx = 2 * c[0] - yu, 2 * c[1] - xu
        v = ndimage.map_coordinates(z, [ry, rx], order=1, cval=0.0)
        mv = ndimage.map_coordinates(m, [ry, rx], order=1, cval=0.0)
        good = mv > 0.999
        if good.sum() < 20:
            return 1e30
        d = zu[good] - v[good]
        return float(np.sum(d * d / (np.abs(zu[good]) + np.abs(v[good]) + noise)) / good.sum())

    start = np.array([half + (y0 - yi), half + (x0 - xi)])
    res = minimize(cost, start, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-12, "maxiter": 400})
    cy, cx = res.x
    if not res.success or abs(cy - half) > 2.5 or abs(cx - half) > 2.5:
        return None
    return xi - half + float(cx), yi - half + float(cy)


def detect_stars(sci: np.ndarray, err: np.ndarray, saturated: np.ndarray | None = None,
                 max_stars: int = MAX_STARS) -> np.ndarray:
    """Звёзды кадра детектора: (x, y, ошибка центроида px, насыщена 0/1).
    Обычные — острые изолированные максимумы > MIN_SNR без пустых пикселей вплотную к пику, центроид гауссом,
    ширина 0,6–1,6 медианы находок кадра (узлы туманности шире). Насыщенные (флаг SATURATED в DQ) — центр по
    симметрии крыльев: так находятся звёзды Gaia, почти все они в NIRCam насыщены."""
    from skyforge.field import block_median_map

    ok = np.isfinite(sci) & np.isfinite(err) & (err > 0)
    bg = block_median_map(np.where(ok, sci, np.nan), 64, fill_empty=True)
    r = np.where(ok, sci - bg, np.nan).astype(np.float32)
    r0 = np.nan_to_num(r)
    snr = np.where(ok, r0 / np.where(ok, err, 1.0), 0.0)
    sm = ndimage.gaussian_filter(r0, 1.0)
    near_bad = ndimage.maximum_filter((~ok).astype(np.uint8), 3) > 0
    h, w = sci.shape
    edge = np.zeros_like(ok)
    edge[10:h - 10, 10:w - 10] = True
    pk = np.argwhere((sm == ndimage.maximum_filter(sm, 9)) & (snr > MIN_SNR) & ~near_bad & edge)
    pk = pk[np.argsort(-sm[pk[:, 0], pk[:, 1]])][: 4 * max_stars]
    loc = sm == ndimage.maximum_filter(sm, 5)
    out = []
    for y, x in pk:
        win = sm[y - 8:y + 9, x - 8:x + 9]
        if (loc[y - 8:y + 9, x - 8:x + 9] & (win > 0.2 * sm[y, x])).sum() > 1:
            continue   # соседка ближе 8 px
        f = _gauss_fit(r, float(x), float(y), 4)
        if f is not None:
            out.append((f[0], f[1], f[2], float(snr[y, x])))
    a = np.asarray(out, dtype=np.float64).reshape(-1, 4)
    if len(a) >= 5:
        med = float(np.median(a[:, 2]))
        a = a[(a[:, 2] > 0.6 * med) & (a[:, 2] < 1.6 * med)]
    a = a[:max_stars]
    stars = np.column_stack([a[:, 0], a[:, 1], np.maximum(a[:, 2] / (2.5 * a[:, 3]), CENTROID_FLOOR_PX),
                             np.zeros(len(a))])
    sat = []
    if saturated is not None and saturated.any():
        lab, n = ndimage.label(saturated)
        objs = ndimage.find_objects(lab)
        med_err = float(np.nanmedian(np.where(ok, err, np.nan)))
        for k, sl in enumerate(objs, start=1):
            size = int((lab[sl] == k).sum())
            if size < 2:
                continue
            cy, cx = ndimage.center_of_mass(lab[sl] == k)
            cy, cx = cy + sl[0].start, cx + sl[1].start
            rb = math.sqrt(size / math.pi)
            half = int(min(SAT_MAX_HALF, max(8, 4 * rb + 4)))
            if not (half + 2 <= cy < h - half - 2 and half + 2 <= cx < w - half - 2):
                continue
            yy, xx = np.ogrid[-half:half + 1, -half:half + 1]
            ring = r[int(cy) - half:int(cy) + half + 1, int(cx) - half:int(cx) + half + 1][
                (np.hypot(yy, xx) > rb + 1) & (np.hypot(yy, xx) < rb + 4)]
            if not np.isfinite(ring).any() or np.nanmedian(ring) < 20 * med_err:
                continue   # не звезда: скопление плохих пикселей без крыльев
            others = lab[int(cy) - half:int(cy) + half + 1, int(cx) - half:int(cx) + half + 1]
            if len(np.unique(others[others > 0])) > 1:
                continue   # другая насыщенная звезда в окне
            c = _symmetric_center(r, cx, cy, half)
            if c is not None:
                sat.append((c[0], c[1], SAT_ERR_PX, 1.0))
    if sat:
        s = np.asarray(sat)
        # насыщенная звезда с конечными значениями в ядре могла пройти и как обычная — оставить одну находку
        if len(stars):
            near = np.hypot(stars[:, None, 0] - s[None, :, 0], stars[:, None, 1] - s[None, :, 1]).min(axis=1) < 3
            stars = stars[~near]
        stars = np.vstack([stars, s])
    return stars


def _plane(ra: np.ndarray, dec: np.ndarray, ra0: float, dec0: float) -> np.ndarray:
    """Касательная плоскость поля, ″ (ξ на восток, η на север); линейно — нужны только разности близких точек."""
    return np.column_stack([(np.asarray(ra) - ra0) * math.cos(math.radians(dec0)) * 3600,
                            (np.asarray(dec) - dec0) * 3600])


def apply_pointing(wcs: WCS, corr: dict[str, Any]) -> WCS:
    """WCS детектора с поправкой экспозиции: точка неба p → c + R(θ)(p − c) + s в касательной плоскости поля.
    Сдвиг — через CRVAL, поворот — CD' = R(θ)·CD (промежуточные координаты: x на восток, y на север); SIP — как был."""
    w = wcs.deepcopy()
    ra0, dec0 = corr["origin"]
    sx, sy = corr["shift"]
    th = corr.get("theta", 0.0)
    cx, cy = corr.get("center", (0.0, 0.0))
    cos0 = math.cos(math.radians(dec0))
    xi = (w.wcs.crval[0] - ra0) * cos0 * 3600
    eta = (w.wcs.crval[1] - dec0) * 3600
    xi2 = cx + math.cos(th) * (xi - cx) - math.sin(th) * (eta - cy) + sx
    eta2 = cy + math.sin(th) * (xi - cx) + math.cos(th) * (eta - cy) + sy
    w.wcs.crval = [ra0 + xi2 / 3600 / cos0, dec0 + eta2 / 3600]
    if th:
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        if w.wcs.has_cd():
            w.wcs.cd = rot @ np.asarray(w.wcs.cd)
        else:   # PC+CDELT: у wcslib PC главнее CD — поворот в PC, масштаб в CDELT
            w.wcs.pc = rot @ np.diag(w.wcs.cdelt) @ w.wcs.get_pc()
            w.wcs.cdelt = [1.0, 1.0]
    w.wcs.set()
    return w


def _coarse(p: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, int, int] | None:
    """Сдвиг «Gaia − звезда» голосованием в окне COARSE_RADIUS: (сдвиг, голосов в пике, во втором пике).
    None — пик меньше COARSE_MIN_VOTES или не вдвое выше второго (случайные совпадения)."""
    if not len(p) or not len(g):
        return None
    near = cKDTree(g).query_ball_point(p, COARSE_RADIUS)
    d = np.array([g[j] - p[i] for i, js in enumerate(near) for j in js]).reshape(-1, 2)
    if len(d) < COARSE_MIN_VOTES:
        return None
    nb = int(round(2 * COARSE_RADIUS / COARSE_BIN))
    hist, ex, ey = np.histogram2d(d[:, 0], d[:, 1], bins=nb, range=[[-COARSE_RADIUS, COARSE_RADIUS]] * 2)
    hs = ndimage.uniform_filter(hist, 3, mode="constant") * 9
    i, j = np.unravel_index(np.argmax(hs), hs.shape)
    peak = int(round(hs[i, j]))
    rest = hs.copy()
    rest[max(i - 3, 0):i + 4, max(j - 3, 0):j + 4] = 0
    second = int(round(rest.max()))
    if peak < COARSE_MIN_VOTES or peak < 2 * second:
        return None
    c = np.array([(ex[i] + ex[i + 1]) / 2, (ey[j] + ey[j + 1]) / 2])
    sel = np.hypot(*(d - c).T) < 2.5 * COARSE_BIN
    return np.median(d[sel], axis=0), peak, second


def _solve(obs: list[tuple], n_exp: int, rot: np.ndarray, centers: np.ndarray, pos: np.ndarray,
           sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """МНК: неизвестные на экспозицию (sx, sy[, θ]). obs — (i, j) пара звёзд или (i, None) звезда—Gaia.
    Возвращает решение (n_exp × 3), дисперсии сдвигов (n_exp × 2): абсолютные (с привязкой всего набора к Gaia) и
    относительные — сдвига экспозиции к среднему по привязанным экспозициям (совмещение между собой), и нормированные
    остатки наблюдений."""
    idx = np.full((n_exp, 3), -1)
    k = 0
    for e in range(n_exp):
        idx[e, :2] = (k, k + 1)
        k += 2
        if rot[e]:
            idx[e, 2] = k
            k += 1
    rows, cols, vals, rhs, w = [], [], [], [], []
    r = 0
    for a, b, ea, eb, target, s in obs:
        for comp in (0, 1):
            for ii, e, sign in ((a, ea, 1.0), (b, eb, -1.0)):
                if ii is None:
                    continue
                rows.append(r); cols.append(idx[e, comp]); vals.append(sign)
                if rot[e]:
                    v = pos[ii] - centers[e]
                    rows.append(r); cols.append(idx[e, 2]); vals.append(sign * (-v[1] if comp == 0 else v[0]))
            rhs.append(target[comp]); w.append(1.0 / s**2)
            r += 1
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(r, k))
    W = sparse.diags(w)
    N = (A.T @ W @ A).toarray()
    rhs = np.asarray(rhs)
    N[np.diag_indices(k)] += 1e-12 * max(1.0, float(np.max(np.diag(N))))
    cov = np.linalg.inv(N)
    sol = cov @ (A.T @ (W @ rhs))
    res = (A @ sol - rhs).reshape(-1, 2) / np.asarray([o[5] for o in obs])[:, None]
    out = np.zeros((n_exp, 3))
    cv = np.zeros((n_exp, 2))
    for e in range(n_exp):
        out[e, :2] = sol[idx[e, :2]]
        cv[e] = np.diag(cov)[idx[e, :2]]
        if rot[e]:
            out[e, 2] = sol[idx[e, 2]]
    # относительная дисперсия: var(s_e − среднее s по привязанным); непривязанные (дисперсия от регуляризации,
    # > 1″²) в среднее не входят
    rel = np.zeros((n_exp, 2))
    for comp in (0, 1):
        C = cov[np.ix_(idx[:, comp], idx[:, comp])]
        u = (np.diag(C) < 1.0).astype(float)
        u /= max(u.sum(), 1.0)
        rel[:, comp] = np.diag(C) - 2 * (C @ u) + u @ C @ u
    return out, cv, np.clip(rel, 0, None), np.hypot(*res.T)


def solve_pointing(cfg: Config, paths: Paths, files: dict[str, list[Path]], dest: Path) -> dict[str, Any]:
    """Поправки наведения всех экспозиций уровня 2 объекта по Gaia DR3. Кеш: dest/astrometry.json."""
    c = cfg.level2
    allf = sorted({p for v in files.values() for p in v}, key=lambda p: p.name)
    cat = gaia_catalog(cfg, paths)
    key = {"version": ASTRO_VERSION, "files": [[p.name, p.stat().st_size] for p in allf],
           "gaia": [gaia_cache(cfg, paths).name, gaia_cache(cfg, paths).stat().st_size],
           "rules": [c.tweak_max_precision, c.tweak_rot_min_stars, c.tweak_rot_min_deg], "dq_bad": c.dq_bad,
           "field": [cfg.coords, cfg.radius]}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    path = dest / "astrometry.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old.get("digest") == digest:
            console.print(f"  астрометрия по Gaia: из кеша ({path.name})")
            return old
    ra0, dec0 = field_center(cfg)
    from astropy.time import Time

    mids = []
    console.print(f"  астрометрия по Gaia: звёзды в {len(allf)} кадрах детекторов")
    det = []   # (файл, экспозиция, ξ, η, σ″)
    exps = sorted({exposure_of(p.name) for p in allf})
    for fi, p in enumerate(allf):
        with fits.open(p, memmap=False) as h:
            sci = np.array(h["SCI"].data, dtype=np.float32)
            err = np.array(h["ERR"].data, dtype=np.float32)
            dq = np.asarray(h["DQ"].data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                wcs = WCS(h["SCI"].header).celestial
            mids.append(float(h[0].header.get("EXPMID") or h["SCI"].header.get("EXPMID")))
        sat = (dq & 2) != 0   # SATURATED
        sci[(dq & c.dq_bad) != 0] = np.nan
        st = detect_stars(sci, err, sat)
        if not len(st):
            continue
        scale = float(np.mean(np.abs(np.linalg.eigvals(wcs.pixel_scale_matrix)))) * 3600
        ra, dec = wcs.all_pix2world(st[:, 0], st[:, 1], 0)
        xy = _plane(ra, dec, ra0, dec0)
        sg = scale * st[:, 2]
        e = exps.index(exposure_of(p.name))
        det += [(fi, e, xy[k, 0], xy[k, 1], sg[k]) for k in range(len(st))]
    det_a = np.asarray(det, dtype=np.float64).reshape(-1, 5)
    epoch = float(Time(np.mean(mids), format="mjd").decimalyear)
    gra, gdec, gsig = gaia_at(cat, epoch)
    g = _plane(gra, gdec, ra0, dec0)
    if not len(g):
        raise SkyforgeError(f"level2.astrometry: gaia — в {gaia_cache(cfg, paths).name} нет пригодных звёзд Gaia "
                            f"(с собственным движением, RUWE ≤ {GAIA_MAX_RUWE}, ошибка ≤ {GAIA_MAX_ERR_MAS:g} mas)")
    console.print(f"    звёзд в кадрах {len(det_a)}, Gaia на эпоху {epoch:.3f}: {len(g)} (ошибка ≤ {GAIA_MAX_ERR_MAS:g} mas)")
    e_of = det_a[:, 1].astype(int)
    pos_raw = det_a[:, 2:4]
    sig = det_a[:, 4].copy()
    # грубо: наведение — по гидирующей звезде визита, поэтому голосование по визиту целиком (у экспозиции звёзд Gaia мало)
    coarse = np.zeros((len(exps), 2))
    votes = np.zeros(len(exps), dtype=int)
    visit_rows = []
    for v in sorted({visit_of(n) for n in exps}):
        ev = [e for e, n in enumerate(exps) if visit_of(n) == v]
        res = _coarse(pos_raw[np.isin(e_of, ev)], g)
        visit_rows.append({"визит": v, "сдвиг_ξ″": None if res is None else round(float(res[0][0]), 4),
                           "сдвиг_η″": None if res is None else round(float(res[0][1]), 4),
                           "голосов": None if res is None else res[1], "второй_пик": None if res is None else res[2]})
        if res is None:
            console.print(f"    [yellow]визит {v}: грубый сдвиг по Gaia не найден (нет значимого пика)[/]")
            continue
        coarse[ev] = res[0]
        votes[ev] = res[1]
        console.print(f"    визит {v}: грубо ξ {res[0][0]:+.3f}″ η {res[0][1]:+.3f}″ ({np.hypot(*res[0]):.2f}″), "
                      f"голосов {res[1]} (второй пик {res[2]})")
    pos = pos_raw + coarse[e_of]
    # пары одной звезды в разных экспозициях (ближайшая в каждой другой экспозиции) и звёзды Gaia
    tree = cKDTree(pos)
    pairs = tree.query_pairs(MATCH_RADIUS, output_type="ndarray")
    pairs = pairs[e_of[pairs[:, 0]] != e_of[pairs[:, 1]]]
    dist = np.hypot(*(pos[pairs[:, 0]] - pos[pairs[:, 1]]).T)
    seen, keep = set(), []
    for k in np.argsort(dist):
        a, b = pairs[k]
        if (a, e_of[b]) in seen or (b, e_of[a]) in seen:
            continue
        seen.update({(a, e_of[b]), (b, e_of[a])})
        keep.append(k)
    pairs = pairs[keep]
    gd, gi = cKDTree(g).query(pos, distance_upper_bound=GAIA_RADIUS)
    gm = np.flatnonzero(np.isfinite(gd))
    console.print(f"    пар одной звезды в разных экспозициях {len(pairs)}, звёзд на Gaia {len(gm)}")
    centers = np.array([pos[e_of == e].mean(axis=0) if (e_of == e).any() else (0.0, 0.0) for e in range(len(exps))])
    n_obs = np.bincount(np.concatenate([e_of[pairs.ravel()], e_of[gm]]), minlength=len(exps))
    rot = n_obs >= c.tweak_rot_min_stars
    k_int, k_gaia = 1.0, 1.0
    use_p = np.ones(len(pairs), dtype=bool)
    use_g = np.ones(len(gm), dtype=bool)
    def run() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        obs = [(a, b, e_of[a], e_of[b], pos[b] - pos[a], k_int * math.hypot(sig[a], sig[b]))
               for (a, b), u in zip(pairs, use_p) if u]
        obs += [(i, None, e_of[i], None, g[gi[i]] - pos[i], math.hypot(k_gaia * sig[i], gsig[gi[i]]))
                for i, u in zip(gm, use_g) if u]
        return _solve(obs, len(exps), rot, centers, pos, sig)

    for it in range(6):
        sol, cv, cv_rel, nres = run()
        npair = int(use_p.sum())
        ri, rg = nres[:npair], nres[npair:]
        # масштаб ошибок по разбросу (робастно, на ось: медиана |r| двумерного гаусса = 1,177σ)
        k_int *= max(float(np.median(ri)) / 1.177, 1e-3) if len(ri) else 1.0
        if len(rg) >= 5:
            k_gaia *= max(float(np.median(rg)) / 1.177, 1e-3)
        bad_p = np.zeros(len(pairs), dtype=bool)
        bad_p[np.flatnonzero(use_p)] = ri > CLIP * float(np.median(ri)) / 1.177
        bad_g = np.zeros(len(gm), dtype=bool)
        bad_g[np.flatnonzero(use_g)] = rg > CLIP * max(float(np.median(rg)) / 1.177, 1e-9) if len(rg) else False
        new_p, new_g = use_p & ~bad_p, use_g & ~bad_g
        small = rot & (np.degrees(np.abs(sol[:, 2])) < c.tweak_rot_min_deg)
        if np.array_equal(new_p, use_p) and np.array_equal(new_g, use_g) and not small.any() and it >= 2:
            break
        use_p, use_g = new_p, new_g
        rot = rot & ~small
    # решение, ковариация и счётчики — на одних и тех же наблюдениях и масштабах ошибок (последний проход их сменил)
    sol, cv, cv_rel, _ = run()
    # правило 3g — к совмещению экспозиций между собой (двоение, цвет); привязка всего набора к Gaia общая для всех
    # экспозиций и сдвигает только координаты — пишется отдельно
    prec = np.sqrt(cv_rel.mean(axis=1)) / SW_SCALE
    prec_abs = np.sqrt(cv.mean(axis=1)) / SW_SCALE
    final = coarse + sol[:, :2]
    out_files: dict[str, Any] = {}
    rows = []
    for e, name in enumerate(exps):
        ok = n_obs[e] > 0 and prec[e] < c.tweak_max_precision
        status = ("поворот" if rot[e] else "сдвиг") if ok else (
            "грубый сдвиг визита (Gaia)" if votes[e] else "без поправки (визит не привязан к Gaia)")
        shift = final[e] if ok else coarse[e]
        corr = {"origin": [ra0, dec0], "shift": [float(shift[0]), float(shift[1])],
                "theta": float(sol[e, 2]) if ok and rot[e] else 0.0, "center": [float(v) for v in centers[e]],
                "status": status}
        sel_g = gm[use_g][e_of[gm[use_g]] == e]
        gb = np.hypot(*(g[gi[sel_g]] - pos_raw[sel_g]).T) if len(sel_g) else np.array([])
        corr_pos = pos[sel_g] + sol[e, :2] + (sol[e, 2] * np.column_stack([-(pos[sel_g] - centers[e])[:, 1],
                                                                           (pos[sel_g] - centers[e])[:, 0]]))
        ga = np.hypot(*(g[gi[sel_g]] - corr_pos).T) if len(sel_g) else np.array([])
        rows.append({"экспозиция": name, "визит": visit_of(name),
                     "сдвиг_ξ″": round(float(shift[0]), 4), "сдвиг_η″": round(float(shift[1]), 4),
                     "сдвиг_px": round(float(np.hypot(*shift)) / SW_SCALE, 3),
                     "поворот_град": round(math.degrees(corr["theta"]), 5),
                     "голосов_Gaia": int(votes[e]), "наблюдений": int(n_obs[e]), "точность_px": round(float(prec[e]), 4),
                     "точность_к_Gaia_px": round(float(prec_abs[e]), 4),
                     "Gaia_звёзд": int(len(sel_g)),
                     "Gaia_до_px": round(float(np.median(gb)) / SW_SCALE, 3) if len(gb) else None,
                     "Gaia_после_px": round(float(np.median(ga)) / SW_SCALE, 3) if len(ga) else None,
                     "статус": status})
        for p in allf:
            if exposure_of(p.name) == name:
                out_files[p.name] = corr
    # остатки пар до/после, px коротких волн
    a_, b_ = pairs[use_p, 0], pairs[use_p, 1]

    def moved(i: np.ndarray) -> np.ndarray:
        e = e_of[i]
        v = pos[i] - centers[e]
        return pos[i] + sol[e, :2] + sol[e, 2:3] * np.column_stack([-v[:, 1], v[:, 0]])

    before = np.hypot(*(pos_raw[a_] - pos_raw[b_]).T) / SW_SCALE
    after = np.hypot(*(moved(a_) - moved(b_)).T) / SW_SCALE
    # согласие визитов: средний остаток к Gaia по визиту (вектор, px коротких волн) и средняя разность пар между визитами
    vis = np.array([visit_of(exps[e]) for e in e_of])
    gi_used = gm[use_g]
    gres = (g[gi[gi_used]] - moved(gi_used)) / SW_SCALE
    for vr in visit_rows:
        sel = vis[gi_used] == vr["визит"]
        n = int(sel.sum())
        vr["Gaia_звёзд_после"] = n
        if n:
            m = gres[sel].mean(axis=0)
            vr["Gaia_средний_остаток_px"] = [round(float(m[0]), 4), round(float(m[1]), 4)]
            vr["Gaia_ошибка_среднего_px"] = round(float(np.hypot(*gres[sel].std(axis=0)) / math.sqrt(max(n - 1, 1))), 4)
    pres = (moved(a_) - moved(b_)) / SW_SCALE
    between = []
    for va in sorted(set(vis)):
        for vb in sorted(set(vis)):
            if va >= vb:
                continue
            sel = ((vis[a_] == va) & (vis[b_] == vb)) | ((vis[a_] == vb) & (vis[b_] == va))
            if sel.sum() < 5:
                continue
            sgn = np.where(vis[a_][sel] == va, 1.0, -1.0)[:, None]
            m = (pres[sel] * sgn).mean(axis=0)
            between.append({"визиты": f"{va[-3:]}–{vb[-3:]}", "пар": int(sel.sum()),
                            "средняя_разность_px": [round(float(m[0]), 4), round(float(m[1]), 4)],
                            "ошибка_px": round(float(np.hypot(*pres[sel].std(axis=0)) / math.sqrt(sel.sum() - 1)), 4)})
    for b_row in between:
        console.print(f"    пары визитов {b_row['визиты']}: {b_row['пар']}, средняя разность {b_row['средняя_разность_px']} px "
                      f"± {b_row['ошибка_px']}")
    for vr in visit_rows:
        if vr.get("Gaia_звёзд_после"):
            console.print(f"    визит {vr['визит']}: к Gaia после — средний остаток {vr['Gaia_средний_остаток_px']} px ± "
                          f"{vr['Gaia_ошибка_среднего_px']} ({vr['Gaia_звёзд_после']} звёзд)")
    summary = {"epoch": round(epoch, 4), "gaia_stars": int(len(g)), "detections": int(len(det_a)),
               "pairs": int(use_p.sum()), "gaia_matches": int(use_g.sum()),
               "pairs_before_median_px": round(float(np.median(before)), 4) if len(before) else None,
               "pairs_after_median_px": round(float(np.median(after)), 4) if len(after) else None,
               "error_scale_pairs": round(k_int, 3), "error_scale_gaia": round(k_gaia, 3),
               "exposures": len(exps), "corrected": sum(r["статус"] in ("сдвиг", "поворот") for r in rows)}
    console.print(f"    пары звёзд: медиана расхождения {summary['pairs_before_median_px']} → "
                  f"{summary['pairs_after_median_px']} px; исправлено точно {summary['corrected']} из {len(exps)} экспозиций")
    for vis in sorted({r["визит"] for r in rows}):
        vr = [r for r in rows if r["визит"] == vis]
        console.print(f"    визит {vis}: сдвиг медиана {np.median([r['сдвиг_px'] for r in vr]):.2f} px, точность ≤ "
                      f"{max(r['точность_px'] for r in vr):.4f} px")
    res = {"digest": digest, "summary": summary, "visits": visit_rows, "between_visits": between, "exposures": rows,
           "files": out_files}
    dest.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return res
