"""Фаза 8: проверка i2d MAST программы 7433 (OMC3-N) по Gaia DR3 — для отчёта в STScI.

Для каждого фильтра: звёзды Gaia DR3 (на эпоху наблюдения, с собственным движением) ищутся в i2d MAST; центр — по
симметрии крыльев PSF (почти все звёзды Gaia в NIRCam насыщены). Зона каждой звезды — по покрытию детекторов
визитов с исходными WCS из _cal: только визиты с гидом S1LG003678 (001, 003), только с S1LG092721 (002, 004) или оба.
Печатает медиану вектора «i2d − Gaia» по зонам, пишет таблицу out/omc3n/проверка_9/визиты/i2d_offset_table.md.

Запуск: uv run python scripts/i2d_offset_check.py [own]
(own — то же для своей сборки уровня 2 из data/work/omc3n/level2; таблица own_offset_table.md)
"""

from __future__ import annotations

import glob
import json
import math
import pathlib
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

from skyforge import astrometry as A
from skyforge.config import Config
from skyforge.paths import Paths

warnings.filterwarnings("ignore")
GUIDE = {"001": "S1LG003678", "003": "S1LG003678", "002": "S1LG092721", "004": "S1LG092721"}


def main() -> None:
    own = len(sys.argv) > 1 and sys.argv[1] == "own"
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/omc3n/showcase.yaml", "showcase")
    ra0, dec0 = A.field_center(cfg)
    cal = sorted(glob.glob("data/raw/omc3n/l2/*_cal.fits"))
    mid = float(np.mean([fits.getheader(p)["EXPMID"] for p in cal[::10]]))
    gra, gdec, gsig = A.gaia_at(A.gaia_catalog(cfg, P), float(Time(mid, format="mjd").decimalyear))
    lines = ["| filter | zone (visits by guide star) | Gaia stars | median i2d − Gaia, E / N (arcsec) | median |offset| (arcsec) |",
             "|---|---|---|---|---|"]
    for f in ("F150W", "F200W", "F356W", "F480M"):
        files = [p for p in cal if fits.getheader(p)["FILTER"] == f]
        cover = {"good": np.zeros(len(gra), bool), "bad": np.zeros(len(gra), bool)}
        for p in files:
            w = WCS(fits.getheader(p, "SCI")).celestial
            x, y = w.wcs_world2pix(gra, gdec, 0)
            n = fits.getheader(p, "SCI")["NAXIS1"]
            ins = (x >= 0) & (x < n) & (y >= 0) & (y < n)
            cover["good" if GUIDE[pathlib.Path(p).name[10:13]] == "S1LG003678" else "bad"] |= ins
        if own:
            meta = json.loads(pathlib.Path("data/work/omc3n/level2/meta.json").read_text(encoding="utf-8"))
            i2d = f"data/work/omc3n/level2/{meta['filters'][f]['file']}"
        else:
            i2d = glob.glob(f"data/raw/jwst/jw07433-o001_t001_nircam_clear-{f.lower()}/*i2d.fits")[0]
        with fits.open(i2d) as h:
            img = np.array(h["SCI"].data, dtype=np.float32)
            img[np.asarray(h["WHT"].data) <= 0] = np.nan
            w = WCS(h["SCI"].header)
        scale = float(np.mean(np.abs(np.linalg.eigvals(w.pixel_scale_matrix)))) * 3600
        x, y = w.all_world2pix(gra, gdec, 0)
        half = int(round(0.6 / scale))
        res = {}
        for k in range(len(gra)):
            zone = ("both" if cover["good"][k] and cover["bad"][k] else "good only" if cover["good"][k]
                    else "bad only" if cover["bad"][k] else None)
            if zone is None:
                continue
            # окно поиска 2″ — шире сдвига 1,2″: самый яркий участок, затем центр по симметрии
            R = int(round(2.0 / scale))
            ix, iy = int(round(x[k])), int(round(y[k]))
            if not (R + half + 2 <= iy < img.shape[0] - R - half - 2 and R + half + 2 <= ix < img.shape[1] - R - half - 2):
                continue
            cut = img[iy - R:iy + R + 1, ix - R:ix + R + 1]
            sm = np.nan_to_num(cut, nan=np.nanmax(cut) if np.isfinite(cut).any() else 0)
            from scipy import ndimage

            sm = ndimage.gaussian_filter(sm, 2)
            my, mx = np.unravel_index(np.argmax(sm), sm.shape)
            c = A._symmetric_center(img, ix - R + mx, iy - R + my, half)
            if c is None:
                continue
            cra, cdec = w.all_pix2world(c[0], c[1], 0)
            d = A._plane(np.array([cra]), np.array([cdec]), ra0, dec0)[0] - A._plane(gra[k:k + 1], gdec[k:k + 1], ra0, dec0)[0]
            if np.hypot(*d) < 1.8:
                res.setdefault(zone, []).append(d)
        names = {"good only": "only visits 001/003 (S1LG003678)", "bad only": "only visits 002/004 (S1LG092721)",
                 "both": "overlap of both"}
        for z in ("good only", "bad only", "both"):
            if z not in res:
                continue
            v = np.array(res[z])
            m = np.median(v, axis=0)
            mag = float(np.median(np.hypot(*v.T)))
            print(f"{f} {names[z]}: {len(v)} stars, median offset E {m[0]:+.3f} N {m[1]:+.3f}″, |offset| {mag:.3f}″")
            lines.append(f"| {f} | {names[z]} | {len(v)} | {m[0]:+.3f} / {m[1]:+.3f} | {mag:.3f} |")
    out = pathlib.Path(f"out/omc3n/проверка_9/визиты/{'own' if own else 'i2d'}_offset_table.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
