"""Фаза 9, шаг 1: разведка наземных архивов для ярких галактик — только метаданные, заголовки и превью.

Архивы: NOIRLab Astro Data Archive (DECam: instcal, resampled, stacked), ESO Science Archive (ObsCore TAP, изображения
Phase 3 и обработанные), Pan-STARRS DR2 (стеки skycell, заголовки — первые 30 КБ файла запросом Range), Legacy Survey
DR10 (наличие полос в cutout 64 px). Превью: цветные JPEG Legacy Survey и Pan-STARRS (fitscut), по объекту.
Итог: out/galaxies/разведка/recon.json, recon.md (таблица) и 01_превью.png.

Запуск: uv run python scripts/recon_ground.py
"""

from __future__ import annotations

import collections
import io
import json
import math
import pathlib
import time

import numpy as np
import requests
from astropy.io import fits
from PIL import Image

OUT = pathlib.Path("out/galaxies/разведка")
# ra, dec (°), большая и малая ось (′) — NED/RC3
OBJECTS = {
    "NGC 253": (11.888, -25.288, 27.5, 6.8),
    "M83": (204.254, -29.866, 12.9, 11.5),
    "NGC 1365": (53.402, -36.141, 11.2, 6.2),
    "M104": (189.998, -11.623, 8.7, 3.5),
    "NGC 4945": (196.364, -49.468, 20.0, 3.8),
    "Cen A": (201.365, -43.019, 25.7, 20.0),
}
S = requests.Session()


def get(url: str, **kw) -> requests.Response:
    for k in range(3):
        try:
            r = S.get(url, timeout=kw.pop("timeout", 120), **kw)
            r.raise_for_status()
            return r
        except Exception:
            if k == 2:
                raise
            time.sleep(3)
    raise RuntimeError


def noirlab(ra: float, dec: float) -> dict:
    """DECam: центры кадров ближе 0,7° (поле DECam ~2,2° в поперечнике) — галактика целиком в кадре."""
    dra = 0.7 / math.cos(math.radians(dec))
    body = {"outfields": ["proc_type", "prod_type", "ifilter", "exposure", "filesize", "release_date", "proposal",
                          "ra_center", "dec_center"],
            "search": [["instrument", "decam"], ["obs_type", "object"], ["ra_center", ra - dra, ra + dra],
                       ["dec_center", dec - 0.7, dec + 0.7]]}
    r = S.post("https://astroarchive.noirlab.edu/api/adv_search/find/?limit=20000&format=json", json=body, timeout=300)
    r.raise_for_status()
    rows = [x for x in r.json()[1:] if x["prod_type"] in ("image", "image1")]
    now = time.strftime("%Y-%m-%d")
    out = collections.defaultdict(lambda: {"n": 0, "exp_s": 0.0, "gb": 0.0, "proposals": set(), "private": 0})
    for x in rows:
        band = (x["ifilter"] or "?").split()[0]
        k = (x["proc_type"], band)
        o = out[k]
        o["n"] += 1
        o["exp_s"] += x["exposure"] or 0
        o["gb"] += (x["filesize"] or 0) / 1e9
        o["proposals"].add(x["proposal"])
        o["private"] += (x["release_date"] or "9999") > now
    return {f"{p}/{b}": {**v, "proposals": len(v["proposals"]), "exp_s": round(v["exp_s"]), "gb": round(v["gb"], 1)}
            for (p, b), v in sorted(out.items())}


def eso(ra: float, dec: float, rad_deg: float) -> dict:
    q = ("SELECT obs_collection, instrument_name, filter, calib_level, dataproduct_subtype, t_exptime, s_pixel_scale, "
         "access_estsize, obs_release_date FROM ivoa.ObsCore WHERE dataproduct_type='image' AND "
         f"INTERSECTS(s_region, CIRCLE('ICRS', {ra}, {dec}, {rad_deg}))=1")
    r = get("https://archive.eso.org/tap_obs/sync", params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json",
                                                           "QUERY": q, "MAXREC": 20000}, timeout=300)
    d = r.json()
    cols = [c["name"] for c in d["metadata"]]
    out = collections.defaultdict(lambda: {"n": 0, "exp_s": 0.0, "gb": 0.0, "pix": []})
    for row in d["data"]:
        x = dict(zip(cols, row))
        k = (x["obs_collection"], x["instrument_name"], x["filter"], x["calib_level"])
        o = out[k]
        o["n"] += 1
        o["exp_s"] += x["t_exptime"] or 0
        o["gb"] += (x["access_estsize"] or 0) / 1e6      # кБ → ГБ
        if x["s_pixel_scale"]:
            o["pix"].append(x["s_pixel_scale"])
    return {" / ".join(str(v) for v in k): {"n": v["n"], "exp_s": round(v["exp_s"]), "gb": round(v["gb"], 2),
                                           "pix_arcsec": round(float(np.median(v["pix"])), 3) if v["pix"] else None}
            for k, v in sorted(out.items(), key=lambda kv: -kv[1]["n"])}


def ps1(ra: float, dec: float, a: float) -> dict:
    """Стеки skycell по сетке точек на ±a/2; заголовок первого стека каждой полосы — Range первых 30 КБ."""
    if dec < -30.5:
        return {"нет": "южнее −30° Pan-STARRS не снимал"}
    files = collections.defaultdict(set)
    h = a / 2 / 60
    for dy in np.linspace(-h, h, 3):
        for dx in np.linspace(-h, h, 3):
            pr = ra + dx / math.cos(math.radians(dec))
            t = get(f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={pr}&dec={dec + dy}&filters=grizy").text
            for line in t.splitlines()[1:]:
                p = line.split()
                if len(p) >= 8 and p[6] == "stack" if len(p) > 6 else False:
                    files[p[4]].add(p[7])
    out = {}
    for band, fs in sorted(files.items()):
        f = sorted(fs)[0]
        r = S.get("https://ps1images.stsci.edu" + f, headers={"Range": "bytes=0-30000"}, timeout=120)
        hdr = {}
        try:
            with fits.open(io.BytesIO(r.content + b" " * (2880 - len(r.content) % 2880)), ignore_missing_end=True) as hh:
                for ext in hh:
                    hdr.update({k: ext.header.get(k) for k in ("EXPTIME", "NINPUTS", "NAXIS1", "ZNAXIS1", "ZNAXIS2", "CDELT1",
                                                               "BUNIT", "BSOFTEN", "HIERARCH FPA.ZP", "FPA.ZP")
                                if ext.header.get(k) is not None})
        except Exception as e:
            hdr["ошибка"] = type(e).__name__
        out[band] = {"skycells": len(fs), "file_example": f, "header": hdr}
    return out


def legacy(ra: float, dec: float) -> dict:
    r = get("https://www.legacysurvey.org/viewer/cutout.fits", params={"ra": ra, "dec": dec, "layer": "ls-dr10",
                                                                        "pixscale": 0.262, "size": 64, "bands": "griz"})
    with fits.open(io.BytesIO(r.content)) as h:
        a = h[0].data
        hdr = h[0].header
    bands = hdr.get("BANDS", "griz")
    return {"bands": {b: bool(np.nanmax(np.abs(a[i])) > 0) for i, b in enumerate(bands.strip())},
            "pix_arcsec": 0.262}


def previews(name: str, ra: float, dec: float, a: float) -> dict:
    side = max(a * 1.4, 8) * 60                   # ″
    ims = {}
    try:
        px = side / 1000
        r = get("https://www.legacysurvey.org/viewer/cutout.jpg", params={"ra": ra, "dec": dec, "layer": "ls-dr10",
                                                                          "pixscale": round(px, 3), "size": 1000})
        ims["Legacy Survey DR10 (DECam g, r, z)"] = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        ims["Legacy Survey DR10"] = f"нет: {type(e).__name__}"
    if dec > -30.5:
        try:
            t = get(f"https://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra={ra}&dec={dec}&filters=gri").text
            fn = {p.split()[4]: p.split()[7] for p in t.splitlines()[1:]}
            r = get("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi",
                    params={"red": fn["i"], "green": fn["r"], "blue": fn["g"], "ra": ra, "dec": dec,
                            "size": int(side / 0.25), "output_size": 1000, "format": "jpg", "autoscale": 99.5},
                    timeout=300)
            ims["Pan-STARRS DR2 (g, r, i)"] = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception as e:
            ims["Pan-STARRS DR2"] = f"нет: {type(e).__name__}: {e}"[:120]
    return ims


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res, pv = {}, {}
    for name, (ra, dec, a, b) in OBJECTS.items():
        print(name)
        res[name] = {"ra": ra, "dec": dec, "size_arcmin": [a, b]}
        for arch, fn in (("noirlab", lambda: noirlab(ra, dec)), ("eso", lambda: eso(ra, dec, a / 2 / 60)),
                         ("ps1", lambda: ps1(ra, dec, a)), ("legacy", lambda: legacy(ra, dec))):
            try:
                res[name][arch] = fn()
            except Exception as e:
                res[name][arch] = {"ошибка": f"{type(e).__name__}: {e}"[:200]}
            print("  ", arch, "ok" if "ошибка" not in res[name][arch] else res[name][arch]["ошибка"])
        pv[name] = previews(name, ra, dec, a)
    (OUT / "recon.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # лист превью: строка — объект, столбцы — архивы
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["Legacy Survey DR10 (DECam g, r, z)", "Pan-STARRS DR2 (g, r, i)"]
    fig, axs = plt.subplots(len(pv), 2, figsize=(12, 6 * len(pv)), dpi=90)
    for i, (name, ims) in enumerate(pv.items()):
        a = OBJECTS[name][2]
        for j, c in enumerate(cols):
            ax = axs[i, j]
            im = ims.get(c) or next((v for k, v in ims.items() if k.startswith(c.split(" (")[0])), None)
            if isinstance(im, Image.Image):
                ax.imshow(im)
            else:
                ax.text(0.5, 0.5, str(im or "нет данных"), ha="center", va="center", wrap=True, color="w")
                ax.set_facecolor("#222")
            ax.set_title(f"{name} — {c}, поле {max(a * 1.4, 8):.0f}′", fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "01_превью.png")
    print(OUT)


if __name__ == "__main__":
    main()
