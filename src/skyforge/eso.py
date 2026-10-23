"""Наземные данные из ESO Science Archive (фаза 9, mission: ground): сложенные тайлы Phase 3 и вырезка на сервере.

Поиск — TAP ivoa.ObsCore: изображения calib_level 3 инструмента (cfg.instrument) коллекции cfg.archive_collection, тайл
содержит центр (coords). Файл и его карта весов — по datalink (#this, #auxiliary с категорией ANCILLARY.WEIGHTMAP).
Вырезка кругом radius вокруг coords — SODA на сервере ESO (из тайла ~70′ берётся только поле галактики). Аккаунт для
публичных данных не нужен; лицензия ESO — CC BY 4.0.

Сырые вырезки — в data/raw/eso/<объект>/ (только чтение); для конвейера — <объект>_<полоса>.fits: SCI (данные и WCS),
WHT (вес вырезки; ≤ 0 — пусто при чтении, как у i2d), в PRIMARY — фотометрический нуль, PSF, экспозиция, ссылка.
"""

from __future__ import annotations

import io
import os
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import requests
from astropy.io import fits

from skyforge.config import Config
from skyforge.filters import ground_filter, sort_by_wavelength
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths

TAP = "https://archive.eso.org/tap_obs/sync"
DATALINK = "https://archive.eso.org/datalink/links"
SODA = "https://dataportal.eso.org/dataPortal/soda/sync"
LICENSE = "CC BY 4.0"
ACKNOWLEDGMENT = "Based on data obtained from the ESO Science Archive Facility"
# ключи заголовка тайла, которые переносятся в PRIMARY продукта
KEEP = ("TELESCOP", "INSTRUME", "OBJECT", "EXPTIME", "TEXPTIME", "NCOMBINE", "BUNIT", "PHOTZP", "PHOTSYS", "ABMAGLIM",
        "ABMAGSAT", "PSF_FWHM", "FLUXCAL", "REFERENC", "PROG_ID", "ORIGFILE", "PRODCATG", "DATE-OBS", "MJD-OBS", "MJD-END")


def center(cfg: Config) -> tuple[float, float]:
    if not cfg.coords:
        raise SkyforgeError("mission: ground — нужны coords (центр вырезки, градусы)")
    ra, dec = (float(v) for v in cfg.coords.replace(",", " ").split()[:2])
    return ra, dec


def radius_deg(cfg: Config) -> float:
    from astropy.coordinates import Angle

    return float(Angle(cfg.radius).deg)


def query(cfg: Config) -> list[dict[str, Any]]:
    """Тайлы calib_level 3, содержащие центр: по одному на полосу (самая длинная экспозиция)."""
    ra, dec = center(cfg)
    where = [f"dataproduct_type='image'", "calib_level=3", f"instrument_name='{cfg.instrument.upper()}'",
             f"CONTAINS(POINT('ICRS', {ra}, {dec}), s_region)=1"]
    if cfg.archive_collection:
        where.append(f"obs_collection='{cfg.archive_collection}'")
    q = ("SELECT dp_id, obs_collection, filter, t_exptime, s_pixel_scale, obs_release_date, access_estsize "
         "FROM ivoa.ObsCore WHERE " + " AND ".join(where))
    r = requests.get(TAP, params={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "json", "QUERY": q}, timeout=180)
    r.raise_for_status()
    d = r.json()
    cols = [c["name"] for c in d["metadata"]]
    best: dict[str, dict[str, Any]] = {}
    for row in d["data"]:
        x = dict(zip(cols, row))
        f = ground_filter(x["filter"] or "")
        if not f:
            continue
        x["band"] = f
        if f not in best or (x["t_exptime"] or 0) > (best[f]["t_exptime"] or 0):
            best[f] = x
    return [best[f] for f in sort_by_wavelength(list(best), "ground")]


def links(dp_id: str) -> dict[str, str]:
    """datalink: адрес файла (#this) и dp_id карты весов (#auxiliary, ANCILLARY.WEIGHTMAP)."""
    from astropy.io.votable import parse_single_table

    r = requests.get(DATALINK, params={"ID": f"ivo://eso.org/ID?{dp_id}"}, timeout=120)
    r.raise_for_status()
    t = parse_single_table(io.BytesIO(r.content)).to_table()
    out: dict[str, str] = {}
    for row in t:
        sem = str(row["semantics"])
        cat = " ".join(str(row[c]) for c in t.colnames if "categ" in c.lower())
        url = str(row["access_url"])
        if sem == "#this":
            out["this"] = url
        elif sem == "#auxiliary" and "WEIGHT" in cat.upper():
            out["weight"] = url.rsplit("/", 1)[-1]
    return out


def _safe(dp_id: str) -> str:
    """Имя файла из dp_id: двоеточия времени (ADP.2020-08-26T11:45:32.315) в Windows недопустимы."""
    return dp_id.replace(":", "-")


def cutout(dp_id: str, ra: float, dec: float, r_deg: float, dest: Path) -> Path | None:
    """Вырезка кругом на сервере ESO (SODA); файл пишется через .part и помечается «только чтение». None — сервер
    вырезку этого продукта не делает (204 без содержимого: так у несжатых карт весов FDS, 1,76 ГБ на полосу)."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(SODA, params={"ID": f"ivo://eso.org/ID?{dp_id}", "CIRCLE": f"{ra} {dec} {r_deg}"},
                      stream=True, timeout=600) as r:
        r.raise_for_status()
        if r.status_code == 204:
            return None
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    if tmp.stat().st_size == 0:
        tmp.unlink()
        return None
    tmp.replace(dest)
    os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    return dest


def _image(path: Path) -> tuple[np.ndarray, fits.Header, fits.Header]:
    """(данные, заголовок изображения с WCS, заголовок PRIMARY) первого двумерного HDU (.fz — сжатый)."""
    with fits.open(path, memmap=False) as h:
        prim = h[0].header.copy()
        for hdu in h:
            if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
                return np.array(hdu.data, dtype=np.float32), hdu.header.copy(), prim
    raise SkyforgeError(f"{path.name}: нет двумерного изображения")


def make_product(sci_path: Path, wht_path: Path | None, band: str, meta: dict[str, Any], dest: Path) -> dict[str, Any]:
    """FITS для конвейера: SCI (данные, WCS вырезки), WHT (вес; нет карты — 1 там, где данные конечны)."""
    data, hdr, prim = _image(sci_path)
    if wht_path is not None:
        wht, whdr, _ = _image(wht_path)
        if wht.shape != data.shape:
            raise SkyforgeError(f"{band}: вырезка карты весов {wht.shape} не совпадает с данными {data.shape}")
    else:   # карты весов нет — вес 1 там, где данные конечны и не нулевые (у тайлов FDS вне покрытия 0)
        wht = (np.isfinite(data) & (data != 0)).astype(np.float32)
    p = fits.Header()
    for k in KEEP:
        for src in (prim, hdr):
            if k in src and k not in p:
                p[k] = src[k]
    p["FILTER"] = (band, "skyforge: short band name")
    p["ESO_DPID"] = (meta["dp_id"], "ESO Phase 3 product (tile)")
    p["ESO_COLL"] = (meta["obs_collection"], "ESO collection")
    p["LICENSE"] = (LICENSE, "ESO archive data licence")
    p["CUTOUT"] = (meta["cutout"], "SODA circle: ra dec radius, deg")
    p["WHTSRC"] = ("weight map cutout" if wht_path is not None else "data finite and non-zero", "WHT layer source")
    sci = fits.ImageHDU(data, header=hdr, name="SCI")
    for k in ("EXTNAME", "ZIMAGE", "ZCMPTYPE"):
        sci.header.pop(k, None)
    sci.header["EXTNAME"] = "SCI"
    hdus = fits.HDUList([fits.PrimaryHDU(header=p), sci, fits.ImageHDU(wht, name="WHT")])
    tmp = dest.with_name(dest.name + ".part")
    hdus.writeto(tmp, overwrite=True)
    tmp.replace(dest)
    os.chmod(dest, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    finite = np.isfinite(data) & (wht > 0)
    return {"shape": list(data.shape), "empty_percent": round(float((~finite).mean() * 100), 3),
            **{k.lower(): p.get(k) for k in ("EXPTIME", "PHOTZP", "PHOTSYS", "ABMAGLIM", "ABMAGSAT", "PSF_FWHM",
                                             "BUNIT", "REFERENC")}}


def fetch_eso(cfg: Config, paths: Paths, *, offline: bool = False, list_only: bool = False,
              max_gb: float | None = None) -> Any:
    from skyforge.fetch import Available, FetchResult, Product, load_manifest, local_products, save_manifest

    found = local_products(cfg, paths)
    want = list(cfg.filters)
    for f, prod in found.items():
        console.print(f"  {f}: уже есть ({prod.source}) {prod.path}")
    if offline or (want and all(f in found for f in want) and not list_only):
        missing = [f for f in want if f not in found]
        return FetchResult(products=found, missing=missing)
    ra, dec = center(cfg)
    r_deg = radius_deg(cfg)
    console.print(f"Ищу в ESO: {cfg.instrument.upper()}, коллекция {cfg.archive_collection or 'любая'}, центр {ra} {dec}")
    rows = query(cfg)
    if not rows:
        raise SkyforgeError("в ESO нет сложенных тайлов (calib_level 3) с этим центром: проверь coords, instrument, "
                            "archive_collection")
    manifest = load_manifest(cfg, paths)
    avail = {x["band"]: Available(filter=x["band"], obs_id=x["dp_id"], obsid=x["dp_id"], exptime=x["t_exptime"],
                                  public=True, release=str(x["obs_release_date"])[:10],
                                  proposal_id=x["obs_collection"]) for x in rows}
    for x in rows:
        console.print(f"  {x['band']}: {x['dp_id']} ({x['obs_collection']}), {x['t_exptime']:.0f} с, "
                      f"{x['s_pixel_scale']:.3f}″/px, тайл {x['access_estsize'] / 1e3:.0f} МБ")
    manifest["available"] = {f: asdict(a) for f, a in avail.items()}
    manifest["archive"] = {"name": "ESO Science Archive", "license": LICENSE, "acknowledgment": ACKNOWLEDGMENT,
                           "collection": cfg.archive_collection, "cutout": {"ra": ra, "dec": dec, "radius_deg": r_deg}}
    if list_only:
        save_manifest(cfg, paths, manifest)
        return FetchResult(products=found, available=avail)
    d = paths.raw / "eso" / cfg.slug
    todo = [f for f in (want or list(avail)) if f in avail and f not in found]
    for f in todo:
        x = rows[[r["band"] for r in rows].index(f)]
        lk = links(x["dp_id"])
        console.print(f"  {f}: вырезка {2 * r_deg * 60:.1f}′ из {x['dp_id']} и карты весов {lk.get('weight', '—')}")
        sci = cutout(x["dp_id"], ra, dec, r_deg, d / f"{_safe(x['dp_id'])}_cut.fits.fz")
        wht = (cutout(lk["weight"], ra, dec, r_deg, d / f"{_safe(lk['weight'])}_cut.fits.fz") if lk.get("weight")
               else None)
        dest = d / f"{cfg.slug}_{f}.fits"
        if sci is None:
            raise SkyforgeError(f"{f}: ESO не сделал вырезку {x['dp_id']} (SODA без содержимого)")
        if lk.get("weight") and wht is None:
            console.print(f"    [yellow]{f}: карту весов {lk['weight']} сервер не вырезает — вес по данным "
                          f"(конечные и не нулевые)[/]")
        info = make_product(sci, wht, f, {**x, "cutout": f"{ra} {dec} {r_deg}"}, dest)
        info["weight"] = "cutout" if wht is not None else f"нет вырезки ({lk.get('weight')}); по данным"
        prod = Product(filter=f, path=paths.rel(dest), source="eso", obs_id=x["dp_id"], filename=dest.name,
                       size=dest.stat().st_size, exptime=x["t_exptime"], release=str(x["obs_release_date"])[:10],
                       proposal_id=x["obs_collection"])
        manifest.setdefault("products", {})[f] = asdict(prod)
        manifest.setdefault("headers", {})[f] = info
        save_manifest(cfg, paths, manifest)
        console.print(f"    {dest.name}: {info['shape'][1]}×{info['shape'][0]}, пусто {info['empty_percent']}%, "
                      f"нуль {info['photzp']} {info['photsys']}, PSF {info['psf_fwhm']}″, насыщение с {info['abmagsat']}")
    found = local_products(cfg, paths)
    return FetchResult(products=found, missing=[f for f in want if f not in found], available=avail)

