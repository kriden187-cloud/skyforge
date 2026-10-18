"""Чтение FITS. Файлы открываются только на чтение; данные копируются в нативный float32."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_scales

from skyforge.filters import effective_filter


@dataclass
class FitsInfo:
    telescope: str
    instrument: str
    filter: str | None
    target: str
    shape: tuple[int, int]
    pixel_scale_arcsec: float
    exptime: float | None
    date_obs: str | None
    bunit: str | None


def _sci_index(hdul: fits.HDUList) -> int:
    for i, hdu in enumerate(hdul):
        if hdu.name == "SCI" and hdu.data is not None:
            return i
    for i, hdu in enumerate(hdul):
        if getattr(hdu, "data", None) is not None and hdu.data.ndim == 2:
            return i
    raise ValueError("в файле нет двумерного изображения")


def _wcs(header: fits.Header) -> WCS:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        w = WCS(header).celestial
    if not w.has_celestial:
        raise ValueError("в заголовке нет небесной WCS")
    return w


def read_info(path: Path) -> FitsInfo:
    with fits.open(path, memmap=True, mode="readonly") as hdul:
        primary = hdul[0].header
        i = _sci_index(hdul)
        h = hdul[i].header
        shape = (int(h["NAXIS2"]), int(h["NAXIS1"]))
        scale = float(np.mean(proj_plane_pixel_scales(_wcs(h)))) * 3600

    def get(*keys: str) -> str | None:
        for k in keys:
            for hdr in (primary, h):
                v = hdr.get(k)
                if v not in (None, ""):
                    return str(v)
        return None

    filt = effective_filter(";".join(x for x in (get("FILTER"), get("PUPIL"), get("FILTER1"), get("FILTER2")) if x))
    exptime = get("EFFEXPTM", "TEXPTIME", "EXPTIME")
    return FitsInfo(
        telescope=(get("TELESCOP") or "").upper(),
        instrument=(get("INSTRUME") or "").upper(),
        filter=filt,
        target=get("TARGPROP", "TARGNAME") or "",
        shape=shape,
        pixel_scale_arcsec=scale,
        exptime=float(exptime) if exptime else None,
        date_obs=get("DATE-BEG", "DATE-OBS"),
        bunit=get("BUNIT"),
    )


def load_sci(path: Path) -> tuple[np.ndarray, WCS]:
    """Научный кадр как float32 в нативном порядке байтов; пиксели без экспозиции → NaN."""
    with fits.open(path, memmap=True, mode="readonly") as hdul:
        i = _sci_index(hdul)
        data = np.array(hdul[i].data, dtype=np.float32)  # копия: сырой файл не трогаем
        wcs = _wcs(hdul[i].header)
        names = [h.name for h in hdul]
        if "WHT" in names:
            wht = hdul["WHT"].data
            if wht is not None and wht.shape == data.shape:
                data[np.asarray(wht) <= 0] = np.nan
    data[~np.isfinite(data)] = np.nan
    return data, wcs
