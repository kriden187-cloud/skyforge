"""Фаза 8: источник по фильтру — level2_filters: часть фильтров из своей сборки уровня 2, остальные — i2d уровня 3."""

from __future__ import annotations

from pathlib import Path

import pytest

from skyforge.config import Config
from skyforge.fetch import Product
from skyforge.paths import Paths


def _cfg(**kw) -> Config:
    base = dict(target="OMC3-N", filters=["F150W", "F200W", "F356W", "F480M"], reference="F356W")
    return Config(**{**base, **kw})


def test_level2_filters_rules():
    c = _cfg(level2_filters=["f150w", "F200W"])
    assert c.level2_filters == ["F150W", "F200W"]
    assert c.uses_level2("F150W") and c.uses_level2("F200W") and not c.uses_level2("F356W") and c.any_level2()
    assert not _cfg().any_level2() and not _cfg().uses_level2("F150W")
    allc = _cfg(source="level2")
    assert allc.any_level2() and all(allc.uses_level2(f) for f in allc.filters)
    with pytest.raises(ValueError, match="только при source: level3"):
        _cfg(source="level2", level2_filters=["F150W"])
    with pytest.raises(ValueError, match="нет в filters"):
        _cfg(level2_filters=["F090W"])
    with pytest.raises(ValueError, match="только JWST"):
        _cfg(mission="hst", instrument="wfc3", level2_filters=["F150W"])


def test_level2_filters_digest_only_when_set():
    """Без поля хеш обработки прежний (принятые объекты те же); с полем — другой."""
    assert _cfg().digest() == _cfg(level2_filters=None).digest()
    assert _cfg(level2_filters=["F150W"]).digest() != _cfg().digest()


def _prod(f: str, path: str, source: str = "mast") -> Product:
    return Product(filter=f, path=path, source=source, obs_id="o", filename=Path(path).name, size=1, exptime=100.0,
                   release="2026-09-09", proposal_id="7433")


def test_require_products_mixes_level2_and_level3(tmp_path, monkeypatch):
    import skyforge.fetch as F
    import skyforge.level2 as L

    paths = Paths(tmp_path)
    l3 = {f: _prod(f, f"data/raw/{f}_i2d.fits") for f in ("F150W", "F200W", "F356W", "F480M")}
    monkeypatch.setattr(F, "local_products", lambda cfg, p: dict(l3))
    built = tmp_path / "data" / "work" / "omc3n" / "level2"
    built.mkdir(parents=True)
    for f in ("F150W", "F200W"):
        (built / f"{f}_l2_x.fits").write_bytes(b"0" * 10)
    monkeypatch.setattr(L, "built_products", lambda cfg, p: {f: built / f"{f}_l2_x.fits" for f in ("F150W", "F200W")})
    cfg = _cfg(level2_filters=["F150W", "F200W"])
    got = F.require_products(cfg, paths)
    assert [got[f].source for f in ("F150W", "F200W", "F356W", "F480M")] == ["level2", "level2", "mast", "mast"]
    assert got["F150W"].path.endswith("F150W_l2_x.fits") and got["F356W"].path == "data/raw/F356W_i2d.fits"
    assert F.require_products(cfg, paths, level3=True)["F150W"].source == "mast"     # сборке нужны i2d
    with pytest.raises(Exception, match="нет своей сборки"):
        F.require_products(_cfg(level2_filters=["F150W", "F480M"]), paths)


def test_fetch_level2_asks_only_level2_filters(tmp_path, monkeypatch):
    import skyforge.fetch as F

    avail = {f: {"obsid": f"obs-{f}"} for f in ("F150W", "F200W", "F356W", "F480M")}
    monkeypatch.setattr(F, "load_manifest", lambda cfg, p: {"available": avail})
    asked = []
    monkeypatch.setattr(F, "product_lists", lambda obsids: asked.extend(obsids) or [])
    res = F.fetch_level2(_cfg(level2_filters=["F150W", "F200W"]), Paths(tmp_path), list_only=True)
    assert sorted(asked) == ["obs-F150W", "obs-F200W"] and sorted(res["files"]) == ["F150W", "F200W"]


def test_build_skips_level3_filters(tmp_path, monkeypatch):
    """Сборка — только фильтры из уровня 2; экспозиции уровня 2 для остальных не нужны."""
    import skyforge.fetch as F
    import skyforge.level2 as L

    paths = Paths(tmp_path)
    for f in ("F150W", "F356W"):
        (tmp_path / f"{f}_i2d.fits").write_bytes(b"0")
    monkeypatch.setattr(F, "require_products",
                        lambda cfg, p, level3=False: {f: _prod(f, str(tmp_path / f"{f}_i2d.fits")) for f in ("F150W", "F356W")})
    cal = tmp_path / "a_cal.fits"
    cal.write_bytes(b"0")
    monkeypatch.setattr(F, "level2_files", lambda cfg, p: {"F150W": [cal]})
    built = []

    def fake_build(cfg, f, i2d, files, dest):
        built.append(f)
        dest.write_bytes(b"0")
        return {}, {"destripe": ([], []), "overlaps": ([], []), "stars": ([], []), "astrometry": (f, None),
                    "curves": None}

    monkeypatch.setattr(L, "_build_filter", fake_build)
    monkeypatch.setattr(L, "cache_previews", lambda *a, **k: None)
    monkeypatch.setattr(L, "contact_sheet", lambda *a, **k: None)
    monkeypatch.setattr(L, "_astrometry_sheet", lambda *a, **k: None)
    cfg = _cfg(filters=["F150W", "F356W"], level2_filters=["F150W"])
    L.build(cfg, paths, tmp_path / "out")
    assert built == ["F150W"]
    assert L.build(_cfg(filters=["F150W", "F356W"]), paths, tmp_path / "out") == {"skipped": "source: level3"}


def test_blend_large_scale_takes_large_from_i2d_small_from_own(tmp_path):
    """Фаза 8, гибрид: своя сборка с другим масштабом и уровнем (0,87·i2d + 0,1), без полос; i2d с полосами по строкам.
    Гибрид: крупный масштаб как у i2d, полос нет (мелкий масштаб своей сборки); пустые пиксели своей — из i2d."""
    import numpy as np
    from astropy.io import fits
    from scipy import ndimage

    from skyforge.level2 import blend_large_scale

    rng = np.random.default_rng(4)
    yy, xx = np.indices((600, 800))
    sky = (1.0 + 2.0 * np.exp(-((yy - 300) ** 2 + (xx - 400) ** 2) / (2 * 150 ** 2))).astype(np.float32)
    noise = rng.normal(0, 0.01, sky.shape).astype(np.float32)
    stripes = rng.normal(0, 0.02, (600, 1)).astype(np.float32) * np.ones((1, 800), np.float32)
    i2d = sky + noise + stripes
    own = (0.87 * sky + 0.1 + noise).astype(np.float32)
    own[100:104, 200:204] = np.nan                                      # всё отброшено в стеке
    own_wht = np.ones_like(own)
    own_wht[100:104, 200:204] = 0                                       # в стеке там ни одной экспозиции
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(own, name="SCI"), fits.ImageHDU(own_wht, name="WHT")]
                 ).writeto(tmp_path / "own.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(i2d, name="SCI"), fits.ImageHDU(np.ones_like(i2d), name="WHT")]
                 ).writeto(tmp_path / "i2d.fits")
    info = blend_large_scale(tmp_path / "own.fits", tmp_path / "i2d.fits", 64, tmp_path / "blend.fits")
    with fits.open(tmp_path / "blend.fits") as h:
        out = h["SCI"].data.astype(np.float64)
        fromi2d = h["FROMI2D"].data.astype(bool)
        wht = h["WHT"].data
    assert (wht[100:104, 200:204] > 0).all()                            # вес заполненных — от i2d (иначе align их обнулит)
    inner = (slice(150, 450), slice(150, 650))                          # вдали от краёв кадра
    # на масштабах много больше сигмы — как i2d (при сигме — остаток G(d − G d), он не ноль по построению)
    large = ndimage.gaussian_filter(out - i2d, 192)[inner]
    level = np.median(i2d[inner])                                       # как в критерии: доля яркости
    assert abs(np.median(large)) < 0.005 * level and np.percentile(np.abs(large), 95) < 0.01 * level
    # полосы — скачки медианы от строки к соседней строке (плавная кривизна пятна в разность почти не входит)
    rows_i2d = np.std(np.diff(np.median(i2d[inner], axis=1))) / np.sqrt(2)
    rows_out = np.std(np.diff(np.median(out[inner], axis=1))) / np.sqrt(2)
    assert rows_out < 0.3 * rows_i2d                                    # полос i2d в гибриде нет
    assert fromi2d[100:104, 200:204].all() and fromi2d.sum() == 16 and info["filled_from_i2d_px"] == 16
    assert np.allclose(out[100:104, 200:204], i2d[100:104, 200:204])


def test_blend_fields_do_not_change_digest_or_stack_key(tmp_path):
    from skyforge.config import Level2Config
    from skyforge.level2 import _key

    base, blend = _cfg(), _cfg(level2=Level2Config(blend_large_scale="i2d", blend_sigma_px=64))
    assert base.digest() == _cfg(level2=Level2Config()).digest() and blend.digest() != base.digest()
    f, i2d = tmp_path / "a_cal.fits", tmp_path / "a_i2d.fits"
    f.write_bytes(b"0")
    i2d.write_bytes(b"0")
    assert _key(base, [f], i2d) == _key(blend, [f], i2d)               # стек тот же — гибрид отдельным файлом
