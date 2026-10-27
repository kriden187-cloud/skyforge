"""Аудит 2026-10-27, этап 2: манифест исходных данных sources/<slug>.json и остановка при несовпадении."""

from __future__ import annotations

import json
import os

import pytest

from skyforge.config import Config
from skyforge.log import SkyforgeError
from skyforge.paths import Paths
from skyforge.sources import HashCache, lock_path, source_files, verify


def _project(tmp_path):
    """Проект с одним продуктом уровня 3 в манифесте fetch."""
    paths = Paths(tmp_path)
    cfg = Config(target="X", filters=["F150W"])
    f = tmp_path / "data" / "raw" / "jwst" / "x_i2d.fits"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"sci" * 1000)
    man = paths.manifest(cfg.slug, cfg.mission, cfg.instrument)
    man.parent.mkdir(parents=True)
    man.write_text(json.dumps({"products": {"F150W": {"path": "data/raw/jwst/x_i2d.fits", "source": "mast",
                                                      "data_uri": "mast:JWST/product/x_i2d.fits"}}}), encoding="utf-8")
    return paths, cfg, f


def _lock(paths, cfg):
    hc = HashCache(paths)
    files = [{**e, "size": paths.abs(e["path"]).stat().st_size, "sha256": hc.sha256(paths.abs(e["path"]))}
             for e in source_files(cfg, paths)]
    lp = lock_path(paths, cfg.slug)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps({"tag": "phase-x", "commit": "0" * 40, "files": files}), encoding="utf-8")


def test_verify_stops_on_changed_missing_or_unknown_file(tmp_path):
    paths, cfg, f = _project(tmp_path)
    assert verify(cfg, paths) == {"lock": None}                       # нет манифеста — новый объект, не проверяется
    _lock(paths, cfg)
    info = verify(cfg, paths)
    assert info["files"] == 1 and info["tag"] == "phase-x"
    assert source_files(cfg, paths)[0]["origin"] == "mast:JWST/product/x_i2d.fits"
    # архив обновил файл (тот же размер, другие байты) — остановка, а не обработка нового
    f.write_bytes(b"SCI" * 1000)
    os.utime(f, ns=(f.stat().st_atime_ns, f.stat().st_mtime_ns + 10**9))
    with pytest.raises(SkyforgeError, match="другой файл"):
        verify(cfg, paths)
    f.unlink()
    with pytest.raises(SkyforgeError, match="нет на диске"):
        verify(cfg, paths)
    # конвейер читает файл, которого нет в манифесте (добавлен фильтр) — тоже остановка
    f.write_bytes(b"sci" * 1000)
    man = paths.manifest(cfg.slug, cfg.mission, cfg.instrument)
    m = json.loads(man.read_text(encoding="utf-8"))
    g = f.with_name("y_i2d.fits")
    g.write_bytes(b"y")
    m["products"]["F200W"] = {"path": "data/raw/jwst/y_i2d.fits", "source": "mast"}
    man.write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(SkyforgeError, match="не в манифесте"):
        verify(cfg.model_copy(update={"filters": ["F150W", "F200W"]}), paths)


def test_hash_cache_recomputes_only_changed_files(tmp_path):
    paths, cfg, f = _project(tmp_path)
    hc = HashCache(paths)
    h1 = hc.sha256(f)
    hc.save()
    hc2 = HashCache(paths)
    assert hc2.sha256(f) == h1 and hc2.computed == 0
    f.write_bytes(b"other")
    assert hc2.sha256(f) != h1 and hc2.computed == 1
