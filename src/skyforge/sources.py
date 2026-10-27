"""Манифест исходных данных принятого результата (аудит 2026-10-27, этап 2): sources/<slug>.json в репозитории.

Принятая картинка воспроизводима, только пока на диске лежат ровно те исходные файлы, из которых она собрана: архив может
обновить продукт (новая калибровка MAST, новый релиз ESO), и скачанный заново файл даст другую картинку. Второй копии данных
нет (решение владельца 2026-10-24) — исходники только в data/raw. Поэтому манифест фиксирует sha256 каждого файла, который
читает конвейер, и тег с коммитом принятого результата; этап fetch сверяет файлы с манифестом и при несовпадении или
отсутствии файла останавливает обработку, а не берёт обновлённый.

Что входит: продукты уровня 3 из манифеста fetch (i2d, drz, продукты ESO), исходные вырезки ESO (.fits.fz, из них собраны
продукты), экспозиции уровня 2 (_cal) для фильтров из уровня 2 и кеш звёзд Gaia (level2.astrometry: gaia). Хеши файлов
кешируются по (размер, время изменения) в data/cache/sha256.json — повторная сверка не читает гигабайты.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths

SOURCES_VERSION = 1
CHUNK = 8 << 20


def lock_path(paths: Paths, slug: str) -> Path:
    return paths.root / "sources" / f"{slug}.json"


def _cache_path(paths: Paths) -> Path:
    return paths.root / "data" / "cache" / "sha256.json"


class HashCache:
    """sha256 файлов по (путь, размер, mtime_ns): пересчёт только у изменившихся файлов."""

    def __init__(self, paths: Paths) -> None:
        self.paths = paths
        self.p = _cache_path(paths)
        self.data: dict[str, dict[str, Any]] = json.loads(self.p.read_text(encoding="utf-8")) if self.p.exists() else {}
        self.computed = 0

    def sha256(self, path: Path) -> str:
        st = path.stat()
        key = self.paths.rel(path)
        e = self.data.get(key)
        if e and e["size"] == st.st_size and e["mtime_ns"] == st.st_mtime_ns:
            return e["sha256"]
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
        self.data[key] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": h.hexdigest()}
        self.computed += 1
        return self.data[key]["sha256"]

    def save(self) -> None:
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.p.write_text(json.dumps(self.data, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def source_files(cfg: Config, paths: Paths) -> list[dict[str, str]]:
    """Файлы, которые читает конвейер для этого конфига: [{path, role, filter, origin}] (пути — от корня проекта)."""
    from skyforge.fetch import level2_manifest, load_manifest

    out: list[dict[str, str]] = []
    prods = load_manifest(cfg, paths).get("products", {})
    for f in cfg.filters:
        p = prods.get(f)
        if p is None:
            continue
        if p.get("source") == "eso":
            from skyforge.eso import _safe

            out.append({"path": p["path"], "role": "продукт ESO для конвейера (SCI/WHT из вырезки)", "filter": f,
                        "origin": f"ESO {p['obs_id']}"})
            cut = Path(p["path"]).with_name(f"{_safe(p['obs_id'])}_cut.fits.fz")
            out.append({"path": cut.as_posix(), "role": "вырезка SODA с сервера ESO", "filter": f,
                        "origin": f"ESO {p['obs_id']}"})
        else:
            role = "уровень 3 (i2d)" if p["path"].endswith("_i2d.fits") else "уровень 3 (drz)" \
                if p["path"].endswith("_drz.fits") else "уровень 3"
            out.append({"path": p["path"], "role": role, "filter": f, "origin": p.get("data_uri") or p.get("obs_id", "")})
    if cfg.any_level2():
        mp = level2_manifest(cfg, paths)
        man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"files": {}}
        for f, items in man.get("files", {}).items():
            if f not in cfg.filters or not cfg.uses_level2(f):
                continue
            for it in items:
                out.append({"path": it["path"], "role": "уровень 2 (_cal)", "filter": f,
                            "origin": f"mast:JWST/product/{Path(it['path']).name}"})
        if cfg.level2.astrometry == "gaia":
            from skyforge.astrometry import gaia_cache

            out.append({"path": paths.rel(gaia_cache(cfg, paths)), "role": "кеш звёзд Gaia DR3 (астрометрия)",
                        "filter": "", "origin": "Gaia DR3 (запрос astroquery, кеш)"})
    seen, uniq = set(), []
    for e in out:
        if e["path"] not in seen:
            seen.add(e["path"])
            uniq.append(e)
    return uniq


def _git(paths: Paths, *args: str) -> str:
    r = subprocess.run(["git", *args], cwd=paths.root, capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise SkyforgeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def write_lock(cfgs: list[Config], paths: Paths, tag: str) -> Path:
    """sources/<slug>.json по всем стилям объекта: файлы, sha256, тег и коммит принятого результата, файлы baseline."""
    from PIL import Image

    from skyforge.baseline import baseline_dir
    from skyforge.publish import pixels_sha256

    slug = cfgs[0].slug
    commit = _git(paths, "rev-list", "-n", "1", tag)
    hc = HashCache(paths)
    files: dict[str, dict[str, Any]] = {}
    for cfg in cfgs:
        for e in source_files(cfg, paths):
            p = paths.abs(e["path"])
            if not p.exists():
                raise SkyforgeError(f"{slug}: исходного файла нет на диске — {e['path']}; манифест не записан")
            if e["path"] not in files:
                files[e["path"]] = {**e, "size": p.stat().st_size, "sha256": hc.sha256(p)}
    hc.save()
    base = []
    Image.MAX_IMAGE_PIXELS = None
    for png in sorted(baseline_dir(paths, slug, tag).glob("*.png")):
        with Image.open(png) as im:
            pix = pixels_sha256(im)
        base.append({"path": paths.rel(png), "file_sha256": hc.sha256(png), "pixels_sha256": pix})
    hc.save()
    data = {"version": SOURCES_VERSION, "slug": slug, "tag": tag, "commit": commit,
            "styles": sorted({c.style for c in cfgs if c.style}),
            "storage": "data/raw (одна копия, решение владельца 2026-10-24)",
            "total_bytes": sum(e["size"] for e in files.values()), "files": sorted(files.values(), key=lambda e: e["path"]),
            "baseline": base}
    lp = lock_path(paths, slug)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return lp


def verify(cfg: Config, paths: Paths) -> dict[str, Any]:
    """Сверка файлов, которые прочитает конвейер, с sources/<slug>.json. Нет манифеста — ничего не проверяется (новый
    объект до приёмки). Файл не в манифесте, отсутствует или с другим sha256 — SkyforgeError: обработка останавливается."""
    lp = lock_path(paths, cfg.slug)
    if not lp.exists():
        return {"lock": None}
    lock = json.loads(lp.read_text(encoding="utf-8"))
    known = {e["path"]: e for e in lock["files"]}
    hc = HashCache(paths)
    bad: list[str] = []
    n = size = 0
    try:
        for e in source_files(cfg, paths):
            want = known.get(e["path"])
            p = paths.abs(e["path"])
            if want is None:
                bad.append(f"не в манифесте: {e['path']}")
            elif not p.exists():
                bad.append(f"нет на диске: {e['path']} (sha256 {want['sha256'][:16]}…)")
            elif p.stat().st_size != want["size"] or hc.sha256(p) != want["sha256"]:
                bad.append(f"другой файл: {e['path']} — размер {p.stat().st_size} / {want['size']}, "
                           f"sha256 {hc.sha256(p)[:16]}… / {want['sha256'][:16]}…")
            else:
                n += 1
                size += want["size"]
    finally:
        hc.save()
    if bad:
        raise SkyforgeError(f"{cfg.slug}: исходные файлы не совпадают с {paths.rel(lp)} (принято: тег {lock['tag']}, "
                            f"коммит {lock['commit'][:12]}) — обработка остановлена:\n  " + "\n  ".join(bad[:20])
                            + (f"\n  … ещё {len(bad) - 20}" if len(bad) > 20 else ""))
    console.print(f"  исходники: {n} файлов, {size / 1e9:.1f} ГБ совпадают с {paths.rel(lp)} "
                  f"(тег {lock['tag']}, коммит {lock['commit'][:12]}; хешей посчитано заново {hc.computed})")
    return {"lock": paths.rel(lp), "tag": lock["tag"], "commit": lock["commit"], "files": n, "bytes": size,
            "hashed": hc.computed}
