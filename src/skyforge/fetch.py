"""MAST: поиск, выбор продуктов, скачивание в кеш. Плюс папка data/inbox/ для ручных файлов.

Порядок поиска файла для фильтра: inbox → уже скачанное (манифест) → MAST.
Сырые файлы после скачивания помечаются «только чтение» и больше не меняются.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import requests
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn
from rich.table import Table

from skyforge.config import Config, slugify
from skyforge.filters import effective_filter, sort_by_wavelength, wavelength_um
from skyforge.fitsio import read_info
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths

MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
DOWNLOAD_RETRIES = 5
CHUNK = 1 << 20
# Замер 2026-10-22 (300 МБ): одно соединение до MAST ~1 МБ/с и с прокси, и без; скорость соединений сильно разная
# (4 соединения: 18, 16 и 4 МБ/с), поэтому файл режется на куски PIECE, а соединения берут их из очереди —
# медленное задерживает один кусок, а не четверть файла.
PIECE = 32 << 20

INSTRUMENTS = {
    "jwst": {"nircam": "NIRCAM/IMAGE", "miri": "MIRI/IMAGE", "niriss": "NIRISS/IMAGE"},
    "hst": {
        "wfc3_uvis": "WFC3/UVIS", "wfc3/uvis": "WFC3/UVIS", "uvis": "WFC3/UVIS",
        "wfc3_ir": "WFC3/IR", "wfc3/ir": "WFC3/IR", "ir": "WFC3/IR",
        "acs": "ACS/WFC", "acs_wfc": "ACS/WFC", "acs/wfc": "ACS/WFC",
        # В MAST продукты HAP для WFPC2 записаны как WFPC2/PC, хотя это мозаика всех четырёх чипов.
        "wfpc2": "WFPC2/PC", "wfpc2/pc": "WFPC2/PC",
    },
}
# Подгруппа продукта уровня 3 в порядке предпочтения.
SUBGROUPS = {"jwst": ["I2D"], "hst": ["DRC", "DRZ", "DRW"]}
# У Hubble одно наблюдение бывает в нескольких обработках. HAP-SVM — один визит, все чипы в одной
# мозаике, астрометрия по Gaia (фильтры разных лет совпадают); старые CAL* — без общей привязки;
# HAP-MVM — огромные плитки неба из многих программ.
PROVENANCE_RANK = {"HAP-SVM": 2, "HAP-MVM": 0}


@dataclass
class Product:
    filter: str
    path: str  # относительно корня проекта
    source: str  # "mast" | "inbox"
    obs_id: str | None = None
    filename: str = ""
    size: int | None = None
    exptime: float | None = None
    release: str | None = None
    proposal_id: str | None = None
    data_uri: str | None = None


@dataclass
class Available:
    filter: str
    obs_id: str
    obsid: str
    exptime: float | None
    public: bool
    release: str | None
    proposal_id: str | None


@dataclass
class FetchResult:
    products: dict[str, Product]
    missing: list[str] = field(default_factory=list)
    available: dict[str, Available] = field(default_factory=dict)


def mast_instrument(cfg: Config) -> str:
    table = INSTRUMENTS.get(cfg.mission, {})
    key = cfg.instrument.lower()
    if key in table:
        return table[key]
    if key.upper() in table.values():  # уже в формате MAST: "NIRCAM/IMAGE"
        return key.upper()
    raise SkyforgeError(f"инструмент '{cfg.instrument}' для {cfg.mission} не знаю; варианты: {', '.join(table)}")


# --- выбор наблюдений и продуктов (чистые функции, покрыты тестами) ---


def _mjd_to_date(mjd: Any) -> str | None:
    if mjd is None:
        return None
    from astropy.time import Time

    return Time(float(mjd), format="mjd").to_value("iso", subfmt="date")


def choose_observations(rows: list[dict[str, Any]]) -> dict[str, Available]:
    """Одно наблюдение на фильтр: публичное важнее закрытого, затем лучшая обработка (для Hubble),
    затем самая длинная экспозиция."""
    best: dict[str, tuple[tuple, Available]] = {}
    for r in rows:
        f = effective_filter(r.get("filters") or "")
        if not f:
            continue
        public = str(r.get("dataRights") or "").upper() == "PUBLIC"
        exptime = float(r["t_exptime"]) if r.get("t_exptime") is not None else None
        item = Available(
            filter=f,
            obs_id=str(r["obs_id"]),
            obsid=str(r["obsid"]),
            exptime=exptime,
            public=public,
            release=_mjd_to_date(r.get("t_obs_release")),
            proposal_id=str(r["proposal_id"]) if r.get("proposal_id") is not None else None,
        )
        prov = str(r.get("provenance_name") or "").upper()
        rank = PROVENANCE_RANK.get(prov, 1)
        key = (public, rank, exptime or 0.0, item.obs_id)
        if f not in best or key > best[f][0]:
            best[f] = (key, item)
    return {f: v[1] for f, v in best.items()}


def choose_product(prods: list[dict[str, Any]], obs_id: str, mission: str, calib_level: int) -> dict[str, Any] | None:
    """Комбинированный продукт самого наблюдения: уровень 3, а не отдельные кадры уровня 2 с тем же суффиксом."""
    order = SUBGROUPS[mission]
    cands = [
        p for p in prods
        if str(p.get("productType")).upper() == "SCIENCE"
        and int(p.get("calib_level") or -1) == calib_level
        and str(p.get("productSubGroupDescription")).upper() in order
        and str(p.get("productFilename", "")).lower().startswith(obs_id.lower())
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: (order.index(str(p["productSubGroupDescription"]).upper()), -int(p.get("size") or 0)))
    return cands[0]


def _table_rows(table: Any) -> list[dict[str, Any]]:
    rows = []
    for rec in table:
        row = {}
        for col in table.colnames:
            v = rec[col]
            if hasattr(v, "mask") and bool(getattr(v, "mask", False)):
                v = None
            elif hasattr(v, "item"):
                v = v.item()
            if isinstance(v, bytes):
                v = v.decode()
            row[col] = v
        rows.append(row)
    return rows


# --- MAST ---


def query_mast(cfg: Config) -> list[dict[str, Any]]:
    from astroquery.mast import Observations

    kw: dict[str, Any] = dict(
        obs_collection=cfg.mission.upper(),
        instrument_name=mast_instrument(cfg),
        calib_level=cfg.calib_level,
        dataproduct_type="image",
        intentType="science",
        radius=cfg.radius,
    )
    if cfg.proposal_id:
        kw["proposal_id"] = cfg.proposal_id
    if cfg.mast_target:
        kw["target_name"] = cfg.mast_target
    if cfg.coords:
        kw["coordinates"] = cfg.coords
    else:
        kw["objectname"] = cfg.target
    return _table_rows(Observations.query_criteria(**kw))


def product_lists(obsids: list[str]) -> list[dict[str, Any]]:
    from astroquery.mast import Observations

    if not obsids:
        return []
    return _table_rows(Observations.get_product_list(obsids))


def _server_info(uri: str) -> tuple[int | None, bool]:
    """(размер, который отдаёт сервер — Content-Length на HEAD; отдаёт ли диапазоны — Accept-Ranges: bytes).
    (None, False) — если не узнать."""
    try:
        r = requests.head(MAST_DOWNLOAD_URL, params={"uri": uri}, timeout=(30, 60), allow_redirects=True)
        if not r.ok:
            return None, False
        size = int(r.headers["Content-Length"]) if r.headers.get("Content-Length") else None
        return size, r.headers.get("Accept-Ranges", "").lower() == "bytes"
    except (requests.RequestException, ValueError):
        return None, False


def split_ranges(size: int, piece: int | None = None) -> list[tuple[int, int]]:
    """Куски байтов [начало, конец] (конец включительно) по piece (по умолчанию PIECE) для очереди соединений."""
    step = piece or PIECE
    return [(a, min(a + step, size) - 1) for a in range(0, size, step)]


def _fetch_range(uri: str, path: Path, start: int, end: int, progress: Progress, task: Any) -> None:
    """Один диапазон в свой файл с докачкой: уже скачанное (размер файла) не повторяется."""
    total = end - start + 1
    last: Exception | None = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        have = path.stat().st_size if path.exists() else 0
        if have > total:
            path.unlink()
            have = 0
        if have == total:
            return
        try:
            with requests.get(MAST_DOWNLOAD_URL, params={"uri": uri}, headers={"Range": f"bytes={start + have}-{end}"},
                              stream=True, timeout=(30, 120)) as r:
                if r.status_code != 206:
                    raise IOError(f"сервер не отдал диапазон (HTTP {r.status_code})")
                with path.open("ab") as fh:
                    for chunk in r.iter_content(CHUNK):
                        fh.write(chunk)
                        progress.advance(task, len(chunk))
        except (requests.RequestException, IOError) as e:
            last = e
            console.print(f"  [yellow]{path.name}: обрыв ({e.__class__.__name__}), попытка {attempt}/{DOWNLOAD_RETRIES}[/]")
            time.sleep(min(30, 2 ** attempt))
    raise SkyforgeError(f"не удалось скачать {path.name}: {last}")


def _download_ranges(uri: str, dest: Path, size: int, ranges: list[tuple[int, int]], parallel: int,
                     progress: Progress, task: Any) -> None:
    """Куски файла из очереди в parallel соединений (<имя>.partN), затем склейка в <имя>.part, проверка размера и
    переименование."""
    from concurrent.futures import ThreadPoolExecutor

    pieces = [dest.with_name(f"{dest.name}.part{k}") for k in range(len(ranges))]
    progress.update(task, completed=sum(p.stat().st_size for p in pieces if p.exists()))
    with ThreadPoolExecutor(max(1, min(parallel, len(ranges)))) as ex:     # пул — очередь кусков
        list(ex.map(lambda k: _fetch_range(uri, pieces[k], *ranges[k], progress, task), range(len(ranges))))
    joined = dest.with_name(dest.name + ".part")
    with joined.open("wb") as out:
        for p in pieces:
            with p.open("rb") as fh:
                shutil.copyfileobj(fh, out, CHUNK)
    got = joined.stat().st_size
    if got != size:
        joined.unlink()
        raise SkyforgeError(f"{dest.name}: после склейки {got} байт вместо {size} — части оставлены для докачки")
    joined.replace(dest)
    os.chmod(dest, stat.S_IREAD)
    for p in pieces:
        p.unlink()


def download(uri: str, dest: Path, expected_size: int | None, parts: int = 1, progress: Progress | None = None) -> int:
    """Скачивание с докачкой и проверкой размера; возвращает размер файла. parts > 1 и сервер с Accept-Ranges — файл
    частями параллельно (одно соединение до MAST ~1 МБ/с, замер 2026-10-22). progress — общая полоса для нескольких
    файлов сразу; без неё — своя."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if progress is None:
        with Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                      TimeRemainingColumn(), console=console, transient=True) as own:
            return download(uri, dest, expected_size, parts, own)
    part = dest.with_name(dest.name + ".part")
    # Фаза 6: размер в списке продуктов MAST бывает старым — файл перепроцессирован (W3 F182M, CAL_VER 3.0.0, на 5760
    # байт больше), и проверка по нему отбрасывала целый файл пять раз подряд. Мерило — размер, который отдаёт сервер.
    server, ranges_ok = _server_info(uri) if (expected_size or parts > 1) else (None, False)
    if server and expected_size and server != expected_size:
        console.print(f"  [yellow]{dest.name}: в списке продуктов MAST {expected_size} байт, сервер отдаёт {server} — "
                      f"проверяю по серверу[/]")
    expected_size = server or expected_size
    task = progress.add_task(dest.name, total=expected_size)
    try:
        ranges = split_ranges(expected_size) if (parts > 1 and ranges_ok and expected_size) else []
        if len(ranges) > 1:
            _download_ranges(uri, dest, expected_size, ranges, parts, progress, task)
            return dest.stat().st_size
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            have = part.stat().st_size if part.exists() else 0
            if expected_size and have > expected_size:
                part.unlink()
                have = 0
            if not (expected_size and have == expected_size):
                headers = {"Range": f"bytes={have}-"} if have else {}
                try:
                    with requests.get(MAST_DOWNLOAD_URL, params={"uri": uri}, headers=headers,
                                      stream=True, timeout=(30, 120)) as r:
                        if r.status_code == 416:  # всё уже скачано
                            pass
                        else:
                            r.raise_for_status()
                            if have and r.status_code != 206:
                                have = 0  # сервер не умеет докачку — начинаем заново
                            progress.update(task, completed=have)
                            with part.open("ab" if have else "wb") as fh:
                                for chunk in r.iter_content(CHUNK):
                                    fh.write(chunk)
                                    progress.advance(task, len(chunk))
                except requests.RequestException as e:
                    last_error = e
                    console.print(f"  [yellow]обрыв ({e.__class__.__name__}), попытка {attempt}/{DOWNLOAD_RETRIES}[/]")
                    time.sleep(min(30, 2 ** attempt))
                    continue
            size = part.stat().st_size if part.exists() else 0
            if expected_size is None or size == expected_size:
                part.replace(dest)
                os.chmod(dest, stat.S_IREAD)
                return dest.stat().st_size
            last_error = IOError(f"размер {size} вместо {expected_size}")
            console.print(f"  [yellow]файл неполный ({size}/{expected_size} байт), докачиваю[/]")
        raise SkyforgeError(
            f"не удалось скачать {dest.name}: {last_error}. Можно скачать вручную с mast.stsci.edu "
            f"и положить в data/inbox/<объект>/"
        )
    finally:
        progress.remove_task(task)


def download_many(jobs: list[tuple[str, Path, int | None]], workers: int, parts: int,
                  on_done: Any = None) -> None:
    """Несколько файлов одновременно (до workers), каждый — частями (parts); общая полоса прогресса. on_done(i, размер)
    вызывается в главном потоке после каждого готового файла (манифест — после каждого: обрыв не теряет прогресс)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if not jobs:
        return
    with Progress(TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
                  TimeRemainingColumn(), console=console, transient=True) as progress, \
            ThreadPoolExecutor(max(1, min(workers, len(jobs)))) as ex:
        futs = {ex.submit(download, uri, dest, size, parts, progress): i for i, (uri, dest, size) in enumerate(jobs)}
        for fut in as_completed(futs):
            got = fut.result()
            if on_done is not None:
                on_done(futs[fut], got)


# --- inbox и манифест ---


def scan_inbox(cfg: Config, paths: Paths) -> dict[str, Product]:
    """FITS из data/inbox/<slug>/ и из корня inbox/, если в заголовке тот же объект."""
    files: list[tuple[Path, bool]] = []
    target_dir = paths.inbox / cfg.slug
    if target_dir.is_dir():
        files += [(p, True) for p in sorted(target_dir.rglob("*.fits"))]
    if paths.inbox.is_dir():
        files += [(p, False) for p in sorted(paths.inbox.glob("*.fits"))]
    inst_key = mast_instrument(cfg).split("/")[0].lower()
    found: dict[str, Product] = {}
    for p, in_target_dir in files:
        try:
            info = read_info(p)
        except Exception as e:  # битый или чужой файл не должен ронять конвейер
            console.print(f"  [yellow]inbox: пропускаю {p.name}: {e}[/]")
            continue
        if not in_target_dir and slugify(info.target or "-") != cfg.slug:
            continue
        if cfg.mission.upper() not in info.telescope or inst_key not in info.instrument.lower():
            continue
        if not info.filter:
            console.print(f"  [yellow]inbox: в {p.name} не нашёл имя фильтра в заголовке[/]")
            continue
        prod = Product(filter=info.filter, path=paths.rel(p), source="inbox", filename=p.name,
                       size=p.stat().st_size, exptime=info.exptime)
        if info.filter in found and (found[info.filter].size or 0) >= (prod.size or 0):
            continue
        found[info.filter] = prod
    return found


def load_manifest(cfg: Config, paths: Paths) -> dict[str, Any]:
    p = paths.manifest(cfg.slug, cfg.mission, cfg.instrument)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"target": cfg.target, "mission": cfg.mission, "instrument": cfg.instrument, "products": {}}


def save_manifest(cfg: Config, paths: Paths, manifest: dict[str, Any]) -> None:
    p = paths.manifest(cfg.slug, cfg.mission, cfg.instrument)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def local_products(cfg: Config, paths: Paths) -> dict[str, Product]:
    """Что уже есть на диске без сети: inbox важнее скачанного."""
    want = set(cfg.filters)
    found: dict[str, Product] = {}
    for f, d in load_manifest(cfg, paths).get("products", {}).items():
        prod = Product(**d)
        if prod.source == "mast" and paths.abs(prod.path).exists() and (not want or f in want):
            found[f] = prod
    for f, prod in scan_inbox(cfg, paths).items():
        if not want or f in want:
            found[f] = prod
    order = sort_by_wavelength(list(found), cfg.mission, cfg.instrument)
    return {f: found[f] for f in order}


def require_products(cfg: Config, paths: Paths, level3: bool = False) -> dict[str, Product]:
    """Файлы для конвейера. Фильтры из уровня 2 (source: level2 — все, иначе level2_filters, фаза 8) — своя сборка из
    экспозиций (этап build) вместо i2d; level3=True — всегда i2d (нужны самой сборке: сетка и маски)."""
    prods = local_products(cfg, paths)
    if cfg.any_level2() and not level3:
        from skyforge.level2 import built_products

        built = built_products(cfg, paths)
        lacking = [f for f in prods if cfg.uses_level2(f) and f not in built]
        if lacking:
            raise SkyforgeError(f"уровень 2 — нет своей сборки для {lacking}: сначала skyforge build")
        prods = {f: (Product(filter=f, path=paths.rel(built[f]), source="level2", obs_id=p.obs_id,
                             filename=built[f].name, size=built[f].stat().st_size, exptime=p.exptime,
                             release=p.release, proposal_id=p.proposal_id) if cfg.uses_level2(f) else p)
                 for f, p in prods.items()}
    if not prods:
        raise SkyforgeError(f"для {cfg.target} нет файлов: сначала skyforge fetch (или положи FITS в data/inbox/{cfg.slug}/)")
    missing = [f for f in cfg.filters if f not in prods]
    if missing:
        if not cfg.allow_missing:
            raise SkyforgeError(f"нет фильтров {missing}, а allow_missing: false")
        console.print(f"[yellow]Нет данных для {', '.join(missing)} — продолжаю без них (allow_missing)[/]")
    return prods


# --- главная функция ---


def _print_available(cfg: Config, avail: dict[str, Available], have: dict[str, Product]) -> None:
    t = Table(title=f"{cfg.target}: {cfg.mission.upper()} {mast_instrument(cfg)}, уровень {cfg.calib_level}")
    for col in ("фильтр", "λ, мкм", "наблюдение", "экспозиция, с", "доступ", "на диске"):
        t.add_column(col)
    for f in sort_by_wavelength(list(avail), cfg.mission, cfg.instrument):
        a = avail[f]
        access = "открыт" if a.public else f"[red]эксклюзив до {a.release}[/]"
        disk = have[f].source if f in have else ""
        want = "" if not cfg.filters or f in cfg.filters else " [dim](не в конфиге)[/]"
        t.add_row(f + want, f"{wavelength_um(f, cfg.mission, cfg.instrument):.3f}", a.obs_id,
                  f"{a.exptime:.0f}" if a.exptime else "—", access, disk)
    console.print(t)


def fetch(cfg: Config, paths: Paths, *, offline: bool = False, list_only: bool = False,
          max_gb: float | None = None) -> FetchResult:
    want = list(cfg.filters)
    found = local_products(cfg, paths)
    manifest = load_manifest(cfg, paths)
    for f, prod in found.items():
        console.print(f"  {f}: уже есть ({prod.source}) {prod.path}")

    need_network = list_only or not want or any(f not in found for f in want)
    avail: dict[str, Available] = {}
    if need_network and offline:
        console.print("[yellow]Режим offline: в MAST не хожу[/]")
    elif need_network:
        console.print(f"Ищу в MAST: {cfg.target} ({cfg.mission.upper()} {mast_instrument(cfg)})…")
        try:
            avail = choose_observations(query_mast(cfg))
        except SkyforgeError:
            raise
        except Exception as e:
            raise SkyforgeError(
                f"MAST не ответил: {e}\nПроверь HTTPS_PROXY в .env, или скачай FITS вручную в data/inbox/{cfg.slug}/"
            ) from e
        if not avail:
            console.print("[yellow]В MAST ничего не нашлось: проверь имя объекта, radius, инструмент[/]")
        _print_available(cfg, avail, found)
        manifest["available"] = {f: asdict(a) for f, a in avail.items()}

        todo = [f for f in (want or list(avail)) if f not in found and f in avail]
        locked = [f for f in todo if not avail[f].public]
        for f in locked:
            console.print(f"[red]{f}: данные под эксклюзивом до {avail[f].release}, пропускаю[/]")
        todo = [f for f in todo if avail[f].public]

        if todo and not list_only:
            prods = product_lists([avail[f].obsid for f in todo])
            picks: dict[str, dict[str, Any]] = {}
            for f in todo:
                p = choose_product(prods, avail[f].obs_id, cfg.mission, cfg.calib_level)
                if p is None:
                    console.print(f"[yellow]{f}: в наблюдении {avail[f].obs_id} нет продукта уровня {cfg.calib_level}[/]")
                else:
                    picks[f] = p
            total_gb = sum(int(p.get("size") or 0) for p in picks.values()) / 1e9
            console.print(f"Скачать: {len(picks)} файл(ов), {total_gb:.2f} ГБ")
            if max_gb is not None and total_gb > max_gb:
                raise SkyforgeError(f"{total_gb:.1f} ГБ больше лимита --max-gb {max_gb}; укажи --filters или подними лимит")
            order = list(picks)
            dests = {f: paths.raw / cfg.mission / avail[f].obs_id / picks[f]["productFilename"] for f in order}

            def record(f: str) -> None:
                a, p, dest = avail[f], picks[f], dests[f]
                prod = Product(filter=f, path=paths.rel(dest), source="mast", obs_id=a.obs_id,
                               filename=p["productFilename"], size=dest.stat().st_size, exptime=a.exptime,
                               release=a.release, proposal_id=a.proposal_id, data_uri=p["dataURI"])
                found[f] = prod
                manifest["products"][f] = asdict(prod)
                save_manifest(cfg, paths, manifest)  # после каждого файла: обрыв не теряет прогресс

            jobs = [f for f in order if not dests[f].exists()]
            for f in order:
                if f not in jobs:
                    record(f)
            for f in jobs:
                console.print(f"  {f}: {picks[f]['productFilename']} ({int(picks[f]['size']) / 1e6:.0f} МБ)")
            if jobs:
                console.print(f"  одновременно файлов: {min(cfg.fetch.workers, len(jobs))}, частей на файл: до {cfg.fetch.parts}")
            download_many([(picks[f]["dataURI"], dests[f], int(picks[f]["size"]) if picks[f].get("size") else None)
                           for f in jobs], cfg.fetch.workers, cfg.fetch.parts, lambda i, _size: record(jobs[i]))

    save_manifest(cfg, paths, manifest)
    missing = [f for f in want if f not in found]
    if missing and not list_only:
        hint = f"положи FITS вручную в data/inbox/{cfg.slug}/"
        if not cfg.allow_missing:
            raise SkyforgeError(f"нет фильтров {missing}; {hint}")
        console.print(f"[yellow]Нет данных: {', '.join(missing)} ({hint})[/]")
    order = sort_by_wavelength(list(found), cfg.mission, cfg.instrument)
    return FetchResult(products={f: found[f] for f in order}, missing=missing, available=avail)


# --- уровень 2: отдельные экспозиции (_cal.fits) ---


def level2_dir(cfg: Config, paths: Paths) -> Path:
    return paths.raw / cfg.slug / "l2"


def level2_manifest(cfg: Config, paths: Paths) -> Path:
    return paths.raw / "manifests" / f"{cfg.slug}_{cfg.mission}_{cfg.instrument}_l2.json"


def level2_files(cfg: Config, paths: Paths) -> dict[str, list[Path]]:
    """Скачанные _cal.fits по фильтрам (по манифесту; только те, что на диске и правильного размера)."""
    p = level2_manifest(cfg, paths)
    if not p.exists():
        return {}
    man = json.loads(p.read_text(encoding="utf-8"))
    out: dict[str, list[Path]] = {}
    for f, items in man.get("files", {}).items():
        if cfg.filters and f not in cfg.filters:
            continue
        files = [paths.abs(it["path"]) for it in items
                 if paths.abs(it["path"]).exists() and paths.abs(it["path"]).stat().st_size == it["size"]]
        if files:
            out[f] = sorted(files)
    return out


def fetch_level2(cfg: Config, paths: Paths, *, list_only: bool = False, max_gb: float | None = None) -> dict[str, Any]:
    """Продукты уровня 2 (_cal.fits) тех же наблюдений, из которых собраны мозаики уровня 3."""
    if cfg.mission != "jwst":
        raise SkyforgeError("source: level2 пока только для JWST (_cal.fits)")
    man3 = load_manifest(cfg, paths)
    avail = man3.get("available", {})
    want = [f for f in (cfg.filters or list(avail)) if cfg.uses_level2(f)]
    missing = [f for f in want if f not in avail]
    if missing:
        raise SkyforgeError(f"нет сведений о наблюдениях для {missing}: сначала skyforge fetch (уровень 3)")
    prods = product_lists([avail[f]["obsid"] for f in want])
    obs_of = {avail[f]["obsid"]: f for f in want}
    picks: dict[str, dict[str, dict[str, Any]]] = {f: {} for f in want}
    for p in prods:
        if (str(p.get("productSubGroupDescription")).upper() == "CAL" and int(p.get("calib_level") or -1) == 2
                and str(p.get("productType")).upper() == "SCIENCE"):
            f = obs_of.get(str(p.get("parent_obsid")))
            if f:
                picks[f][p["productFilename"]] = p
    total = sum(int(p["size"]) for d in picks.values() for p in d.values())
    t = Table(title=f"{cfg.target}: уровень 2 (_cal.fits)")
    for col in ("фильтр", "файлов", "ГБ"):
        t.add_column(col)
    for f in want:
        t.add_row(f, str(len(picks[f])), f"{sum(int(p['size']) for p in picks[f].values()) / 1e9:.2f}")
    t.add_row("всего", str(sum(len(d) for d in picks.values())), f"{total / 1e9:.2f}")
    console.print(t)
    if list_only:
        return {"files": {f: len(d) for f, d in picks.items()}, "gb": round(total / 1e9, 2)}
    if max_gb is not None and total / 1e9 > max_gb:
        raise SkyforgeError(f"уровень 2: {total / 1e9:.1f} ГБ больше лимита --max-gb {max_gb}")
    d = level2_dir(cfg, paths)
    mp = level2_manifest(cfg, paths)
    man = json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {"files": {}}
    for f in want:
        # готовый файл появляется только после проверки размера по серверу — есть файл, значит скачан; размер в манифест —
        # фактический (в списке продуктов MAST бывает старый, фаза 6)
        names = sorted(picks[f])
        jobs = [n for n in names if not (d / n).exists()]
        for n in jobs:
            console.print(f"  {f}: {n} ({int(picks[f][n]['size']) / 1e6:.0f} МБ)")
        download_many([(picks[f][n]["dataURI"], d / n, int(picks[f][n]["size"])) for n in jobs],
                      cfg.fetch.workers, cfg.fetch.parts)
        man["files"][f] = [{"path": paths.rel(d / n), "size": (d / n).stat().st_size} for n in names]
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"files": {f: len(v) for f, v in man["files"].items()}, "gb": round(total / 1e9, 2)}
