"""Фаза 4: поиск нетронутых объектов — `skyforge discover`.

MAST: наблюдения JWST (по умолчанию NIRCam, уровень 3), у которых эксклюзив истёк не раньше `since` и уже открыт.
Строки MAST (одна на продукт i2d: фильтр, наблюдение) группируются по (программа, цель): одна строка на объект.
Отбор: фильтров не меньше min_filters, хотя бы один широкополосный, площадь поля (объединение многоугольников
s_region всех фильтров) не меньше min_area, объект протяжённый — тип цели и заголовок программы не из списков
исключений (одиночные звёзды, Солнечная система, калибровка, пустые поля глубоких обзоров, далёкие квазары…).

Сверка с пресс-релизами: цели картинок webbtelescope.org (ныне science.nasa.gov/asset/webb) и esawebb.org — имя
и координаты, совпадение по каталожному имени или ближе press_radius_arcmin. Уже обработанное официально — в конец
списка с пометкой. Галерея сохраняется в data/reference/press_targets.json с датой: запуск воспроизводится без сети.

Разбор ответа MAST, группировка, площадь и сверка — чистые функции (тест на сохранённом образце, без сети).
"""

from __future__ import annotations

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from skyforge.filters import effective_filter
from skyforge.log import SkyforgeError, console

MAST_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
MAST_PORTAL = "https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html?searchQuery="
UA = {"User-Agent": "skyforge-discover/1.0 (research; contact via github)"}


class DiscoverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission: str = "JWST"
    instrument: str = "NIRCAM/IMAGE"
    calib_level: int = 3
    # эксклюзив истёк не раньше этой даты (ISO) или за последние since_months месяцев от даты запуска
    since: date | None = None
    since_months: int = Field(12, ge=1)
    min_filters: int = Field(3, ge=1)
    require_wide: bool = True
    min_area_arcmin2: float = Field(4.0, ge=0)
    top: int = Field(10, ge=1)
    press_radius_arcmin: float = Field(3.0, gt=0)
    press_file: str = "data/reference/press_targets.json"
    # протяжённый объект: тип цели (первое слово target_classification) не из этого списка...
    exclude_classes: list[str] = ["Solar System", "Calibration", "Star"]
    # пустые поля глубоких обзоров: тип «…; Blank field» (MAST ставит его и туманностям — их спасают keep_*)
    exclude_class_words: list[str] = ["Blank field"]
    # ...кроме типов с этими словами (звёзды с протяжённым окружением: протозвёзды, диски, туманности)
    keep_class_words: list[str] = ["Protostar", "Herbig-Haro", "Nebula", "Star-forming", "Star forming", "Outflow",
                                   "H II", "HII", "Molecular", "Stellar Cluster", "Cluster", "Wolf-Rayet",
                                   "Supernova remnant", "Planetary"]
    # исключение по типу снимается, если в заголовке/имени цели есть слова протяжённой цели или каталожное имя
    # (HH, NGC, IC, M, W, IRAS, Sh2, RCW, LDN, LBN…): у HH 211 в MAST тип «Unidentified; Blank field»
    keep_words: list[str] = ["nebula", "herbig", "outflow", "star-forming", "star forming", "molecular cloud",
                             "h ii", "hii region", "pillar", "supernova remnant", "planetary nebula", "globule",
                             "bubble", "protostar", "photodissociation", "pdr"]
    # в заголовке программы / имени цели нет слов, указывающих на точечную или пустую цель (проверяется первым)
    exclude_words: list[str] = [
        "calibration", "flux standard", "astrometric", "blank field", "deep field", "pointing", "parallel",
        "quasar", "qso", "redshift", "high-z", "z=", "z~", "reionization", "lyman", "lensed", "supernova",
        r"\bsn\d", r"\bgrb", "transient", "exoplanet", "brown dwarf", "comet", "asteroid", "kuiper",
        "moving target", "trans-neptunian", "occultation", "coronagraph", "shadow", "white dwarf",
        "binary", "flare", "afterglow", "tidal disruption", "host galaxy", "cosmos", "goods", "ceers", "jades",
        "primer", r"\buds\b", r"\begs\b", "hudf", "frontier field", "dark matter", "weak lensing",
    ]
    # не больше стольких объектов одной программы в списке (обзоры из десятков однотипных целей)
    per_program: int = Field(2, ge=1)
    # превью (полноразмерный JPEG i2d) качаются для стольких лучших — второй лист: структуру видно только глазами
    shortlist: int = Field(30, ge=1)


def load_config(path: Path) -> DiscoverConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return DiscoverConfig(**(data or {}))


def since_date(cfg: DiscoverConfig, today: date) -> date:
    if cfg.since:
        return cfg.since
    y, m = today.year, today.month - cfg.since_months
    while m <= 0:
        y, m = y - 1, m + 12
    day = min(today.day, 28)
    return date(y, m, day)


def mjd(d: date) -> float:
    return (datetime(d.year, d.month, d.day) - datetime(1858, 11, 17)).days


def mjd_to_date(v: float) -> date:
    return (datetime(1858, 11, 17) + timedelta(days=float(v))).date()


# --- MAST ---


def query_mast(cfg: DiscoverConfig, today: date) -> list[dict[str, Any]]:
    """Наблюдения уровня 3, открытые между since и сегодня. Сеть — через HTTPS_PROXY окружения."""
    from astroquery.mast import Observations

    from skyforge.fetch import _table_rows

    try:
        tab = Observations.query_criteria(
            obs_collection=cfg.mission, instrument_name=cfg.instrument, calib_level=cfg.calib_level,
            dataproduct_type="image", intentType="science",
            t_obs_release=[mjd(since_date(cfg, today)), mjd(today) + 1])
    except Exception as e:  # noqa: BLE001 — сеть/сервер: честная ошибка, не пустой список
        raise SkyforgeError(f"MAST не ответил: {e.__class__.__name__}: {e}\nПроверь HTTPS_PROXY (как для fetch)") from e
    return _table_rows(tab)


_POLY = re.compile(r"POLYGON\s*(?:ICRS|J2000)?\s*((?:[-+]?\d+(?:\.\d+)?\s*)+)", re.I)


def polygons(s_region: str) -> list[np.ndarray]:
    """'POLYGON ra dec ra dec … POLYGON …' → список массивов (N, 2) в градусах."""
    out = []
    for m in _POLY.finditer(s_region or ""):
        v = np.array(m.group(1).split(), dtype=float)
        if len(v) >= 6 and len(v) % 2 == 0:
            out.append(v.reshape(-1, 2))
    return out


def union_area_arcmin2(polys: list[np.ndarray], step_arcsec: float = 3.0) -> float:
    """Площадь объединения многоугольников неба: касательная плоскость в центре, растр step_arcsec."""
    if not polys:
        return 0.0
    from matplotlib.path import Path as MPath

    allp = np.vstack(polys)
    ra0, dec0 = float(np.mean(allp[:, 0])), float(np.mean(allp[:, 1]))
    cosd = math.cos(math.radians(dec0))

    def proj(p):
        dra = (p[:, 0] - ra0 + 540) % 360 - 180
        return np.stack([dra * cosd * 3600, (p[:, 1] - dec0) * 3600], -1)

    xy = [proj(p) for p in polys]
    lo, hi = np.min(np.vstack(xy), 0), np.max(np.vstack(xy), 0)
    gx = np.arange(lo[0], hi[0] + step_arcsec, step_arcsec)
    gy = np.arange(lo[1], hi[1] + step_arcsec, step_arcsec)
    if gx.size * gy.size > 4_000_000:                       # огромная мозаика — растр грубее
        return union_area_arcmin2(polys, step_arcsec * 2)
    X, Y = np.meshgrid(gx, gy)
    pts = np.stack([X.ravel(), Y.ravel()], -1)
    inside = np.zeros(len(pts), dtype=bool)
    for p in xy:
        inside |= MPath(p).contains_points(pts)
    return float(inside.sum() * step_arcsec ** 2 / 3600)


def is_wide(name: str) -> bool:
    return bool(re.match(r"^F\d{3,4}W2?$", name))


@dataclass
class Candidate:
    target: str
    proposal_id: str
    title: str
    pi: str
    classification: str
    ra: float
    dec: float
    filters: list[str]
    area_arcmin2: float
    release: str                     # дата снятия эксклюзива (самая ранняя по продуктам), ISO
    obs_ids: list[str]
    preview_uri: str | None          # jpegURL одного i2d (широкий фильтр наибольшей площади)
    preview_filter: str | None
    kind: str = "протяжённый"        # протяжённый | исключён: причина
    press: list[str] = field(default_factory=list)   # совпадения с пресс-релизами: «источник: имя (расстояние)»
    previews: list = field(default_factory=list)      # запасные превью [(uri, фильтр)]

    @property
    def mast_url(self) -> str:
        return f"{MAST_PORTAL}{self.ra:.5f}%20{self.dec:.5f}"

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "-", f"{self.target}-{self.proposal_id}".lower()).strip("-")


def _kind(cls: str, title: str, target: str, cfg: DiscoverConfig) -> str:
    """Сначала слова-исключения в заголовке/имени (пустые поля, точечные цели), затем тип цели — если его не
    снимают слова протяжённой цели в типе, заголовке или каталожное имя цели."""
    head = (cls or "").split(";")[0].strip()
    text = f"{title} {target}".lower()
    for w in cfg.exclude_words:
        if re.search(w if w.startswith("\\") else re.escape(w), text):
            return f"исключён: «{w.replace(chr(92) + 'b', '').replace(chr(92) + 'd', '#')}» в заголовке/имени"
    keep = (any(w.lower() in (cls or "").lower() for w in cfg.keep_class_words)
            or any(w in text for w in cfg.keep_words) or bool(catalog_ids(target)))
    if not keep:
        if head and any(head.lower().startswith(c.lower()) for c in cfg.exclude_classes):
            return f"исключён: тип «{head}»"
        for w in cfg.exclude_class_words:
            if w.lower() in (cls or "").lower():
                return f"исключён: тип «{w}»"
    return "протяжённый"


def group_rows(rows: Iterable[dict[str, Any]], cfg: DiscoverConfig) -> list[Candidate]:
    """Строки MAST → кандидаты: одна на (программа, цель); без отбора — отбор в select()."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        if str(r.get("dataRights", "PUBLIC")).upper() != "PUBLIC":
            continue
        key = (str(r.get("proposal_id")), str(r.get("target_name")))
        groups.setdefault(key, []).append(r)
    out = []
    for (pid, target), rs in groups.items():
        filt = {}
        for r in rs:
            f = effective_filter(str(r.get("filters", "")))
            if f:
                filt.setdefault(f, []).append(r)
        polys = [p for r in rs for p in polygons(str(r.get("s_region", "")))]
        ra = float(np.mean([float(r["s_ra"]) for r in rs]))
        dec = float(np.mean([float(r["s_dec"]) for r in rs]))
        # превью: широкие фильтры по убыванию площади поля, затем остальные — запасные, если первое не скачалось
        opts = []
        for f, fr in filt.items():
            for r in fr:
                if r.get("jpegURL"):
                    opts.append((not is_wide(f), -union_area_arcmin2(polygons(str(r.get("s_region", ""))), 6.0),
                                 str(r["jpegURL"]), f))
        opts.sort()
        previews = [(u, f) for _, _, u, f in opts]
        cls = str(rs[0].get("target_classification") or "")
        cls = "" if cls in ("--", "None", "nan") else cls
        title = str(rs[0].get("obs_title") or "")
        c = Candidate(
            target=target, proposal_id=pid, title=title, pi=str(rs[0].get("proposal_pi") or ""), classification=cls,
            ra=ra, dec=dec, filters=sorted(filt, key=lambda n: int(re.sub(r"\D", "", n)[:4] or 0)),
            area_arcmin2=round(union_area_arcmin2(polys), 2),
            release=mjd_to_date(min(float(r["t_obs_release"]) for r in rs)).isoformat(),
            obs_ids=sorted({str(r.get("obs_id")) for r in rs}),
            preview_uri=previews[0][0] if previews else None, preview_filter=previews[0][1] if previews else None,
            previews=previews[1:4])
        c.kind = _kind(cls, title, target, cfg)
        out.append(c)
    return out


def select(cands: list[Candidate], cfg: DiscoverConfig) -> tuple[list[Candidate], dict[str, int]]:
    """Отбор по правилам конфига; счётчики отсева по причинам."""
    why: dict[str, int] = {}
    keep = []
    for c in cands:
        reason = None
        if len(c.filters) < cfg.min_filters:
            reason = f"фильтров < {cfg.min_filters}"
        elif cfg.require_wide and not any(is_wide(f) for f in c.filters):
            reason = "нет широкополосного"
        elif c.area_arcmin2 < cfg.min_area_arcmin2:
            reason = f"площадь < {cfg.min_area_arcmin2:g} кв. угл. мин"
        elif c.kind != "протяжённый":
            reason = "не протяжённый (тип или заголовок)"
        if reason:
            why[reason] = why.get(reason, 0) + 1
        else:
            keep.append(c)
    return keep, why


# --- пресс-релизы ---

_CAT = re.compile(r"\b(NGC|IC|M|MESSIER|HH|IRAS|SH\s*2|SH2|RCW|LDN|LBN|ESO|UGC|PGC|W|WR|N|G|J)\s*[-_ ]?\s*"
                  r"(\d+(?:[A-Z](?![A-Z]))?(?:[-+]\d+)?)", re.I)   # W51A, но HH211NIRCAM → HH211


def catalog_ids(name: str) -> set[str]:
    """Каталожные имена в строке: 'Messier 16' → {'M16'}, 'NGC-346' → {'NGC346'}, 'W51A-NIRCAM' → {'W51A'}."""
    out = set()
    for m in _CAT.finditer(name or ""):
        cat = re.sub(r"\s+", "", m.group(1).upper()).replace("MESSIER", "M").replace("SH2", "SH2")
        if cat in ("N", "G", "J") and len(m.group(2)) < 3:        # N, G, J — только с длинным номером (N44 — нет)
            continue
        out.add(f"{cat}{m.group(2).upper()}")
    return out


def _sep_arcmin(ra1, dec1, ra2, dec2) -> float:
    r1, d1, r2, d2 = map(math.radians, (ra1, dec1, ra2, dec2))
    c = math.sin(d1) * math.sin(d2) + math.cos(d1) * math.cos(d2) * math.cos(r1 - r2)
    return math.degrees(math.acos(max(-1.0, min(1.0, c)))) * 60


def match_press(c: Candidate, press: list[dict[str, Any]], radius_arcmin: float) -> list[str]:
    ids = catalog_ids(c.target) | catalog_ids(c.title)
    hits = []
    for p in press:
        why = []
        if p.get("ra") is not None and p.get("dec") is not None:
            d = _sep_arcmin(c.ra, c.dec, p["ra"], p["dec"])
            if d <= radius_arcmin:
                why.append(f"{d:.1f}′")
        pid = catalog_ids(" ".join(p.get("names", [])) + " " + p.get("title", ""))
        if ids & pid:
            why.append("имя " + ",".join(sorted(ids & pid)))
        if why:
            hits.append(f"{p['source']}: {p.get('title') or ', '.join(p.get('names', []))} ({'; '.join(why)})")
    return hits


def cap_per_program(cands: list[Candidate], n: int) -> list[Candidate]:
    """Не больше n объектов одной программы (порядок сохраняется): обзоры из десятков однотипных целей (J-Virgo —
    ~70 галактик, VENUS — ~40 скоплений) иначе занимают весь список."""
    seen: dict[str, int] = {}
    out = []
    for c in cands:
        seen[c.proposal_id] = seen.get(c.proposal_id, 0) + 1
        if seen[c.proposal_id] <= n:
            out.append(c)
    return out


def rank(cands: list[Candidate]) -> list[Candidate]:
    """Критерии владельца: без пресс-релиза — вперёд; затем больше фильтров, больше площадь. «Видимая структура на
    превью» — глазами по листу: пиксельная метрика (доля превью выше фона) звёздные и галактические поля от туманностей
    не отделяла (поля карликовых галактик 64–66%, протозвезда HOPS-373 4%) — из ранжирования убрана."""
    return sorted(cands, key=lambda c: (bool(c.press), -len(c.filters), -c.area_arcmin2))


# --- галерея пресс-релизов (сеть, один раз) ---


def _text(html_s: str) -> str:
    import html

    s = re.sub(r"<script.*?</script>|<style.*?</style>", " ", html_s, flags=re.S | re.I)
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)))


def _hms(ra: str, dec: str) -> tuple[float, float] | None:
    try:
        rh = [float(x) for x in re.split(r"[:\s]+", ra.strip())[:3]]
        sgn = -1 if dec.strip()[0] in "-−" else 1
        dd = [abs(float(x)) for x in re.split(r"[:\s°'\"″′]+", dec.strip().lstrip("+-−"))[:3] if x]
        while len(dd) < 3:
            dd.append(0.0)
        return 15 * (rh[0] + rh[1] / 60 + rh[2] / 3600), sgn * (dd[0] + dd[1] / 60 + dd[2] / 3600)
    except (ValueError, IndexError):
        return None


def parse_esa_page(html_s: str, url: str) -> dict[str, Any] | None:
    t = _text(html_s)
    m_ra = re.search(r"Position \(RA\):\s*([\d\s.]+?)\s+Position \(Dec\):\s*([-+−]?\d+°\s*\d+'\s*[\d.]+\"?)", t)
    if not m_ra:
        return None
    pos = _hms(m_ra.group(1), m_ra.group(2))
    names = []
    m_n = re.search(r"Name:\s*(.*?)\s+(?:Distance|Constellation|Type|Category|Position)", t)
    if m_n:
        names = [n.strip() for n in m_n.group(1).split(",") if n.strip()]
    title = re.search(r"<title>\s*(.*?)\s*(?:\||</title>)", html_s, re.S)
    rel = re.search(r"Release date:\s*(\d{1,2} \w+ \d{4})", t)
    return {"source": "esawebb", "url": url, "title": title.group(1).strip() if title else "", "names": names,
            "ra": pos[0] if pos else None, "dec": pos[1] if pos else None, "release": rel.group(1) if rel else None}


def parse_nasa_page(html_s: str, url: str) -> dict[str, Any] | None:
    t = _text(html_s)
    m_ra = re.search(r"R\.A\. Position.*?(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", t)
    m_de = re.search(r"Dec\. Position.*?([-+−]?\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", t)
    if not (m_ra and m_de):
        return None
    pos = _hms(m_ra.group(1), m_de.group(1))
    m_n = re.search(r"identify an astronomical object\.\s*(.*?)\s+Object Description", t)
    title = re.search(r"<title>\s*(.*?)\s*(?:-\s*NASA Science)?\s*</title>", html_s, re.S)
    rel = re.search(r"Release Date\s+(\w+ \d{1,2}, \d{4})", t)
    return {"source": "webbtelescope/nasa", "url": url, "title": title.group(1).strip() if title else "",
            "names": [n.strip() for n in (m_n.group(1).split(",") if m_n else []) if n.strip()],
            "ra": pos[0] if pos else None, "dec": pos[1] if pos else None, "release": rel.group(1) if rel else None}


def build_press_targets(path: Path, workers: int = 6) -> dict[str, Any]:
    """Обход галерей: esawebb.org/images/page/N/ → страницы изображений; science.nasa.gov stma-карты → asset/webb/*.
    В файл — только изображения с координатами (графика, спектры, иллюстрации без координат отбрасываются)."""
    import requests

    sess = requests.Session()
    sess.headers.update(UA)

    def get(u: str) -> str | None:
        for _ in range(3):
            try:
                r = sess.get(u, timeout=(20, 60))
                if r.status_code == 200:
                    return r.text
                if r.status_code == 404:
                    return None
            except requests.RequestException:
                continue
        return None

    first = get("https://esawebb.org/images/")
    if first is None:
        raise SkyforgeError("esawebb.org не ответил — галерею пресс-релизов не собрать (сеть/прокси)")
    pages = max([int(n) for n in re.findall(r"/images/page/(\d+)/", first)] or [1])
    esa_links: set[str] = set()
    for n in range(1, pages + 1):
        s = first if n == 1 else get(f"https://esawebb.org/images/page/{n}/")
        for h in re.findall(r'href="/images/([^"/]+)/"', s or ""):
            if h not in ("page", "archive", "potm") and not h.startswith("?"):
                esa_links.add(f"https://esawebb.org/images/{h}/")
    nasa_links: set[str] = set()
    for k in range(1, 20):
        s = get(f"https://science.nasa.gov/wp-sitemap-posts-stma-{k}.xml")
        if s is None:
            break
        nasa_links |= set(re.findall(r"<loc>(https://science\.nasa\.gov/asset/webb/[^<]+)</loc>", s))
    if not nasa_links:
        raise SkyforgeError("science.nasa.gov не ответил — список изображений Webb (карта stma) не получен")
    console.print(f"  галереи: esawebb {len(esa_links)} страниц, NASA/STScI {len(nasa_links)} страниц — обхожу")

    def one(u: str) -> dict[str, Any] | None:
        s = get(u)
        if not s:
            return None
        return parse_esa_page(s, u) if "esawebb" in u else parse_nasa_page(s, u)

    with ThreadPoolExecutor(workers) as ex:
        items = [x for x in ex.map(one, sorted(esa_links) + sorted(nasa_links)) if x]
    data = {"built": date.today().isoformat(),
            "sources": {"esawebb": "https://esawebb.org/images/", "webbtelescope/nasa":
                        "https://science.nasa.gov/mission/webb/multimedia/images/ (sitemap stma, asset/webb)"},
            "pages_seen": {"esawebb": len(esa_links), "webbtelescope/nasa": len(nasa_links)},
            "targets": items}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return data


def load_press(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SkyforgeError(f"нет {path} — собери галерею: skyforge discover --refresh-press")
    return json.loads(path.read_text(encoding="utf-8"))


# --- превью ---


def fetch_preview(uri: str, dest: Path) -> Path | None:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    import requests

    try:
        r = requests.get(MAST_DOWNLOAD_URL, params={"uri": uri}, headers=UA, timeout=(30, 120))
        r.raise_for_status()
    except requests.RequestException as e:
        console.print(f"  [yellow]превью {uri}: {e.__class__.__name__}[/]")
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def nodata_mask(g: np.ndarray, size: int = 15) -> np.ndarray:
    """Пустоты вне поля на превью MAST — большие плоские белые области (NaN i2d рисуется белым); насыщенная звезда
    меньше окна size и после размыкания уходит."""
    from scipy import ndimage

    return ndimage.binary_opening(g >= 254, np.ones((size, size)))



# --- вывод ---


def candidates_markdown(cands: list[Candidate], cfg: DiscoverConfig, meta: dict[str, Any]) -> str:
    lines = [f"# Кандидаты skyforge discover — {meta['date']}", "",
             f"Запрос MAST: {cfg.mission} {cfg.instrument}, уровень {cfg.calib_level}, эксклюзив истёк "
             f"{meta['since']} … {meta['date']}, открыт (PUBLIC). Строк MAST: {meta['rows']}, объектов (программа, цель): "
             f"{meta['groups']}, прошли отбор: {meta['selected']}.",
             f"Отбор: фильтров ≥ {cfg.min_filters}, широкополосный — {'да' if cfg.require_wide else 'не нужен'}, "
             f"площадь ≥ {cfg.min_area_arcmin2:g} кв. угл. мин, протяжённый. Отсев: "
             + ", ".join(f"{k} — {v}" for k, v in meta["dropped"].items()) + ".",
             f"Пресс-релизы: {meta['press_file']} (собрано {meta['press_built']}, {meta['press_n']} изображений с "
             f"координатами), совпадение по имени или ближе {cfg.press_radius_arcmin:g}′.", "",
             "| # | объект | программа | PI | инструмент | фильтры | площадь, кв. угл. мин | эксклюзив снят | пресс-релиз | "
             "MAST |", "|---|---|---|---|---|---|---|---|---|---|"]
    for i, c in enumerate(cands, 1):
        pr = "нет" if not c.press else "есть: " + "; ".join(c.press[:2]) + (" …" if len(c.press) > 2 else "")
        lines.append(f"| {i} | {c.target} | {c.proposal_id} «{c.title[:70]}» | {c.pi} | {cfg.instrument} | "
                     f"{', '.join(c.filters)} ({len(c.filters)}) | {c.area_arcmin2:g} | {c.release} | {pr} | "
                     f"[MAST]({c.mast_url}) |")
    return "\n".join(lines) + "\n"


def save_candidates_json(cands: list[Candidate], path: Path) -> None:
    path.write_text(json.dumps([asdict(c) for c in cands], ensure_ascii=False, indent=1), encoding="utf-8")


# --- прогон ---


def run(root: Path, cfg: DiscoverConfig, today: date, refresh_press: bool = False) -> Path:
    """Весь шаг: запрос MAST → кандидаты → пресс-релизы → превью → ранжирование → таблица, candidates.md, лист."""
    from PIL import Image
    from rich.table import Table

    Image.MAX_IMAGE_PIXELS = None   # превью MAST больших мозаик — до 370 Мпикс (не «бомба», источник доверенный)

    def small(p: Path, mode: str, size: int):
        im = Image.open(p)
        im.draft(mode, (size, size))                       # JPEG декодируется сразу уменьшенным (DCT 1/2…1/8)
        im = im.convert(mode)
        im.thumbnail((size, size))
        return im

    from skyforge.preview import contact_sheet

    out = root / "out" / "discover" / today.isoformat()
    out.mkdir(parents=True, exist_ok=True)
    press_path = root / cfg.press_file
    press = build_press_targets(press_path) if (refresh_press or not press_path.exists()) else load_press(press_path)
    targets = press["targets"]
    console.print(f"  пресс-релизы: {len(targets)} изображений с координатами ({press_path.name}, собрано {press['built']})")

    rows = query_mast(cfg, today)
    if not rows:
        raise SkyforgeError("MAST вернул 0 наблюдений за период — проверь since и фильтры запроса")
    (out / "mast_rows.json").write_text(json.dumps(rows, ensure_ascii=False, default=str), encoding="utf-8")
    cands = group_rows(rows, cfg)
    kept, why = select(cands, cfg)
    console.print(f"  MAST: {len(rows)} продуктов, {len(cands)} объектов (программа, цель), прошли отбор: {len(kept)}")
    cache = root / "data" / "cache" / "discover"
    for c in kept:
        c.press = match_press(c, targets, cfg.press_radius_arcmin)
    # превью i2d из MAST — полноразмерные JPEG (~5 МБ, ~30 с): качаются только для короткого списка;
    # миниатюры _thumb (128 px) звезду от туманности не отличают
    ranked = cap_per_program(rank(kept), cfg.per_program)
    short = ranked[:cfg.shortlist]

    def prev(c: Candidate) -> None:
        for uri, f in ([(c.preview_uri, c.preview_filter)] if c.preview_uri else []) + list(c.previews):
            p = fetch_preview(uri, cache / Path(uri.split("/")[-1]).name)
            if p is None:
                continue
            g = np.asarray(small(p, "L", 512), dtype=np.float32)
            v = g[~nodata_mask(g, 9)]
            # quicklook MAST бывает растянут по ярким звёздам: > 90% поля — 0, туманность в 8 битах потеряна
            if v.size and (v == 0).mean() > 0.9:
                console.print(f"  [yellow]{c.target}: превью {f} пустое (растянуто по звёздам) — беру другой фильтр[/]")
                continue
            c.preview_uri, c.preview_filter = uri, f
            return

    console.print(f"  превью короткого списка: {len(short)} (не больше {cfg.per_program} на программу)")
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(prev, short))
    top = short[:cfg.top]

    tab = Table(title=f"skyforge discover — {today.isoformat()} (эксклюзив снят с {since_date(cfg, today)})")
    for col in ("#", "объект", "программа", "фильтров", "площадь′²", "снят", "пресс-релиз"):
        tab.add_column(col)
    for i, c in enumerate(top, 1):
        tab.add_row(str(i), c.target, c.proposal_id, str(len(c.filters)), f"{c.area_arcmin2:g}", c.release,
                    "нет" if not c.press else "есть")
    console.print(tab)
    meta = {"date": today.isoformat(), "since": since_date(cfg, today).isoformat(), "rows": len(rows),
            "groups": len(cands), "selected": len(kept), "dropped": why, "press_file": cfg.press_file,
            "press_built": press["built"], "press_n": len(targets)}
    md = candidates_markdown(top, cfg, meta)
    md += (f"\n## Короткий список: {len(short)} из {len(kept)} прошедших отбор (превью скачаны — лист "
           f"candidates_{len(short)}.png; не больше {cfg.per_program} на программу)\n\n") + "\n".join(
        f"{i}. {c.target} — {c.proposal_id} «{c.title[:80]}», {len(c.filters)} ф., {c.area_arcmin2:g}′², "
        f"пресс-релиз: {'есть' if c.press else 'нет'}"
        for i, c in enumerate(short, 1)) + "\n"
    md += (f"\n## Все прошедшие отбор с пресс-релизом ({sum(1 for c in ranked if c.press)} после ограничения на "
           f"программу)\n\n") + "\n".join(f"- {c.target} ({c.proposal_id}): {'; '.join(c.press[:2])}"
                                            for c in ranked if c.press) + "\n"
    (out / "candidates.md").write_text(md, encoding="utf-8")
    save_candidates_json(ranked, out / "candidates.json")
    (out / "discover.yaml").write_text(yaml.safe_dump(json.loads(cfg.model_dump_json()), allow_unicode=True,
                                                      sort_keys=False), encoding="utf-8")
    def sheet(cs: list[Candidate], path: Path, cols: int, tile: float, title: str) -> None:
        tiles, labels = [], []
        for i, c in enumerate(cs, 1):
            p = cache / Path(c.preview_uri.split("/")[-1]).name if c.preview_uri else None
            if p is not None and p.exists():
                a = np.asarray(small(p, "RGB", 900), dtype=np.float32) / 255
                void = nodata_mask(a.mean(-1) * 255, 9)
                v = a[~void]
                if v.size:   # quicklook MAST растянут почти линейно — слабое уходит в чёрное; одна asinh для всех клеток
                    lo, hi = np.percentile(v, [1, 99.5])
                    a = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1e-3), 0, None) * 10) / np.arcsinh(10)
                a = np.clip(a, 0, 1)
                a[void] = 0.07                                         # белые пустоты вне поля — тёмно-серые
                tiles.append(a[::-1])
            else:
                tiles.append(np.zeros((300, 300, 3), dtype=np.float32))
            labels.append(f"{i}. {c.target[:22]} · {c.proposal_id} · {c.preview_filter or '—'}\n"
                          f"релиз: {'есть' if c.press else 'нет'} · {len(c.filters)} ф. · {c.area_arcmin2:.0f}′²")
        contact_sheet(tiles, labels, path, cols=cols, tile_inches=tile, title=title)

    sheet(top, out / "candidates.png", 5, 4.6,
          f"skyforge discover {today.isoformat()}: {len(top)} кандидатов (превью i2d MAST, один фильтр; asinh для показа)")
    sheet(short, out / f"candidates_{len(short)}.png", 6, 3.6,
          f"skyforge discover {today.isoformat()}: короткий список {len(short)} (структура — глазами; asinh для показа)")
    console.print(f"  [green]{out / 'candidates.md'}[/], лист {out / 'candidates.png'}")
    return out
