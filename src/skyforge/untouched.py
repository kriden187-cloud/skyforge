"""Фаза 6, шаг 1: кандидаты discover, которых никто не обрабатывал.

Для короткого списка discover ищутся публичные цветные обработки данных JWST: Wikimedia Commons (API поиска файлов),
Flickr (страница поиска, без ключа API) и поиск в сети. Поиск в сети из сценария не работает: DuckDuckGo (HTML и lite)
после двух запросов отвечает HTTP 202 с антибот-страницей (2026-10-21), — поэтому те же запросы выполняет поиск агента,
а ответы импортируются в кеш (`--web-results`, формат — import_web). Читаются только страницы выдачи и страницы
находок (автор, дата) — ничего не скачивается и не запускается. Ответы кешируются в out/discover/<дата>/web/cache/:
повторный запуск по той же дате сеть не трогает.

Автоматический отбор — только первая половина: в разбор идут результаты, где в тексте (заголовок, сниппет, описание)
есть имя объекта или номер программы вместе с JWST/Webb/NIRCam; научные статьи и справочники не разбираются.
Вердикт по каждой находке — в out/discover/<дата>/web/review.yaml, руками после чтения страницы: без вердикта находка
«не разобрана», а объект в таблице — «не проверено», а не «не найдено».
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import yaml

from skyforge.discover import Candidate, preview_sheet
from skyforge.log import SkyforgeError, console

BROWSER_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/124.0 Safari/537.36"}
COMMONS_UA = {"User-Agent": "skyforge-untouched/0.1 (https://github.com/kriden187-cloud/skyforge)"}
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
FLICKR_SEARCH = "https://www.flickr.com/search/"
PAUSE = {"flickr": 1.5, "commons": 0.5}       # между запросами к одному источнику, с

# Площадки, где проверки нет — записываются в отчёт с датой (проверено 2026-10-21 из этой среды).
UNAVAILABLE = {
    "Reddit": "reddit.com — 403, old.reddit.com — заставка «Welcome to Reddit» без результатов",
    "AstroBin": "внутренний поиск — 403",
    "X/Twitter": "x.com — страница на JavaScript без выдачи (нужен вход)",
    "DuckDuckGo из сценария": "HTTP 202, антибот-страница после двух запросов; поиск в сети — поиском агента",
}

JWST_WORDS = re.compile(r"jwst|webb|nircam", re.I)
OFFICIAL = ("esawebb.org", "esa.int", "nasa.gov", "webbtelescope.org", "stsci.edu", "spacetelescope.org",
            "esahubble.org", "eso.org", "noirlab.edu")
SCIENCE = ("arxiv.org", "adsabs.harvard.edu", "iopscience.iop.org", "aanda.org", "academic.oup.com", "nature.com",
           "science.org", "researchgate.net", "semanticscholar.org", "sciencedirect.com", "springer.com", "mdpi.com",
           "ned.ipac.caltech.edu", "cds.unistra.fr", "wikipedia.org", "wikidata.org", "u-strasbg.fr", "aas.org",
           "astro-ph", "scholar.google")
PLATFORMS = {"flickr.com": "Flickr", "commons.wikimedia.org": "Wikimedia Commons", "astrobin.com": "AstroBin",
             "reddit.com": "Reddit", "x.com": "X", "twitter.com": "X", "instagram.com": "Instagram",
             "facebook.com": "Facebook", "youtube.com": "YouTube", "deviantart.com": "DeviantArt",
             "pixelfed": "Pixelfed", "mastodon": "Mastodon", "bsky.app": "Bluesky", "artstation.com": "ArtStation",
             "behance.net": "Behance", "tumblr.com": "Tumblr", "imgur.com": "Imgur", "threads.net": "Threads",
             "cloudynights.com": "Cloudy Nights", "stargazerslounge.com": "Stargazers Lounge",
             "pinterest.": "Pinterest", "hackers.pub": "hackers.pub"}

VERDICTS = {
    "эти данные": "цветная обработка JWST-данных этой программы (этого поля)",
    "другие данные JWST": "обработка JWST того же объекта по другой программе",
    "официальное": "изображение ESA/NASA/STScI",
    "нет": "не цветная обработка этих данных: другой объект, наземный снимок, статья, страница данных",
    "не открылась": "страница не читается; вердикт по сниппету невозможен",
}


# --- имена и запросы ---

def load_names(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {k: [str(n) for n in v] for k, v in (data.get("names") or {}).items()}


def default_names(target: str) -> list[str]:
    """Имя цели MAST → вид для поиска: «NGC-1326» → «NGC 1326», «HH288_offset» → «HH 288»."""
    t = re.sub(r"[_-](offset|nircam|final|jwst)\b", "", target, flags=re.I)
    t = re.sub(r"[_-]+", " ", t).strip()
    spaced = re.sub(r"^([A-Za-z]+)\s*(\d)", r"\1 \2", t)
    return list(dict.fromkeys([spaced, spaced.replace(" ", "")]))


def names_for(c: Candidate, names: dict[str, list[str]]) -> list[str]:
    return names.get(c.target) or default_names(c.target)


def build_queries(names: list[str], program: str) -> list[tuple[str, str]]:
    """Имя в 2–3 написаниях с JWST, первое — ещё с Webb и NIRCam; номер программы; Commons и Flickr — по каждому имени."""
    qs = [("web", f'"{n}" JWST') for n in names]
    qs += [("web", f'"{names[0]}" Webb'), ("web", f'"{names[0]}" NIRCam')]
    qs += [("web", f"JWST program {program}"), ("web", f"JWST {program} NIRCam")]
    qs += [("commons", f'"{n}" (JWST OR Webb OR NIRCam)') for n in names]
    qs += [("flickr", f"{n} JWST") for n in names] + [("flickr", f"{names[0]} Webb")]
    return qs


def name_pattern(name: str) -> re.Pattern:
    """«HH 288» ловит HH288, HH-288, HH_288; границы — не буква и не цифра (W3 не ловит www3)."""
    parts = re.findall(r"[a-z]+|\d+(?:\.\d+)?|[+]", name.lower())
    body = r"[\s\-_]*".join(re.escape(p) for p in parts)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])", re.I)


def is_relevant(text: str, names: list[str], program: str, context: str = "") -> bool:
    """Имя или номер программы — в самом результате (text); JWST/Webb/NIRCam — в нём же или в context (запрос,
    когда у результата нет сниппета)."""
    if not JWST_WORDS.search(text + " " + context):
        return False
    if any(name_pattern(n).search(text) for n in names):
        return True
    return bool(re.search(rf"(?<!\d){re.escape(program)}(?!\d)", text))


def site_class(url: str) -> tuple[str, str]:
    """(класс, площадка): official | science | platform | other."""
    host = urlparse(url).netloc.lower()
    for k, name in PLATFORMS.items():
        if k in host:
            return "platform", name
    if any(host == d or host.endswith("." + d) for d in OFFICIAL):
        return "official", host
    if any(d in host for d in SCIENCE):
        return "science", host
    return "other", host


def canonical(url: str) -> str:
    u = urlparse(url)
    if "flickr.com" in u.netloc:
        m = re.match(r"(/photos/[^/]+/\d+)", u.path)
        return f"https://www.flickr.com{m.group(1)}/" if m else url
    return url.split("#")[0]


# --- источники (каждый возвращает status, total, results) ---

def _strip(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse_flickr_search(page: str) -> tuple[int, list[dict[str, str]]]:
    i = page.find("modelExport:")
    if i < 0:
        raise ValueError("на странице поиска Flickr нет modelExport")
    obj, _ = json.JSONDecoder().raw_decode(page[i + len("modelExport:"):].lstrip())
    ph = obj["main"]["search-photos-lite-models"][0]["data"]["photos"]["data"]
    out = []
    for item in ph.get("_data", []):
        p = (item or {}).get("data") or {}
        if not p.get("id"):
            continue
        owner = p.get("pathAlias") or p.get("ownerNsid")
        out.append({"url": f"https://www.flickr.com/photos/{owner}/{p['id']}/", "title": p.get("title", ""),
                    "text": _strip(p.get("description", ""))[:600],
                    "author": p.get("realname") or p.get("username") or ""})
    return int(ph.get("totalItems") or len(out)), out


def _search(source: str, query: str, get: Callable) -> tuple[str, int, list[dict[str, str]]]:
    if source == "commons":
        r = get("GET", COMMONS_API, params={"action": "query", "list": "search", "srsearch": query, "srnamespace": 6,
                                            "srlimit": 50, "format": "json"}, headers=COMMONS_UA)
        q = r.json()["query"]
        res = [{"url": "https://commons.wikimedia.org/wiki/" + x["title"].replace(" ", "_"), "title": x["title"],
                "text": _strip(x.get("snippet", ""))} for x in q["search"]]
        return "ok", int(q["searchinfo"]["totalhits"]), res
    if source == "flickr":
        r = get("GET", FLICKR_SEARCH, params={"text": query}, headers=BROWSER_UA)
        if r.status_code != 200:
            return f"HTTP {r.status_code}", 0, []
        total, res = parse_flickr_search(r.text)
        return "ok", total, res
    raise ValueError(source)


def _http() -> Callable:
    import requests

    sess = requests.Session()

    def get(method: str, url: str, **kw):
        return sess.request(method, url, timeout=(20, 60), **kw)
    return get


class Cache:
    """Ответы по ключу (источник, запрос) — JSON в web/cache/; offline — только кеш."""

    def __init__(self, folder: Path, offline: bool):
        self.folder, self.offline = folder, offline
        self.last: dict[str, float] = {}
        self._get: Callable | None = None

    def _path(self, kind: str, key: str) -> Path:
        return self.folder / f"{kind}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"

    def fetch(self, kind: str, key: str, fn: Callable[[Callable], dict[str, Any]]) -> dict[str, Any]:
        p = self._path(kind, key)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        if self.offline:
            return {"status": "нет в кеше (offline)"}
        if self._get is None:
            self._get = _http()
        wait = PAUSE.get(kind, 1.0) - (time.monotonic() - self.last.get(kind, 0.0))
        if wait > 0:
            time.sleep(wait)
        rec: dict[str, Any] = {}
        for attempt in range(3):
            try:
                rec = fn(self._get)
            except Exception as e:                       # сеть, разбор — записать, не выдумывать результат
                rec = {"status": f"ошибка: {e.__class__.__name__}: {str(e)[:120]}"}
            self.last[kind] = time.monotonic()
            if not str(rec.get("status", "ok")).startswith(("заблокировано", "ошибка")):
                break
            time.sleep(20 * (attempt + 1))
        rec["fetched"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if rec.get("status") == "ok" or "status" not in rec:      # неудачи не кешируются: следующий запуск повторит
            self.folder.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        return rec

    def search(self, source: str, query: str) -> dict[str, Any]:
        if source == "web":                       # только импортированное (import_web): сценарий в поисковик не ходит
            p = self._path(source, f"{source}\n{query}")
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"status": "нет ответа поиска в сети"}

        def fn(get):
            status, total, res = _search(source, query, get)
            return {"source": source, "query": query, "status": status, "total": total, "results": res}
        return self.fetch(source, f"{source}\n{query}", fn)


def import_web(path: Path, cache: Cache) -> int:
    """Ответы поиска агента → кеш: JSON-список {query, status, results: [{url, title, text}], searched}; query — ровно
    как в build_queries. Пустой results при status ok — «ничего не найдено», это тоже ответ."""
    n = 0
    for rec in json.loads(path.read_text(encoding="utf-8")):
        res = [{"url": r["url"], "title": r.get("title", ""), "text": r.get("text", "")} for r in rec.get("results", [])]
        out = {"source": "web", "query": rec["query"], "status": rec.get("status", "ok"), "total": len(res),
               "results": res, "fetched": rec.get("searched", ""), "via": "поиск агента"}
        if out["status"] != "ok":
            continue
        p = cache._path("web", f"web\n{rec['query']}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    return n


def enrich(hit: dict[str, Any], cache: Cache) -> None:
    """Автор и дата первой загрузки: Commons — imageinfo, Flickr — datePosted со страницы фото."""
    if hit["platform"] == "Wikimedia Commons":
        title = hit["url"].split("/wiki/", 1)[1].replace("_", " ")

        def fn(get):
            r = get("GET", COMMONS_API, params={"action": "query", "titles": title, "prop": "imageinfo",
                                                "iiprop": "user|timestamp|extmetadata", "iilimit": 50,
                                                "format": "json"}, headers=COMMONS_UA).json()
            page = next(iter(r["query"]["pages"].values()))
            infos = page.get("imageinfo") or []
            if not infos:
                return {"status": "ok"}
            first, meta = infos[-1], infos[0].get("extmetadata", {})
            return {"status": "ok", "user": first.get("user"), "date": first.get("timestamp", "")[:10],
                    "artist": _strip((meta.get("Artist") or {}).get("value", ""))[:120]}
        info = cache.fetch("commons", "info\n" + title, fn)
        hit.setdefault("author", info.get("artist") or info.get("user") or "")
        hit["date"] = info.get("date", "")
    elif hit["platform"] == "Flickr":
        def fn(get):
            r = get("GET", hit["url"], headers=BROWSER_UA)
            m = re.search(r'"datePosted":"(\d+)"', r.text)
            return {"status": "ok" if r.status_code == 200 else f"HTTP {r.status_code}",
                    "date": datetime.fromtimestamp(int(m.group(1)), timezone.utc).date().isoformat() if m else ""}
        hit["date"] = cache.fetch("flickr", "page\n" + hit["url"], fn).get("date", "")


# --- сверка ---

@dataclass
class Check:
    cand: Candidate
    names: list[str]
    queries: list[dict[str, Any]] = field(default_factory=list)   # source, query, status, total, n
    hits: list[dict[str, Any]] = field(default_factory=list)      # к разбору: url, title, text, sources, class, platform


def sweep(c: Candidate, names: list[str], cache: Cache) -> Check:
    ck = Check(c, names)
    seen: dict[str, dict[str, Any]] = {}
    for source, q in build_queries(names, c.proposal_id):
        rec = cache.search(source, q)
        ck.queries.append({"source": source, "query": q, "status": rec.get("status"), "total": rec.get("total", 0),
                           "n": len(rec.get("results", []))})
        for r in rec.get("results", []):
            url = canonical(r["url"])
            text = " ".join([r.get("title", ""), r.get("text", ""), url])
            # поиск агента отдаёт только заголовок и ссылку: JWST/Webb/NIRCam берётся из запроса, имя — нет
            ctx = q if source == "web" and not r.get("text") else ""
            cls, platform = site_class(url)
            if cls == "science" or not is_relevant(text, names, c.proposal_id, ctx):
                continue
            h = seen.setdefault(url, {"url": url, "title": r.get("title", ""), "text": r.get("text", "")[:300],
                                      "class": cls, "platform": platform, "sources": []})
            if r.get("author"):
                h["author"] = r["author"]
            if source not in h["sources"]:
                h["sources"].append(source)
    ck.hits = list(seen.values())
    for h in ck.hits:
        enrich(h, cache)
    return ck


def load_review(path: Path) -> dict[str, Any]:
    """hits: url → вердикт по находкам шаблонного поиска; extra: находки дополнительного поиска (target, url, …);
    extra_searches: журнал дополнительных поисков (target, where, query, status)."""
    if not path.exists():
        return {"structure": {}, "hits": {}, "extra": [], "extra_searches": []}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    hits = data.get("hits") or {}
    extra = data.get("extra") or []
    bad = {u: v.get("verdict") for u, v in hits.items() if v.get("verdict") not in VERDICTS}
    bad.update({e.get("url"): e.get("verdict") for e in extra if e.get("verdict") not in VERDICTS})
    if bad:
        raise SkyforgeError(f"{path}: неизвестные вердикты {bad}; допустимы: {', '.join(VERDICTS)}")
    return {"structure": data.get("structure") or {}, "hits": hits, "extra": extra,
            "extra_searches": data.get("extra_searches") or []}


def summarize(ck: Check, review: dict[str, Any]) -> dict[str, Any]:
    """Итог по объекту: обработки этих данных, прочие находки, неразобранные, сбои запросов."""
    rv = review["hits"]
    mine = [dict(h, **rv[h["url"]]) for h in ck.hits if rv.get(h["url"], {}).get("verdict") == "эти данные"]
    other = [dict(h, **rv[h["url"]]) for h in ck.hits
             if rv.get(h["url"], {}).get("verdict") in ("другие данные JWST", "официальное")]
    extra = [e for e in review.get("extra", []) if e.get("target") == ck.cand.target]
    mine += [e for e in extra if e["verdict"] == "эти данные"]
    other += [e for e in extra if e["verdict"] in ("другие данные JWST", "официальное")]
    pending = [h for h in ck.hits if h["url"] not in rv]
    failed = [q for q in ck.queries if q["status"] != "ok"]
    if mine:
        state = "найдено"
    elif pending or failed:
        state = "не проверено"
    else:
        state = "не найдено"
    return {"state": state, "mine": sorted(mine, key=lambda h: h.get("date") or "9999"), "other": other,
            "pending": pending, "failed": failed}


def order(checks: list[Check], sums: dict[str, dict[str, Any]]) -> list[Check]:
    """Без обработок → не проверено → с обработками; внутри — свежее открытие выше, затем больше фильтров."""
    rank = {"не найдено": 0, "не проверено": 1, "найдено": 2}
    return sorted(checks, key=lambda ck: (rank[sums[ck.cand.slug]["state"]],
                                          -date.fromisoformat(ck.cand.release).toordinal(), -len(ck.cand.filters)))


def _find(h: dict[str, Any]) -> str:
    who = h.get("author") or "автор не указан"
    return f"[{who}, {h.get('platform', '?')}, {h.get('date') or 'дата ?'}]({h['url']})"


def table_md(checks: list[Check], sums: dict[str, dict[str, Any]], review: dict[str, Any], meta: dict[str, Any]) -> str:
    st = review["structure"]
    lines = [f"# Кандидаты без любительских обработок — {meta['date']}", "",
             f"Короткий список `skyforge discover` {meta['date']}: {len(checks)} объектов. Сверка {meta['date']}: "
             "поиск в сети (поиск агента: DuckDuckGo из сценария блокирует), Wikimedia Commons (API, файлы), "
             "Flickr (страница поиска, без ключа). "
             "Только чтение страниц выдачи и страниц находок. Наземные снимки, научные статьи и страницы данных "
             "не считаются. Вердикт по каждой находке — после чтения страницы (`web/review.yaml`).", "",
             "Не проверено (недоступно из этой среды, " + meta["date"] + "): " +
             "; ".join(f"{k} — {v}" for k, v in UNAVAILABLE.items()) + ".", "",
             f"Запросов: {meta['queries']} (сбоев: {meta['failed']}); находок к разбору: {meta['hits']}, "
             f"разобрано: {meta['reviewed']}.", "",
             "| # | объект | программа | фильтры | площадь, ′² | открыт | пресс-релиз | любительские обработки | структура (превью, глазами) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for i, ck in enumerate(checks, 1):
        c, s = ck.cand, sums[ck.cand.slug]
        # официальные изображения из сверки — по другим данным (иначе это «эти данные» PI): в «Прочие находки»
        official = [h for h in s["other"] if h.get("verdict") == "официальное"]
        press = "есть" if c.press else ("нет (официальное по другим данным — ниже)" if official else "нет")
        if s["state"] == "найдено":
            found = f"{len(s['mine'])}: " + "; ".join(_find(h) for h in s["mine"])
        elif s["state"] == "не проверено":
            why = []
            if s["pending"]:
                why.append(f"не разобрано находок: {len(s['pending'])}")
            if s["failed"]:
                why.append(f"сбоев запросов: {len(s['failed'])}")
            found = "не проверено (" + ", ".join(why) + ")"
        else:
            found = "не найдено"
        lines.append(f"| {i} | {c.target} | {c.proposal_id} | {', '.join(c.filters)} ({len(c.filters)}) | "
                     f"{c.area_arcmin2:g} | {c.release} | {press} | {found} | {st.get(c.target, '—')} |")
    lines += ["", "## Прочие находки по объектам", "",
              "Другие данные JWST того же объекта и официальные изображения — не обработки этих данных, но говорят, "
              "что объект «на слуху».", ""]
    for ck in checks:
        s = sums[ck.cand.slug]
        if s["other"]:
            lines.append(f"- **{ck.cand.target}** ({ck.cand.proposal_id}): " + "; ".join(
                f"{h['verdict']}: {_find(h)}" + (f" — {h['evidence'][:260]}" if h.get("evidence") else "")
                for h in s["other"]))
    lines += ["", "## Имена для поиска и запросы", ""]
    for ck in checks:
        n = sum(q["n"] for q in ck.queries)
        bad = [f"{q['source']} «{q['query']}»: {q['status']}" for q in ck.queries if q["status"] != "ok"]
        # агенты добора пишут и совмещённые метки («W3-…-3A / W3-…-2») — по вхождению имени цели
        more = sum(1 for e in review.get("extra_searches", []) if ck.cand.target in str(e.get("target", "")))
        lines.append(f"- {ck.cand.target}: {', '.join(ck.names)}; запросов {len(ck.queries)}, результатов {n}, "
                     f"к разбору {len(ck.hits)}" + (f", дополнительных поисков {more}" if more else "") +
                     (f"; сбои: {'; '.join(bad)}" if bad else ""))
    return "\n".join(lines) + "\n"


def todo_md(checks: list[Check], review: dict[str, Any]) -> str:
    rv = review["hits"]
    lines = ["# Находки к разбору", "", "Вердикты — в `review.yaml` (hits: url → verdict, author, platform, date, note). "
             "Допустимые вердикты: " + "; ".join(f"«{k}» — {v}" for k, v in VERDICTS.items()) + ".", ""]
    for ck in checks:
        todo = [h for h in ck.hits if h["url"] not in rv]
        if not todo:
            continue
        lines.append(f"## {ck.cand.target} ({ck.cand.proposal_id}; {', '.join(ck.names)})")
        for h in todo:
            lines.append(f"- [{h['class']} · {h['platform']}] {h['url']}\n  «{h['title'][:120]}» — "
                         f"{h.get('author', '')} {h.get('date', '')} — {h['text'][:200]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def run(root: Path, day: str, shortlist: int | None = None, offline: bool = False,
        web_results: Path | None = None) -> Path:
    out = root / "out" / "discover" / day
    if not (out / "candidates.json").exists():
        raise SkyforgeError(f"нет {out / 'candidates.json'} — сначала skyforge discover")
    dcfg = yaml.safe_load((out / "discover.yaml").read_text(encoding="utf-8"))
    n = shortlist or int(dcfg.get("shortlist", 30))
    cands = [Candidate(**{k: v for k, v in c.items()})
             for c in json.loads((out / "candidates.json").read_text(encoding="utf-8"))][:n]
    names = load_names(root / "configs" / "discover_names.yaml")
    web = out / "web"
    cache = Cache(web / "cache", offline)
    if web_results is not None:
        console.print(f"  поиск в сети: импортировано ответов {import_web(web_results, cache)} из {web_results}")
    checks = []
    for i, c in enumerate(cands, 1):
        ck = sweep(c, names_for(c, names), cache)
        console.print(f"  {i}/{len(cands)} {c.target}: запросов {len(ck.queries)}, к разбору {len(ck.hits)}"
                      + (f", сбоев {sum(q['status'] != 'ok' for q in ck.queries)}" if any(
                          q["status"] != "ok" for q in ck.queries) else ""))
        checks.append(ck)
    web.mkdir(parents=True, exist_ok=True)
    (web / "hits.json").write_text(json.dumps([{"target": ck.cand.target, "program": ck.cand.proposal_id,
                                                "names": ck.names, "queries": ck.queries, "hits": ck.hits}
                                               for ck in checks], ensure_ascii=False, indent=1), encoding="utf-8")
    review = load_review(web / "review.yaml")
    (web / "todo.md").write_text(todo_md(checks, review), encoding="utf-8")
    sums = {ck.cand.slug: summarize(ck, review) for ck in checks}
    ordered = order(checks, sums)
    all_hits = {h["url"] for ck in checks for h in ck.hits}
    meta = {"date": day, "queries": sum(len(ck.queries) for ck in checks),
            "failed": sum(q["status"] != "ok" for ck in checks for q in ck.queries),
            "hits": len(all_hits), "reviewed": len(all_hits & set(review["hits"]))}
    (out / "candidates_untouched.md").write_text(table_md(ordered, sums, review, meta), encoding="utf-8")

    st = review["structure"]
    # лист: туманности первыми (цель фазы — туманность), затем остальные в порядке таблицы; номер — строка таблицы
    clean = [ck for ck in ordered if sums[ck.cand.slug]["state"] == "не найдено"]
    clean = sorted(clean, key=lambda ck: "туманн" not in str(st.get(ck.cand.target, "")))[:10]
    items = [(ck.cand, f"{ordered.index(ck) + 1}. {ck.cand.target[:22]} · {ck.cand.proposal_id} · "
                       f"{len(ck.cand.filters)} ф.\nоткрыт {ck.cand.release} · {str(st.get(ck.cand.target, '—'))[:24]}")
             for ck in clean]
    if items:
        preview_sheet(root / "data" / "cache" / "discover", items, out / "candidates_untouched.png", 5, 4.2,
                      f"skyforge untouched {day}: обработок в сети не найдено — {len(items)} (туманности первыми; "
                      f"номер — строка candidates_untouched.md; превью i2d MAST, asinh)")
    states = {k: sum(s["state"] == k for s in sums.values()) for k in ("не найдено", "не проверено", "найдено")}
    console.print(f"  не найдено: {states['не найдено']}, не проверено: {states['не проверено']}, "
                  f"найдено: {states['найдено']}; находок к разбору {meta['hits'] - meta['reviewed']} — {web / 'todo.md'}")
    return out


# --- разведка лучших (шаг 5): фильтры, объём, наведения, превью всех фильтров, кропы 100% ---

EXPOSURE_RE = re.compile(r"jw(\d{5})(\d{3})(\d{3})_(\d{5})_(\d{5})_([a-z0-9]+)_cal\.fits")


def exposure_layout(cal_files: list[str]) -> dict[str, Any]:
    """Имена _cal.fits (jwPPPPPOOOVVV_GGSAA_EEEEE_детектор) → визиты (наблюдение, визит), плитки мозаики (визит,
    активность GGSAA), дизеринг (экспозиций на плитку, наибольшее), детекторы."""
    parts = [m.groups() for f in cal_files if (m := EXPOSURE_RE.match(f))]
    tiles: dict[tuple[str, str, str], set[str]] = {}
    for _, o, v, g, e, _ in parts:
        tiles.setdefault((o, v, g), set()).add(e)
    return {"exposures": sum(len(s) for s in tiles.values()), "visits": len({t[:2] for t in tiles}),
            "tiles": len(tiles), "dithers": max((len(s) for s in tiles.values()), default=0),
            "detectors": sorted({d for *_, d in parts}), "files": len(parts)}


def recon(root: Path, day: str, targets: list[str]) -> Path:
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    from skyforge.discover import _small, fetch_preview, nodata_mask
    from skyforge.fetch import product_lists
    from skyforge.filters import wavelength_um
    from skyforge.preview import contact_sheet

    Image.MAX_IMAGE_PIXELS = None
    out = root / "out" / "discover" / day
    rows = json.loads((out / "mast_rows.json").read_text(encoding="utf-8"))
    cands = {c["target"]: Candidate(**c) for c in json.loads((out / "candidates.json").read_text(encoding="utf-8"))}
    cache = root / "data" / "cache" / "discover"
    rdir = out / "recon"
    rdir.mkdir(parents=True, exist_ok=True)
    md = [f"# Разведка кандидатов — {day}", "",
          "Источник — MAST (список продуктов наблюдений уровня 3 и их экспозиций уровня 2) и quicklook-превью i2d MAST. "
          "Ничего, кроме превью, не скачано. Визиты, плитки и дизеринг — по именам _cal.fits "
          "(jwPPPPPOOOVVV_GGSAA_EEEEE): визит — (наблюдение, визит), плитка мозаики — активность GGSAA в визите, "
          "дизеринг — экспозиций на плитку; кропы — из 8-битного JPEG quicklook (ступенчатость — от JPEG, не данных); "
          "яркие звёзды и полосы 1/f — глазами по листам.", ""]
    for t in targets:
        c = cands[t]
        mine = [r for r in rows if r["target_name"] == t and str(r["proposal_id"]) == c.proposal_id]
        console.print(f"  разведка {t}: {len(mine)} фильтров")
        per, total3, total2 = [], 0, 0
        for r in sorted(mine, key=lambda r: wavelength_um(r["filters"].split(";")[-1], "jwst", "nircam")):
            f = [x for x in r["filters"].split(";") if x.upper() != "CLEAR"][-1].upper()
            prods = product_lists([str(r["obsid"])])
            i2d = [p for p in prods if str(p["productFilename"]).startswith(r["obs_id"]) and
                   str(p["productFilename"]).endswith("_i2d.fits")]
            cal = [p for p in prods if str(p["productFilename"]).endswith("_cal.fits")]
            s3 = sum(int(p["size"] or 0) for p in i2d)
            s2 = sum(int(p["size"] or 0) for p in cal)
            total3, total2 = total3 + s3, total2 + s2
            lay = exposure_layout([str(p["productFilename"]) for p in cal])
            per.append({"filter": f, "um": wavelength_um(f, "jwst", "nircam"), "exptime": float(r["t_exptime"] or 0),
                        "i2d_gb": s3 / 1e9, "cal_gb": s2 / 1e9, "jpeg": r.get("jpegURL"), **lay})
        md += [f"## {t} — программа {c.proposal_id}, {c.pi}", "",
               f"«{c.title}». Открыт {c.release}; поле {c.area_arcmin2:g}′²; i2d уровня 3 всего {total3 / 1e9:.1f} ГБ, "
               f"экспозиции _cal {total2 / 1e9:.1f} ГБ.", "",
               "| фильтр | λ, мкм | выдержка, с | i2d, ГБ | визитов | плиток | дизеринг | экспозиций | детекторы |",
               "|---|---|---|---|---|---|---|---|---|"]
        md += [f"| {p['filter']} | {p['um']:.3f} | {p['exptime']:.0f} | {p['i2d_gb']:.2f} | {p['visits']} | {p['tiles']} | "
               f"{p['dithers']} | {p['exposures']} | {', '.join(p['detectors'])} |" for p in per]
        md += ["", f"Листы: `recon/{c.slug}_filters.png` (все фильтры, asinh), `recon/{c.slug}_crops.png` "
                   "(кропы 1000 px в 100% из quicklook).", ""]
        tiles, labels, crops, clabels = [], [], [], []
        for p in per:
            if not p["jpeg"]:
                continue
            img = fetch_preview(p["jpeg"], cache / Path(p["jpeg"].split("/")[-1]).name)
            if img is None:
                continue
            item = Candidate(**{**c.__dict__, "preview_uri": p["jpeg"], "preview_filter": p["filter"]})
            tiles.append(item)
            labels.append(f"{p['filter']} · {p['um']:.2f} мкм · {p['exptime']:.0f} с")
            full = np.asarray(Image.open(img).convert("L"), dtype=np.float32)
            small = np.asarray(_small(img, "L", 600), dtype=np.float32)
            valid = ~nodata_mask(small, 9)
            # окно 1000 px целиком в данных, ближайшее к центру поля: центр масс у мозаики модулей A и B — в щели
            h = 500
            k = max(int(round(2 * h * small.shape[0] / full.shape[0])), 1)
            frac = ndimage.uniform_filter(valid.astype(np.float32), size=k, mode="constant")
            ok = frac >= min(0.98, frac.max())
            ys, xs = np.nonzero(valid)
            gy, gx = np.nonzero(ok)
            j = int(np.argmin((gy - ys.mean()) ** 2 + (gx - xs.mean()) ** 2))
            cy, cx = int(gy[j] / small.shape[0] * full.shape[0]), int(gx[j] / small.shape[1] * full.shape[1])
            a = full[max(cy - h, 0):cy + h, max(cx - h, 0):cx + h] / 255
            v = a[a < 0.99]
            if v.size:
                lo, hi = np.percentile(v, [1, 99.5])
                a = np.arcsinh(np.clip((a - lo) / max(hi - lo, 1e-3), 0, None) * 10) / np.arcsinh(10)
            crops.append(np.clip(a, 0, 1)[::-1])
            clabels.append(f"{p['filter']} · центр поля, 100%")
        preview_sheet(cache, list(zip(tiles, labels)), rdir / f"{c.slug}_filters.png", 4, 3.6,
                      f"{t} ({c.proposal_id}): все фильтры, quicklook i2d MAST, asinh")
        if crops:
            contact_sheet(crops, clabels, rdir / f"{c.slug}_crops.png", cols=4, tile_inches=3.6,
                          title=f"{t}: кропы 1000 px в 100% (полосы 1/f, звёзды)")
    (rdir / "recon.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rdir / "recon.md"
