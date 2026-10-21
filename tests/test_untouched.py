"""Сверка кандидатов с обработками в сети (фаза 6, шаг 1): разбор ответов, отбор находок, порядок — без сети."""

import json

import pytest

from skyforge.discover import Candidate
from skyforge.log import SkyforgeError
from skyforge.untouched import (Cache, Check, build_queries, canonical, default_names, import_web, is_relevant,
                                load_review, name_pattern, order, parse_flickr_search, site_class, summarize)


def cand(target, release="2026-01-01", filters=("F200W", "F444W", "F356W")):
    return Candidate(target=target, proposal_id="4547", title="t", pi="p", classification="", ra=0.0, dec=0.0,
                     filters=list(filters), area_arcmin2=10.0, release=release, obs_ids=[], preview_uri=None,
                     preview_filter=None)


def test_default_names():
    assert default_names("NGC-1326") == ["NGC 1326", "NGC1326"]
    assert default_names("HH288_offset") == ["HH 288", "HH288"]
    assert default_names("LHA-120-N-159-NIRCam")[0] == "LHA 120 N 159"


def test_name_pattern_and_relevance():
    p = name_pattern("HH 288")
    assert p.search("JWST view of HH288") and p.search("hh-288 outflow") and not p.search("HH 2880")
    assert not name_pattern("W3").search("see www3.example.org")
    assert is_relevant("Sh2-305 seen by JWST NIRCam", ["Sh 2-305"], "4547")
    assert not is_relevant("Sh2-305 from my backyard", ["Sh 2-305"], "4547")          # нет JWST/Webb/NIRCam
    assert is_relevant("JWST program 4547 data", ["Sharpless 305"], "4547")
    assert not is_relevant("JWST program 45470", ["Sharpless 305"], "4547")


def test_site_class_and_canonical():
    assert site_class("https://www.flickr.com/photos/x/1/") == ("platform", "Flickr")
    assert site_class("https://esawebb.org/images/x/")[0] == "official"
    assert site_class("https://arxiv.org/abs/1")[0] == "science"
    assert site_class("https://someblog.net/post")[0] == "other"
    assert canonical("https://www.flickr.com/photos/markmccaughrean/55209728128/in/photostream/?x=1") == \
        "https://www.flickr.com/photos/markmccaughrean/55209728128/"


def test_build_queries_cover_names_and_program():
    qs = build_queries(["Sh2-305", "Sharpless 305"], "4547")
    web = [q for s, q in qs if s == "web"]
    assert '"Sh2-305" JWST' in web and '"Sh2-305" Webb' in web and '"Sh2-305" NIRCam' in web
    assert "JWST program 4547" in web
    assert {s for s, _ in qs} == {"web", "commons", "flickr"}


def test_parse_flickr_search():
    model = {"main": {"search-photos-lite-models": [{"data": {"photos": {"data": {"totalItems": 7, "_data": [
        {"data": {"id": "55200778378", "pathAlias": "andrealuck", "realname": "Andrea Luck", "title": "Sh2-305 JWST",
                  "description": "Target: S305_offset <a href='x'>link</a>"}},
        None]}}}}]}}
    total, res = parse_flickr_search("<script>modelExport: " + json.dumps(model) + ",\n</script>")
    assert total == 7 and len(res) == 1
    assert res[0]["url"] == "https://www.flickr.com/photos/andrealuck/55200778378/"
    assert res[0]["author"] == "Andrea Luck" and "<a" not in res[0]["text"]


def test_import_web_feeds_cache(tmp_path):
    src = tmp_path / "r.json"
    src.write_text(json.dumps([{"query": '"NGC 248" JWST', "status": "ok", "searched": "2026-10-21",
                                "results": [{"url": "https://a.org/x", "title": "t", "text": "s"}]},
                               {"query": "broken", "status": "ошибка: таймаут", "results": []}]), encoding="utf-8")
    cache = Cache(tmp_path / "cache", offline=True)
    assert import_web(src, cache) == 1
    assert cache.search("web", '"NGC 248" JWST')["results"][0]["url"] == "https://a.org/x"
    assert cache.search("web", "broken")["status"] != "ok"           # не импортированное — не «ничего не найдено»


def test_summarize_states_and_order(tmp_path):
    ok = [{"source": "web", "query": "q", "status": "ok", "total": 1, "n": 1}]
    hit = {"url": "https://f/1", "title": "", "text": "", "class": "platform", "platform": "Flickr", "sources": ["flickr"]}
    a = Check(cand("A", "2026-03-01"), ["A"], ok, [dict(hit)])                       # обработка найдена
    b = Check(cand("B", "2026-02-01"), ["B"], ok, [])                                # ничего
    c = Check(cand("C", "2026-05-01", ("F200W",) * 5), ["C"], ok, [])                # ничего, свежее
    d = Check(cand("D", "2026-06-01"), ["D"], ok, [dict(hit, url="https://f/2")])    # не разобрано
    e = Check(cand("E", "2026-07-01"), ["E"], [dict(ok[0], status="нет ответа поиска в сети")], [])
    review = {"structure": {}, "hits": {"https://f/1": {"verdict": "эти данные", "author": "X", "date": "2026-04-03"}}}
    sums = {ck.cand.slug: summarize(ck, review) for ck in (a, b, c, d, e)}
    assert [sums[x.cand.slug]["state"] for x in (a, b, c, d, e)] == \
        ["найдено", "не найдено", "не найдено", "не проверено", "не проверено"]
    assert [ck.cand.target for ck in order([a, b, c, d, e], sums)] == ["C", "B", "E", "D", "A"]


def test_load_review_rejects_unknown_verdict(tmp_path):
    p = tmp_path / "review.yaml"
    p.write_text("hits:\n  https://x: {verdict: может быть}\n", encoding="utf-8")
    with pytest.raises(SkyforgeError):
        load_review(p)


def test_exposure_layout():
    from skyforge.untouched import exposure_layout

    files = ["jw05437003001_11201_00001_nrcalong_cal.fits", "jw05437003001_11201_00001_nrcblong_cal.fits",
             "jw05437003001_11201_00002_nrcalong_cal.fits", "jw05437003002_11201_00001_nrcalong_cal.fits",
             "jw05437-o003_t003_nircam_clear-f360m_i2d.fits"]
    lay = exposure_layout(files)
    assert (lay["visits"], lay["tiles"], lay["dithers"], lay["exposures"], lay["files"]) == (2, 2, 2, 3, 4)
    assert lay["detectors"] == ["nrcalong", "nrcblong"]


def test_relevance_context_gives_only_jwst():
    assert is_relevant("Westerhout 3 processed", ["Westerhout 3"], "5437", context='"Westerhout 3" JWST')
    assert not is_relevant("Lawn mower parts", ["HH 900"], "5408", context='"HH 900" Webb')   # имя только в запросе
    assert not is_relevant("Westerhout 3 from my backyard", ["Westerhout 3"], "5437")
