"""Фаза 4: skyforge discover без сети — разбор ответа MAST по сохранённому образцу, группировка, площадь поля,
отбор протяжённых целей, сверка с пресс-релизами (имя и координаты), разбор страниц галерей."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from skyforge.discover import (Candidate, DiscoverConfig, catalog_ids, group_rows, match_press, parse_esa_page,
                               parse_nasa_page, polygons, rank, select, since_date, union_area_arcmin2)

SAMPLE = Path(__file__).parent / "data" / "mast_discover_sample.json"   # 67 строк MAST (8 пар программа+цель), 2026-10-20


@pytest.fixture(scope="module")
def groups():
    rows = json.loads(SAMPLE.read_text(encoding="utf-8"))
    return {c.target: c for c in group_rows(rows, DiscoverConfig())}


def test_group_rows_one_line_per_target_and_program(groups):
    assert len(groups) == 8
    hh = groups["HH211NIRCAM"]
    assert hh.proposal_id == "4531"
    # пара «широкий блокирующий; узкий» → узкий фильтр: F444W;F466N → F466N
    assert hh.filters == ["F200W", "F210M", "F212N", "F444W", "F466N", "F470N"]
    assert 8 < hh.area_arcmin2 < 16                  # одно наведение NIRCam, два модуля ≈ 9,7 кв. угл. мин и мозаика
    assert hh.release == "2026-01-25"
    assert hh.preview_filter in ("F200W", "F444W") and hh.preview_uri.endswith("_i2d.jpg")
    assert len(groups["LMC-CAL-FIELD"].filters) == 16


def test_select_keeps_extended_and_drops_fields_points_and_no_wide(groups):
    cfg = DiscoverConfig()
    kept, why = select(list(groups.values()), cfg)
    assert {c.target for c in kept} == {"HH211NIRCAM", "SMC-SW-Bar-3"}
    # HH 211 в MAST — «Unidentified; Blank field»: тип снимается каталожным именем HH211
    assert groups["HH211NIRCAM"].kind == "протяжённый"
    assert groups["POINTING2"].kind.startswith("исключён")            # пустое поле обзора
    assert groups["COS-87259"].kind.startswith("исключён")            # далёкая галактика
    assert groups["54P_deVico_Slow"].kind.startswith("исключён")      # комета
    assert groups["LMC-CAL-FIELD"].kind.startswith("исключён")        # калибровка
    assert why["нет широкополосного"] == 1                            # W51A: только M и N
    assert sum(why.values()) + len(kept) == len(groups)


def test_union_area_and_polygons():
    sq = "POLYGON 10.0 0.0 10.0333333 0.0 10.0333333 0.0333333 10.0 0.0333333"      # 2′ × 2′ у экватора
    assert union_area_arcmin2(polygons(sq)) == pytest.approx(4.0, rel=0.05)
    assert union_area_arcmin2(polygons(sq + " " + sq)) == pytest.approx(4.0, rel=0.05)            # объединение
    far = "POLYGON 11.0 0.0 11.0333333 0.0 11.0333333 0.0333333 11.0 0.0333333"
    assert union_area_arcmin2(polygons(sq + " " + far), step_arcsec=6) == pytest.approx(8.0, rel=0.08)
    assert len(polygons(sq + " " + far)) == 2 and polygons("CIRCLE 1 2 3") == []


def test_catalog_ids():
    assert catalog_ids("HH211NIRCAM") == catalog_ids("HH 211") == {"HH211"}
    assert catalog_ids("Messier 16") == {"M16"} and catalog_ids("NGC-346") == {"NGC346"}
    assert catalog_ids("W51A-NIRCAM") == {"W51A"}
    assert catalog_ids("Carina Nebula, NGC 3324") == {"NGC3324"}
    assert catalog_ids("F444W") == set() and catalog_ids("SMC-SW-Bar-3") == set()


def _cand(target, ra, dec, n=4, area=10.0):
    return Candidate(target=target, proposal_id="1", title="", pi="", classification="", ra=ra, dec=dec,
                     filters=[f"F{100 + i}W" for i in range(n)], area_arcmin2=area, release="2026-01-01", obs_ids=[],
                     preview_uri=None, preview_filter=None)


def test_match_press_by_coords_and_name():
    press = [{"source": "esawebb", "title": "near", "names": ["Something"], "ra": 10.0, "dec": 20.0 + 2.5 / 60},
             {"source": "nasa", "title": "HH 211 (NIRCam)", "names": ["HH 211"], "ra": 55.0, "dec": 32.0},
             {"source": "nasa", "title": "far", "names": ["NGC 1"], "ra": 10.0, "dec": 20.0 + 4 / 60}]
    c = _cand("HH211NIRCAM", 10.0, 20.0)
    hits = match_press(c, press, 3.0)
    assert len(hits) == 2 and "2.5′" in hits[0] and "имя HH211" in hits[1]
    assert match_press(_cand("X-1", 100.0, -30.0), press, 3.0) == []


def test_rank_owner_criteria():
    a = _cand("a", 0, 0, n=6)
    b = _cand("b", 0, 0, n=3)
    c = _cand("c", 0, 0, n=9)
    c.press = ["esawebb: c"]
    d = _cand("d", 0, 0, n=3, area=30.0)
    assert [x.target for x in rank([a, b, c, d])] == ["a", "d", "b", "c"]


def test_since_date():
    assert since_date(DiscoverConfig(), date(2026, 10, 20)) == date(2025, 10, 20)
    assert since_date(DiscoverConfig(since_months=3), date(2026, 2, 28)) == date(2025, 11, 28)
    assert since_date(DiscoverConfig(since=date(2026, 1, 1)), date(2026, 10, 20)) == date(2026, 1, 1)


ESA_HTML = """<title>
    Carina Nebula Jets (NIRCam Narrowband Filters) | ESA/Webb
</title><table><tr><th scope="row">
    Name:
</th><td colspan="2"><a href="#">Carina Nebula</a>, <a href="#">NGC 3324</a></td></tr><tr><th scope="row">
    Distance:
</th><td>7600 light years</td></tr><tr><th>Release date:</th><td>
    15 December 2022, 16:00
</td></tr></table><h3 id="coordinates-heading">Coordinates</h3><table><tr><th scope="row" class="title">Position (RA):</th>
<td>10 36 51.13</td></tr><tr><th scope="row" class="title">Position (Dec):</th><td>-58&deg 37' 12.79"</td></tr><tr>
<th scope="row" class="title">Field of view:</th><td>7.29 x 4.23 arcminutes</td></tr></table>"""

NASA_HTML = """<title>NGC 7129 (NIRCam Image) - NASA Science</title><div class="stma-tooltip cursor-pointer">R.A. Position
<div class="stma-tooltiptext"><span class="stma-tooltip-title">R.A. Position</span><span class="stma-tooltip-body">Right
ascension – analogous to longitude – is one component of an object's position.</span></div></div></div><div>21:43:05.02</div>
<div class="stma-tooltip cursor-pointer">Dec. Position<div class="stma-tooltiptext"><span>Dec. Position</span><span>Declination
– analogous to latitude – is one component of an object's position.</span></div></div></div><div>+66:06:45.83</div>
<span class="stma-label">Object Name</span><div class="stma-tooltiptext"><span>Object Name</span><span>A name or catalog
number that astronomers use to identify an astronomical object.</span></div></div></div><div class="stma-title">NGC 7129</div>
</li><li><span class="stma-label">Object Description</span><span>Release Date</span></div><div>October 6, 2026</div>"""


def test_parse_gallery_pages():
    e = parse_esa_page(ESA_HTML, "https://esawebb.org/images/x/")
    assert e["names"] == ["Carina Nebula", "NGC 3324"] and e["title"].startswith("Carina Nebula Jets")
    assert e["ra"] == pytest.approx(15 * (10 + 36 / 60 + 51.13 / 3600)) and e["dec"] == pytest.approx(-(58 + 37 / 60 + 12.79 / 3600))
    assert e["release"] == "15 December 2022"
    n = parse_nasa_page(NASA_HTML, "https://science.nasa.gov/asset/webb/x/")
    assert n["names"] == ["NGC 7129"] and n["title"] == "NGC 7129 (NIRCam Image)"
    assert n["ra"] == pytest.approx(15 * (21 + 43 / 60 + 5.02 / 3600)) and n["dec"] == pytest.approx(66 + 6 / 60 + 45.83 / 3600)
    assert parse_esa_page("<html>graphic, no coordinates</html>", "u") is None


def test_cap_per_program_keeps_order():
    from skyforge.discover import cap_per_program

    cs = [_cand(f"t{i}", 0, 0) for i in range(6)]
    for c, pid in zip(cs, ["7763", "7763", "6882", "7763", "6882", "1"]):
        c.proposal_id = pid
    assert [c.target for c in cap_per_program(cs, 2)] == ["t0", "t1", "t2", "t4", "t5"]


def test_nodata_mask_white_voids_not_stars():
    """Превью MAST: пустоты вне поля белые (NaN i2d) — маска пустот; насыщенная звезда (3×3) — не пустота."""
    from skyforge.discover import nodata_mask

    img = np.full((200, 300), 20.0)
    img[:, 200:] = 255
    img[50:53, 50:53] = 255
    m = nodata_mask(img)
    assert m[:, 210:].all() and not m[50:53, 50:53].any() and not m[:, :190].any()
