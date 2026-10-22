"""Фаза 5: публикация — метаданные туда и обратно, стоп-слова подписи, детерминированность версий, тон при уменьшении."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _meta() -> dict:
    return {
        "title": "Sh 2-305 — JWST NIRCam (showcase)",
        "credit": "NASA, ESA, CSA, STScI; processing: Denis",
        "credit_ru": "NASA, ESA, CSA, STScI; обработка: Denis",
        "description": "Sh 2-305, JWST NIRCam, program 4547. Config configs/sh2305/showcase.yaml @ phase-4.",
        "description_ru": "Sh 2-305, JWST NIRCam, программа 4547. Конфиг configs/sh2305/showcase.yaml, тег phase-4.",
        "copyright": "(c) 2026 Denis. CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/). Data: NASA, ESA, CSA, STScI.",
        "license": "CC BY-SA 4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "config": "configs/sh2305/showcase.yaml",
        "tag": "phase-4",
        "config_digest": "676fb71874f17f51",
        "software": "skyforge test",
    }


def _image(h: int = 300, w: int = 420) -> np.ndarray:
    yy, xx = np.indices((h, w))
    rgb = np.stack([xx / w, yy / h, 0.5 + 0.4 * np.sin(xx / 17.0) * np.cos(yy / 23.0)], -1)
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def test_metadata_roundtrip_png_tiff_jpeg(tmp_path):
    """Заголовок, источник, описание и права читаются обратно из PNG, TIFF и JPEG и совпадают с подписью;
    русский текст — в PNG (iTXt), TIFF (JSON) и JPEG (XMP)."""
    from skyforge.publish import read_meta, write_jpeg, write_png, write_tiff

    meta, u8 = _meta(), _image()
    u16 = (u8.astype(np.uint16) * 257)
    write_png(u8, tmp_path / "m.png", meta)
    write_tiff(u16, tmp_path / "m.tif", meta)
    write_jpeg(u8, tmp_path / "m.jpg", meta, long=200, quality=92)
    for name in ("m.png", "m.tif", "m.jpg"):
        got = read_meta(tmp_path / name)
        for k in ("title", "credit", "description", "copyright"):
            assert got[k] == meta[k], (name, k, got[k])
        assert got["description_ru"] == meta["description_ru"], name
    assert Image.open(tmp_path / "m.jpg").getexif().get_ifd(0x8769).get(0xA001) == 1      # ColorSpace = sRGB


def test_stop_words_rejected():
    from skyforge.publish import find_stop_words, load_publish

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    assert find_stop_words("Первая цветная обработка данных", pub.stop_words)
    assert find_stop_words("The FIRST image ever", pub.stop_words)
    assert find_stop_words("Уникальный кадр", pub.stop_words)
    assert not find_stop_words("Известные более ранние обработки: см. список", pub.stop_words)


def test_publish_yaml_has_no_stop_words():
    """Тексты подписи в configs/publish.yaml — без «первая», «уникальная», «единственная»."""
    from skyforge.publish import find_stop_words, load_publish

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    texts = []
    for o in pub.objects.values():
        texts += o.notes_ru + o.notes_en
        for s in o.styles.values():
            texts += [s.shown_ru, s.shown_en, s.limits_ru, s.limits_en]
        if o.flickr:
            texts += [o.flickr.title, *o.flickr.paragraphs, ", ".join(o.flickr.tags)]
    bad = {t: find_stop_words(t, pub.stop_words) for t in texts if find_stop_words(t, pub.stop_words)}
    assert not bad, bad


def test_caption_has_required_lines_and_no_stop_words():
    """Подпись: объект, телескоп, программа и руководитель, даты, фильтры с длиной волны и цветом, MAST, источник,
    лицензия, конфиг и тег; для Sh2-305 — строка про дату открытия данных и список ранних обработок."""
    from skyforge.publish import Facts, caption_md, find_stop_words, load_publish

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    f = Facts(slug="sh2305", style="showcase", target="Sh 2-305", mission="jwst", telescope="JWST", instrument="NIRCam",
              program="4547", program_title="The stellar & sub-stellar initial mass function in Sharpless 305: imaging",
              pi="McCaughrean, Mark J.", obs_start="2025-03-31", obs_end="2025-03-31", release="2026-04-02",
              filters=[("F182M", 1.82, "#3050ff"), ("F444W", 4.40, "#ff2020")], config="configs/sh2305/showcase.yaml",
              tag="phase-4", config_digest="676fb71874f17f51", bin=1, master_sha256="0" * 64, author="Denis",
              accepted="2026-10-21")
    for lang in ("ru", "en"):
        md = caption_md(f, pub, lang)
        assert not find_stop_words(md, pub.stop_words), lang
        early = ("Тевено", "Карпантье", "Picturavis") if lang == "ru" else ("Thévenot", "Carpentier", "Picturavis")
        for s in ("Sh 2-305", "4547", "McCaughrean", "2025-03-31", "F182M", "#3050ff", "MAST", "NASA, ESA, CSA, STScI",
                  pub.license, "configs/sh2305/showcase.yaml", "phase-4", "2026-04-02", *early):
            assert s in md, (lang, s)
        assert f"{pub.repo}/tree/phase-4" in md and f"{pub.repo}/blob/phase-4/configs/sh2305/showcase.yaml" in md
    hst = Facts(**{**f.__dict__, "mission": "hst", "slug": "ngc3132hst"})
    md = caption_md(hst, pub, "en")
    assert "NASA, ESA, STScI; processing: Denis" in md and "NASA, ESA, CSA, STScI" not in md   # у Hubble нет CSA


def test_commons_wikitext():
    """Шаг 2 публикации: текст страницы файла Commons — {{Information}} с описанием EN и RU, датой наблюдений, {{own}} с
    программой, репозиторием и тегом, автором и источником данных, {{cc-by-sa-4.0}}, категориями из publish.yaml и
    предложенным именем файла; для Sh2-305 — ссылки на другие обработки; без стоп-слов; у Hubble нет CSA."""
    from skyforge.publish import Facts, commons_wikitext, find_stop_words, load_publish

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    f = Facts(slug="sh2305", style="showcase", target="Sh 2-305", mission="jwst", telescope="JWST", instrument="NIRCam",
              program="4547", program_title="The stellar & sub-stellar initial mass function in Sharpless 305: imaging",
              pi="McCaughrean, Mark J.", obs_start="2025-03-31", obs_end="2025-03-31", release="2026-04-02",
              filters=[("F182M", 1.82, "#3050ff"), ("F444W", 4.40, "#ff2020")], config="configs/sh2305/showcase.yaml",
              tag="phase-4", config_digest="676fb71874f17f51", bin=1, master_sha256="0" * 64, author="Denis Krivosheev",
              accepted="2026-10-21")
    w = commons_wikitext(f, pub)
    for s in ("{{Information", "{{en|1=", "{{ru|1=", "|date=2025-03-31", "{{own}}", "4547", "Mark J. McCaughrean",
              "https://github.com/kriden187-cloud/skyforge", "phase-4", "|author=Denis Krivosheev", "NASA, ESA, CSA, STScI",
              "{{cc-by-sa-4.0", "[[Category:Sh 2-305]]", "[[Category:Images by NIRCam]]",
              "Sharpless 305 JWST NIRCam 5 filters (Krivosheev).png", "[[:File:Sharpless 305 JWST NIRCam (long wavelength).jpg"):
        assert s in w, s
    assert not find_stop_words(w, pub.stop_words)
    hst = Facts(**{**f.__dict__, "slug": "ngc3132hst", "mission": "hst", "obs_start": "1995-12-07", "obs_end": "1995-12-07"})
    wh = commons_wikitext(hst, pub)
    assert "NASA, ESA, STScI" in wh and "CSA" not in wh and "[[Category:Hubble images of nebulae]]" in wh


def _sh_facts(**over):
    from skyforge.publish import Facts

    base = dict(slug="sh2305", style="showcase", target="Sh 2-305", mission="jwst", telescope="JWST", instrument="NIRCam",
                program="4547", program_title="", pi="McCaughrean, Mark J.", obs_start="2025-03-31", obs_end="2025-03-31",
                release="2026-04-02", filters=[("F182M", 1.82, "#3050ff"), ("F335M", 3.35, "#c0ff40")],
                config="configs/sh2305/showcase.yaml", tag="phase-4", config_digest="676fb71874f17f51", bin=1,
                master_sha256="0" * 64, author="Denis Krivosheev", accepted="2026-10-21")
    return Facts(**{**base, **over})


def test_flickr_text_and_published_registry():
    """Текст Flickr: заголовок, факты из данных, фильтры с цветом, ссылка на репозиторий и тег, кредит, лицензия, теги,
    без стоп-слов; у Hubble — подписи фильтров вместо длин волн и кредит без CSA. Реестр published — таблица из publish.yaml,
    ссылка Flickr попадает в «other versions» текста Commons."""
    from skyforge.publish import commons_wikitext, find_stop_words, flickr_text, load_publish, published_md

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    f = _sh_facts()
    t = flickr_text(f, pub, "sh2305_showcase", (6076, 6971), 46.0)
    for s in ("FILE TO UPLOAD: sh2305_showcase_web.jpg", "46 MB", "Attribution-ShareAlike (CC BY-SA 4.0)",
              "=== TITLE ===", "Sharpless 305 (Sh 2-305)", "Program 4547, PI Mark J. McCaughrean; observed 2025-03-31",
              "since 2026-04-02", "F182M 1.82 µm blue", "F335M 3.35 µm yellow-green (PAH emission)",
              "https://github.com/kriden187-cloud/skyforge (tag phase-4, configs/sh2305/showcase.yaml)",
              "Credit: NASA, ESA, CSA, STScI; processing: Denis Krivosheev. Processing licensed CC BY-SA 4.0.", "RCW 8"):
        assert s in t, s
    assert not find_stop_words(t, pub.stop_words)
    h = flickr_text(_sh_facts(slug="ngc3132hst", mission="hst", pi="Trauger, John",
                              filters=[("F502N", 0.502, "#0000ff")]), pub, "ngc3132hst_showcase", (934, 1065), 1.7)
    assert "FILE TO UPLOAD: ngc3132hst_showcase_master.png  (934×1065;" in h and "F502N [O III] blue" in h
    assert "PI John Trauger" in h and "CSA" not in h
    md = published_md(f, pub)
    assert "| Flickr | https://www.flickr.com/photos/205062605@N07/55603618036/ | 2026-10-21 | CC BY-SA 4.0 |" in md
    assert "| Wikimedia Commons | (ждёт:" in md
    assert published_md(_sh_facts(slug="ngc3132", style="measured"), pub) is None      # стиль нигде не публиковался
    assert "|other versions=* [https://www.flickr.com/photos/205062605@N07/55603618036/ Flickr (2026-10-21)]" in commons_wikitext(f, pub)


def test_published_registry_by_part(tmp_path):
    """Реестр с частями кадра (W3-2, половины A/B): тексты части берут только свои ссылки, тексты полного кадра — только
    записи без части; таблица реестра — с колонкой части; часть, которой нет в parts, — ошибка загрузки."""
    import pytest

    from skyforge.log import SkyforgeError
    from skyforge.publish import commons_wikitext, load_publish, published_md, reddit_text

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    parts = {pt.name: pt for pt in pub.objects["w3_2"].parts}
    f = _sh_facts(slug="w3_2", pi="Best, William", program="5437", tag="phase-6", config="configs/w3_2/showcase.yaml",
                  filters=[("F140M", 1.40, "#6030ff"), ("F444W", 4.44, "#ff2020")])
    fa, fb = "55605113763", "55605398725"
    ra = reddit_text(f, pub, "w3_2_showcase_A", parts["A"])
    assert fa in ra and fb not in ra and "File:W3_Westerhout_3_JWST_NIRCam_module_A" in ra
    full = reddit_text(f, pub, "w3_2_showcase")
    assert "Full resolution" not in full                                 # полный кадр нигде не выложен
    ca = commons_wikitext(f, pub, parts["A"], (("x.png", "x"),))
    assert f"{fa}/ Flickr (2026-10-22)]" in ca and fb not in ca
    assert fa not in commons_wikitext(f, pub, parts["A2"]) and fa not in commons_wikitext(f, pub)
    md = published_md(f, pub)
    assert "| Часть | Площадка |" in md and f"| A | Flickr | https://www.flickr.com/photos/205062605@N07/{fa}/ |" in md
    assert "| полный кадр | Reddit | (ждёт аккаунта) |" in md
    bad = (ROOT / "configs" / "publish.yaml").read_text(encoding="utf-8").replace("part: B, platform: Flickr",
                                                                                "part: Z, platform: Flickr")
    (tmp_path / "p.yaml").write_text(bad, encoding="utf-8")
    with pytest.raises(SkyforgeError, match="Z"):
        load_publish(tmp_path / "p.yaml")


def test_owner_decisions_in_publish_yaml():
    """Решения владельца 2026-10-21: CC BY-SA 4.0, автор Denis Krivosheev, без водяного знака; Hubble — JPEG 98."""
    from skyforge.publish import load_publish

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    assert pub.license == "CC BY-SA 4.0" and pub.author == "Denis Krivosheev" and not pub.watermark
    assert pub.objects["ngc3132hst"].jpeg_quality == 98 and pub.objects["sh2305"].jpeg_quality is None


def test_versions_are_deterministic(tmp_path):
    from skyforge.publish import write_jpeg, write_png

    meta, u8 = _meta(), _image()
    digests = []
    for k in range(2):
        write_png(u8, tmp_path / f"a{k}.png", meta)
        write_jpeg(u8, tmp_path / f"a{k}.jpg", meta, long=200, quality=92)
        digests.append([hashlib.sha256((tmp_path / f"a{k}.{e}").read_bytes()).hexdigest() for e in ("png", "jpg")])
    assert digests[0] == digests[1]


def test_downsized_version_keeps_tone(tmp_path):
    """Уменьшенная JPEG-версия против уменьшенного (усреднением) мастера: медиана |разности| ≤ 1 уровня по каналам."""
    from skyforge.publish import write_jpeg

    u8 = _image(900, 1260)
    write_jpeg(u8, tmp_path / "w.jpg", _meta(), long=400, quality=92)
    j = np.asarray(Image.open(tmp_path / "w.jpg").convert("RGB"), dtype=np.float32)
    ref = np.asarray(Image.fromarray(u8).resize((j.shape[1], j.shape[0]), Image.BOX), dtype=np.float32)
    assert max(float(np.median(np.abs(j[..., c] - ref[..., c]))) for c in range(3)) <= 1.0


def test_resize_keeps_size_if_smaller():
    from skyforge.publish import resize_long

    u8 = _image(100, 150)
    assert resize_long(u8, 800).shape == u8.shape
    assert max(resize_long(_image(900, 1260), 400).shape[:2]) == 400


@pytest.mark.parametrize("words", [["перв"]])
def test_find_stop_words_word_stems(words):
    from skyforge.publish import find_stop_words

    assert find_stop_words("ПЕРВОЕ", words) == ["перв"]


def test_reddit_text():
    """Фаза 6: текст Reddit — файл, сабреддиты с напоминанием о правилах, заголовок, комментарий с фактами из данных,
    фильтрами с цветами (F140M — фиолетово-синий, отличается от синего F182M), ссылкой на репозиторий, кредитом и
    лицензией; без стоп-слов (служебные заголовки тоже проверяются)."""
    from skyforge.publish import find_stop_words, load_publish, reddit_text

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    f = _sh_facts(slug="w3_2", pi="Best, William", program="5437", obs_start="2024-10-29", release="2025-10-30",
                  tag="phase-6", config="configs/w3_2/showcase.yaml",
                  filters=[("F140M", 1.40, "#6030ff"), ("F182M", 1.82, "#3050ff"), ("F405N", 4.05, "#ff9020")])
    t = reddit_text(f, pub, "w3_2_showcase")
    for s in ("FILE TO UPLOAD: w3_2_showcase_social.jpg", "r/jwst", "check each subreddit's rules", "=== TITLE ===",
              "[OC]", "program 5437 (PI William Best), observed 2024-10-29, public in MAST since 2025-10-30",
              "F140M 1.40 µm violet-blue", "F182M 1.82 µm blue", "F405N 4.05 µm orange (Brackett-alpha)",
              "https://github.com/kriden187-cloud/skyforge (tag phase-6, configs/w3_2/showcase.yaml)",
              "Credit: NASA, ESA, CSA, STScI; processing: Denis Krivosheev. Processing licensed CC BY-SA 4.0"):
        assert s in t, s
    assert not find_stop_words(t, pub.stop_words)


def test_publish_part_is_exact_crop_and_repeatable(tmp_path, monkeypatch):
    """Фаза 6: часть кадра — вырезка из принятого PNG и TIFF без изменения пикселей, хеш пикселей части в publish.json,
    повтор публикации даёт те же файлы (хеши)."""
    import json

    import numpy as np
    from PIL import Image

    import skyforge.publish as P
    from skyforge.config import Config
    from skyforge.paths import Paths
    from skyforge.publish import load_publish, pixels_sha256

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (120, 300, 4), dtype=np.uint8)
    arr[..., 3] = 255
    t16 = arr.astype(np.uint16) * 257
    slices = [(slice(10, 110), slice(0, 120)), (slice(5, 120), slice(170, 300))]
    monkeypatch.setattr(P, "part_slices", lambda *a, **k: slices)
    f = _sh_facts(slug="w3_2", pi="Best, William", program="5437", tag="phase-6", config="configs/w3_2/showcase.yaml",
                  filters=[("F140M", 1.40, "#6030ff"), ("F444W", 4.44, "#ff2020")])
    cfg = Config(target="W3", filters=["F140M", "F444W"])
    parts = [pt for pt in pub.objects["w3_2"].parts if pt.region is not None]
    hashes = []
    for run in ("a", "b"):
        stem = tmp_path / run / "w3_2_showcase"
        stem.parent.mkdir()
        for pt in parts:
            files, sha = P.publish_part(Paths(tmp_path), pub, cfg, f, pt, parts, arr, t16, stem)
            crop = arr[slices[pt.region]]
            png = stem.with_name(f"{stem.name}_{pt.name}_master.png")
            got = np.asarray(Image.open(png))
            assert np.array_equal(got, crop), pt.name
            assert sha == pixels_sha256(Image.fromarray(np.ascontiguousarray(crop)))
            man = json.loads(stem.with_name(f"{stem.name}_{pt.name}_publish.json").read_text(encoding="utf-8"))
            assert man["pixels_sha256"] == sha
            hashes.append(man["files"])
            cap = stem.with_name(f"{stem.name}_{pt.name}_caption.md").read_text(encoding="utf-8")
            assert pt.label_ru in cap and pt.label_en in cap and "w3_2_showcase_master.png" in cap
    assert hashes[:2] == hashes[2:]                                      # повтор — те же файлы
    a_commons = (tmp_path / "a" / "w3_2_showcase_A_commons.txt").read_text(encoding="utf-8")
    assert "module B bright cloud" in a_commons and "8 filters (Krivosheev).png|full field" in a_commons
    assert "fragment" not in a_commons                                   # фрагменты — своя группа


def test_publish_fragment_rect_is_exact_crop(tmp_path):
    """Фаза 6, шаг 3: фрагмент между щелями — прямоугольник rect принятого PNG без изменения пикселей; повтор — те же
    файлы; Commons other versions — другие фрагменты и диптих, без половин A/B; ограничение фрагмента в подписи."""
    import json

    import numpy as np
    import pytest
    from PIL import Image

    import skyforge.publish as P
    from skyforge.config import Config
    from skyforge.log import SkyforgeError
    from skyforge.paths import Paths
    from skyforge.publish import load_publish, pixels_sha256

    pub = load_publish(ROOT / "configs" / "publish.yaml")
    parts = pub.objects["w3_2"].parts
    frags = [pt for pt in parts if pt.rect is not None]
    assert [pt.name for pt in frags] == ["A2", "A4", "B3"]
    H, W = 1682, max(pt.rect[3] for pt in frags)
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, (H, W, 4), dtype=np.uint8)
    arr[..., 3] = 255
    t16 = arr.astype(np.uint16) * 257
    f = _sh_facts(slug="w3_2", pi="Best, William", program="5437", tag="phase-6", config="configs/w3_2/showcase.yaml",
                  filters=[("F140M", 1.40, "#6030ff"), ("F444W", 4.44, "#ff2020")])
    cfg = Config(target="W3", filters=["F140M", "F444W"])
    hashes = []
    for run in ("a", "b"):
        stem = tmp_path / run / "w3_2_showcase"
        stem.parent.mkdir()
        for pt in frags:
            files, sha = P.publish_part(Paths(tmp_path), pub, cfg, f, pt, parts, arr, t16, stem)
            r0, r1, c0, c1 = pt.rect
            crop = arr[r0:r1, c0:c1]
            got = np.asarray(Image.open(stem.with_name(f"{stem.name}_{pt.name}_master.png")))
            assert np.array_equal(got, crop), pt.name
            assert sha == pixels_sha256(Image.fromarray(np.ascontiguousarray(crop)))
            man = json.loads(stem.with_name(f"{stem.name}_{pt.name}_publish.json").read_text(encoding="utf-8"))
            assert man["region"] == pt.rect and man["slice_rows_cols"] == pt.rect
            hashes.append(man["files"])
            cap = stem.with_name(f"{stem.name}_{pt.name}_caption.md").read_text(encoding="utf-8")
            assert pt.label_ru in cap and "полное поле с заполненными щелями — мастер диптиха" in cap
            assert "800 px" in cap and "13% площади модуля" not in cap         # ограничение фрагмента, не диптиха
            commons = stem.with_name(f"{stem.name}_{pt.name}_commons.txt").read_text(encoding="utf-8")
            others = [o.commons_filename for o in frags if o is not pt]
            assert all(o in commons for o in others) and "8 filters (Krivosheev).png|full field" in commons
            assert "module A dust lanes" not in commons and "module B bright cloud" not in commons
            flickr = stem.with_name(f"{stem.name}_{pt.name}_flickr.txt").read_text(encoding="utf-8")
            assert pt.flickr_title in flickr and "the full field with the gaps filled is published separately" in flickr
    assert hashes[:3] == hashes[3:]
    with pytest.raises(SkyforgeError, match="за краем"):
        P.publish_part(Paths(tmp_path), pub, cfg, f, frags[-1], parts, arr[:, :3000], t16[:, :3000], stem)


def test_part_spec_needs_region_or_rect():
    import pytest
    from pydantic import ValidationError

    from skyforge.publish import PartSpec

    base = {"name": "X", "style": "showcase", "label_ru": "x", "label_en": "x"}
    assert PartSpec(**base, region=0).rect is None
    assert PartSpec(**base, rect=[0, 10, 0, 20]).region is None
    for bad in ({}, {"region": 0, "rect": [0, 10, 0, 20]}, {"rect": [10, 0, 0, 20]}, {"rect": [0, 10, 0]}):
        with pytest.raises(ValidationError):
            PartSpec(**base, **bad)
