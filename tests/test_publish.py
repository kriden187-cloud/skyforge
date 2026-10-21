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
