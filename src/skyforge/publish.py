"""Фаза 5: публикация принятых результатов — мастер, версии под площадки, подпись RU/EN, галерея конфигов.

Берётся принятый результат (report.baseline: data/baselines/<slug>/<тег>/<стиль>_bin<N>.png), а не последний прогон:
мастер PNG — его пиксели с полными метаданными; мастер TIFF (16 бит) — из прогона, чей PNG совпадает с принятым
по пикселям. Версии JPEG — уменьшение Lanczos без резкости, sRGB (EXIF ColorSpace), 4:4:4; подпись — в XMP (UTF-8,
русский и английский) и копией по-английски в EXIF. Всё детерминировано: ни времени, ни случайного в файлах нет.
Факты подписи — из конфига, заголовков FITS, манифеста и метаданных принятого PNG; configs/publish.yaml — только то,
чего в данных нет (лицензия, формула источника, примечания, «что показано / ограничения»).
"""

from __future__ import annotations

import colorsys
import hashlib
import html
import json
import re
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pydantic import BaseModel, ConfigDict, Field

from skyforge import __version__
from skyforge.config import Config
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths

Image.MAX_IMAGE_PIXELS = None
PUBLISH_VERSION = 1
NS = {"x": "adobe:ns:meta/", "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#", "dc": "http://purl.org/dc/elements/1.1/",
      "xmpRights": "http://ns.adobe.com/xap/1.0/rights/", "photoshop": "http://ns.adobe.com/photoshop/1.0/",
      "skyforge": "https://github.com/skyforge/ns/1.0/"}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


# --- настройки (configs/publish.yaml) ---


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SizeSpec(_Strict):
    long: int = Field(..., ge=64, description="длинная сторона, px")
    quality: int = Field(92, ge=50, le=100)


class StyleText(_Strict):
    shown_ru: str = ""
    shown_en: str = ""
    limits_ru: str = ""
    limits_en: str = ""


class CommonsOther(_Strict):
    file: str                              # имя файла на Commons без «File:»
    ru: str
    en: str


class CommonsSpec(_Strict):
    style: str                             # какой стиль загружается на Commons
    filename: str
    categories: list[str]
    others: list[CommonsOther] = []        # другие обработки тех же данных на Commons


class ObjectPub(_Strict):
    accepted: str
    commons: CommonsSpec | None = None
    pi: str | None = None                  # если в заголовках FITS нет (HST drz)
    jpeg_quality: int | None = Field(None, ge=50, le=100)   # качество JPEG всех версий объекта вместо sizes.*.quality
    notes_ru: list[str] = []
    notes_en: list[str] = []
    styles: dict[str, StyleText]


class PublishConfig(_Strict):
    author: str | None = None
    license: str
    license_url: str
    watermark: bool = False
    repo: str | None = None                 # адрес репозитория: ссылки на конфиг и тег
    credit: dict[str, str]
    sizes: dict[str, SizeSpec]
    stop_words: list[str]
    objects: dict[str, ObjectPub]


def load_publish(path: Path) -> PublishConfig:
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for o in (data.get("objects") or {}).values():   # даты YAML → строки
        if "accepted" in o:
            o["accepted"] = str(o["accepted"])
    pub = PublishConfig.model_validate(data)
    if pub.watermark:
        raise SkyforgeError("publish.watermark: подпись на картинке не реализована — решение владельца: только текст и метаданные")
    return pub


def find_stop_words(text: str, words: list[str]) -> list[str]:
    """Слова стоп-списка (по основе, без учёта регистра), найденные в тексте."""
    low = text.lower()
    return [w for w in words if re.search(r"(?<![\w])" + re.escape(w.lower()), low)]


# --- факты подписи ---


@dataclass
class Facts:
    slug: str
    style: str
    target: str
    mission: str
    telescope: str
    instrument: str
    program: str
    program_title: str
    pi: str
    obs_start: str
    obs_end: str
    release: str
    filters: list[tuple[str, float, str]]          # (фильтр, длина волны мкм, цвет #rrggbb)
    config: str
    tag: str
    config_digest: str
    bin: int
    master_sha256: str                             # sha256 пикселей принятого PNG (режим, размер, байты)
    author: str
    accepted: str
    extra: dict[str, Any] = field(default_factory=dict)


TELESCOPE = {"jwst": ("JWST", "космический телескоп «Джеймс Уэбб» (JWST)", "James Webb Space Telescope (JWST)"),
             "hst": ("HST", "космический телескоп «Хаббл» (HST)", "Hubble Space Telescope (HST)")}
INSTRUMENT = {"nircam": "NIRCam", "miri": "MIRI", "wfpc2": "WFPC2", "wfc3": "WFC3", "acs": "ACS"}


def pixels_sha256(img: Image.Image) -> str:
    h = hashlib.sha256(f"{img.mode}:{img.size[0]}x{img.size[1]}:".encode())
    h.update(img.tobytes())
    return h.hexdigest()


def accepted_png(paths: Paths, cfg: Config, style: str) -> Path:
    from skyforge.baseline import baseline_dir

    if not cfg.report.baseline:
        raise SkyforgeError(f"{cfg.slug}: нет report.baseline — публикуется только принятый результат")
    found = sorted(baseline_dir(paths, cfg.slug, cfg.report.baseline).glob(f"{style}_bin*.png"))
    if len(found) != 1:
        raise SkyforgeError(f"{cfg.slug}/{cfg.report.baseline}: принятый PNG стиля {style} — {len(found)} файлов, нужен один")
    return found[0]


def _png_meta(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        d = im.info.get("Description")
    try:
        return json.loads(d) if d else {}
    except json.JSONDecodeError:
        return {}


def object_facts(paths: Paths, cfg: Config, style: str, pub: PublishConfig, base_png: Path) -> Facts:
    from astropy.io import fits

    from skyforge.fetch import load_manifest
    from skyforge.filters import wavelength_um

    prods = load_manifest(cfg, paths).get("products", {})
    heads = []
    for f in cfg.filters:
        if f not in prods:
            raise SkyforgeError(f"{cfg.slug}: фильтра {f} нет в манифесте — skyforge fetch")
        heads.append(fits.getheader(paths.abs(prods[f]["path"]), 0))
    get = lambda k: next((str(h[k]).strip() for h in heads if k in h and str(h[k]).strip()), "")
    beg = sorted(str(h.get("DATE-BEG", h.get("DATE-OBS", "")))[:10] for h in heads)
    end = sorted(str(h.get("DATE-END", h.get("DATE-OBS", "")))[:10] for h in heads)
    po = pub.objects[cfg.slug]
    meta = _png_meta(base_png)
    colors = meta.get("filters") or {}
    filt = sorted(((f, wavelength_um(f, cfg.mission, cfg.instrument), colors.get(f, "")) for f in cfg.filters),
                  key=lambda t: t[1])
    with Image.open(base_png) as im:
        sha = pixels_sha256(im)
    program = get("PROGRAM").lstrip("0") or get("PROPOSID") or (cfg.proposal_id or "")
    return Facts(
        slug=cfg.slug, style=style, target=cfg.target, mission=cfg.mission, telescope=TELESCOPE[cfg.mission][0],
        instrument=INSTRUMENT.get(cfg.instrument.lower(), cfg.instrument.upper()), program=program,
        program_title=get("TITLE"), pi=po.pi or get("PI_NAME"), obs_start=beg[0], obs_end=end[-1],
        release=min((p.get("release") or "") for f, p in prods.items() if f in cfg.filters),
        filters=filt, config=f"configs/{cfg.slug}/{style}.yaml", tag=cfg.report.baseline or "",
        config_digest=meta.get("config_digest", ""), bin=int(re.search(r"_bin(\d+)\.png$", base_png.name).group(1)),
        master_sha256=sha, author=pub.author or cfg.author, accepted=po.accepted)


def color_name(hexc: str, lang: str) -> str:
    if not re.fullmatch(r"#[0-9a-fA-F]{6}", hexc or ""):
        return "—"
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, s, _ = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.15:
        return "белый" if lang == "ru" else "white"
    names = [(15, "красный", "red"), (45, "оранжевый", "orange"), (70, "жёлтый", "yellow"), (100, "жёлто-зелёный", "yellow-green"),
             (150, "зелёный", "green"), (190, "бирюзовый", "cyan"), (215, "голубой", "light blue"), (260, "синий", "blue"),
             (300, "фиолетовый", "violet"), (345, "пурпурный", "magenta"), (361, "красный", "red")]
    deg = h * 360
    for lim, ru, en in names:
        if deg < lim:
            return ru if lang == "ru" else en
    return "—"


def _um(w: float, lang: str) -> str:
    s = f"{w:.3f}" if w < 1 else f"{w:.2f}"      # узкие фильтры Hubble: 0,656 и 0,658 мкм
    return (s.replace(".", ",") + " мкм") if lang == "ru" else (s + " µm")


def credit_line(f: Facts, pub: PublishConfig, lang: str) -> str:
    return f"{pub.credit[f.mission]}; {'обработка' if lang == 'ru' else 'processing'}: {f.author}"


def caption_md(f: Facts, pub: PublishConfig, lang: str) -> str:
    po = pub.objects[f.slug]
    st = po.styles.get(f.style, StyleText())
    ru = lang == "ru"
    tel = TELESCOPE[f.mission][1 if ru else 2]
    obs = f.obs_start if f.obs_start == f.obs_end else f"{f.obs_start} … {f.obs_end}"
    prog = f"{f.program}" + (f" «{f.program_title}»" if ru and f.program_title else f" “{f.program_title}”" if f.program_title else "")
    fl = "; ".join(f"{n} {_um(w, lang)} — {color_name(c, lang)} ({c})" for n, w, c in f.filters)
    run = f"skyforge run {f.slug} -s {f.style} --bin {f.bin}"
    cfg_ref = f"[`{f.config}`]({pub.repo}/blob/{f.tag}/{f.config})" if pub.repo else f"`{f.config}`"
    tag_ref = f"[`{f.tag}`]({pub.repo}/tree/{f.tag})" if pub.repo else f"`{f.tag}`"
    if ru:
        lines = [f"# {f.target} — {f.telescope} {f.instrument}, стиль {f.style}", "",
                 f"- **Объект:** {f.target}",
                 f"- **Телескоп и инструмент:** {tel}, {f.instrument}",
                 f"- **Программа:** {prog}; руководитель — {f.pi}",
                 f"- **Наблюдения:** {obs} (UTC); данные открыты в архиве: {f.release}",
                 f"- **Фильтры (длина волны — цвет на картинке):** {fl}",
                 "- **Источник данных:** MAST (Mikulski Archive for Space Telescopes, STScI)",
                 f"- **Указание источника:** {credit_line(f, pub, 'ru')}",
                 f"- **Лицензия на обработку:** {pub.license} ({pub.license_url}); данные — на условиях {pub.credit[f.mission]}",
                 f"- **Воспроизведение:** конфиг {cfg_ref}, тег {tag_ref} (принято {f.accepted}); `git checkout {f.tag}`, "
                 f"`{run}`; хеш конфига `{f.config_digest}`, хеш пикселей итога (sha256) `{f.master_sha256}`"]
        if st.shown_ru:
            lines.append(f"- **Что показано:** {st.shown_ru}")
        if st.limits_ru:
            lines.append(f"- **Ограничения:** {st.limits_ru}")
        lines += [f"- **Примечание:** {n}" for n in po.notes_ru]
    else:
        lines = [f"# {f.target} — {f.telescope} {f.instrument}, {f.style} style", "",
                 f"- **Object:** {f.target}",
                 f"- **Telescope and instrument:** {tel}, {f.instrument}",
                 f"- **Program:** {prog}; PI — {f.pi}",
                 f"- **Observations:** {obs} (UTC); data public in the archive since: {f.release}",
                 f"- **Filters (wavelength — colour in the image):** {fl}",
                 "- **Data source:** MAST (Mikulski Archive for Space Telescopes, STScI)",
                 f"- **Credit:** {credit_line(f, pub, 'en')}",
                 f"- **License of the processing:** {pub.license} ({pub.license_url}); data under {pub.credit[f.mission]} terms",
                 f"- **Reproduction:** config {cfg_ref}, tag {tag_ref} (accepted {f.accepted}); `git checkout {f.tag}`, "
                 f"`{run}`; config digest `{f.config_digest}`, final pixel hash (sha256) `{f.master_sha256}`"]
        if st.shown_en:
            lines.append(f"- **What is shown:** {st.shown_en}")
        if st.limits_en:
            lines.append(f"- **Limitations:** {st.limits_en}")
        lines += [f"- **Note:** {n}" for n in po.notes_en]
    return "\n".join(lines) + "\n"


def _person(name: str) -> str:
    """«McCaughrean, Mark J.» → «Mark J. McCaughrean» (как в тексте)."""
    if "," in name:
        last, first = (x.strip() for x in name.split(",", 1))
        return f"{first} {last}"
    return name


def commons_wikitext(f: Facts, pub: PublishConfig) -> str:
    """Текст страницы файла Wikimedia Commons: {{Information}} (описание EN и RU, дата наблюдений, {{own}} с программой,
    репозиторием и тегом, автор и источник данных), лицензия {{cc-by-sa-4.0}}, категории из publish.yaml."""
    po = pub.objects[f.slug]
    co = po.commons
    if co is None:
        raise SkyforgeError(f"{f.slug}: нет раздела commons в configs/publish.yaml")
    st = po.styles.get(f.style, StyleText())
    tel_en, tel_ru = TELESCOPE[f.mission][2], TELESCOPE[f.mission][1]
    pi = _person(f.pi)
    fl_en = "; ".join(f"{n} {_um(w, 'en')} — {color_name(c, 'en')}" for n, w, c in f.filters)
    fl_ru = "; ".join(f"{n} {_um(w, 'ru')} — {color_name(c, 'ru')}" for n, w, c in f.filters)
    title_en = f" “{f.program_title}”" if f.program_title else ""
    title_ru = f" «{f.program_title}»" if f.program_title else ""
    en = (f"{f.target}, {tel_en}, {f.instrument}. Program {f.program}{title_en}, PI {pi}; observed {f.obs_start} (UTC), "
          f"data public since {f.release}. Filters (wavelength — colour in the image): {fl_en}. {st.shown_en}")
    ru = (f"{f.target}, {tel_ru}, {f.instrument}. Программа {f.program}{title_ru}, руководитель {pi}; наблюдения "
          f"{f.obs_start} (UTC), данные открыты {f.release}. Фильтры (длина волны — цвет на картинке): {fl_ru}. {st.shown_ru}")
    if co.others:
        en += " Other processings of the same data on Commons: " + ", ".join(f"[[:File:{o.file}|{o.en}]]" for o in co.others) + "."
        ru += " Другие обработки тех же данных на Commons: " + ", ".join(f"[[:File:{o.file}|{o.ru}]]" for o in co.others) + "."
    repo = f"{pub.repo}/tree/{f.tag}" if pub.repo else f"tag {f.tag}"
    data = pub.credit[f.mission]
    lines = [
        f"<!-- Имя файла при загрузке: {co.filename} — текст ниже вставляется целиком как описание страницы файла -->",
        "=={{int:filedesc}}==",
        "{{Information",
        f"|description={{{{en|1={en}}}}}",
        f"{{{{ru|1={ru}}}}}",
        f"|date={f.obs_start}",
        f"|source={{{{own}}}}: processing of public {f.telescope} data from MAST (Mikulski Archive for Space Telescopes, STScI), "
        f"program {f.program}. Pipeline and configuration: {repo} ({f.config})",
        f"|author={f.author}; data: {data}",
        "|permission=",
        "|other versions=",
        "}}",
        "",
        "=={{int:license-header}}==",
        f"{{{{cc-by-sa-4.0|1={f.author}; data: {data}}}}}",
        "",
        *[f"[[Category:{c}]]" for c in co.categories],
    ]
    return "\n".join(lines) + "\n"


def build_meta(f: Facts, pub: PublishConfig) -> dict[str, Any]:
    tel, inst = f.telescope, f.instrument
    fl = ", ".join(n for n, _, _ in f.filters)
    year = f.accepted[:4]
    return {
        "title": f"{f.target} — {tel} {inst} ({f.style})",
        "credit": credit_line(f, pub, "en"),
        "credit_ru": credit_line(f, pub, "ru"),
        "description": f"{f.target}, {tel} {inst}, filters {fl}; program {f.program} (PI {f.pi}), observed {f.obs_start}. "
                       f"Config {f.config} @ {f.tag}, digest {f.config_digest}, pixels sha256 {f.master_sha256[:16]}.",
        "description_ru": f"{f.target}, {tel} {inst}, фильтры {fl}; программа {f.program} (руководитель {f.pi}), "
                          f"наблюдения {f.obs_start}. Конфиг {f.config}, тег {f.tag}, хеш {f.config_digest}, "
                          f"пиксели sha256 {f.master_sha256[:16]}.",
        "copyright": f"© {year} {f.author}. {pub.license} ({pub.license_url}). Data: {pub.credit[f.mission]}.",
        "license": pub.license, "license_url": pub.license_url, "config": f.config, "tag": f.tag,
        "repo": f"{pub.repo}/tree/{f.tag}" if pub.repo else "",
        "config_digest": f.config_digest, "pixels_sha256": f.master_sha256,
        "software": f"skyforge {__version__} (publish v{PUBLISH_VERSION})",
    }


# --- запись и чтение файлов ---


def _ascii(s: str) -> str:
    return s.replace("©", "(c)").replace("—", "-").replace("µ", "u").encode("ascii", "replace").decode()


def write_png(arr: np.ndarray, path: Path, meta: dict[str, Any]) -> Path:
    info = PngInfo()
    info.add(b"sRGB", b"\x00")                                  # sRGB, перцептивный рендеринг
    info.add_itxt("Title", meta["title"])
    info.add_itxt("Author", meta["credit"])
    info.add_itxt("Description", meta["description"])
    info.add_itxt("Description.ru", meta["description_ru"])
    info.add_itxt("Copyright", meta["copyright"])
    info.add_itxt("Software", meta["software"])
    info.add_itxt("Source", "MAST (Mikulski Archive for Space Telescopes, STScI)")
    info.add_itxt("skyforge", json.dumps(meta, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, pnginfo=info)
    return path


def write_tiff(arr16: np.ndarray, path: Path, meta: dict[str, Any]) -> Path:
    import tifffile

    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, arr16, photometric="rgb", compression="zlib", metadata=None,
                     description=json.dumps(meta, ensure_ascii=True, sort_keys=True), software=_ascii(meta["software"]),
                     extratags=[(315, "s", 0, _ascii(meta["credit"]), True), (33432, "s", 0, _ascii(meta["copyright"]), True)])
    return path


def _xmp(meta: dict[str, Any]) -> bytes:
    e = lambda s: html.escape(str(s), quote=False)
    body = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        f'<x:xmpmeta xmlns:x="{NS["x"]}"><rdf:RDF xmlns:rdf="{NS["rdf"]}">'
        f'<rdf:Description rdf:about="" xmlns:dc="{NS["dc"]}" xmlns:xmpRights="{NS["xmpRights"]}" '
        f'xmlns:photoshop="{NS["photoshop"]}" xmlns:skyforge="{NS["skyforge"]}">'
        f'<dc:title><rdf:Alt><rdf:li xml:lang="x-default">{e(meta["title"])}</rdf:li></rdf:Alt></dc:title>'
        f'<dc:creator><rdf:Seq><rdf:li>{e(meta["credit"])}</rdf:li></rdf:Seq></dc:creator>'
        f'<dc:description><rdf:Alt><rdf:li xml:lang="x-default">{e(meta["description"])}</rdf:li>'
        f'<rdf:li xml:lang="en">{e(meta["description"])}</rdf:li><rdf:li xml:lang="ru">{e(meta["description_ru"])}</rdf:li>'
        f'</rdf:Alt></dc:description>'
        f'<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">{e(meta["copyright"])}</rdf:li></rdf:Alt></dc:rights>'
        f'<xmpRights:WebStatement>{e(meta["license_url"])}</xmpRights:WebStatement>'
        f'<photoshop:Credit>{e(meta["credit"])}</photoshop:Credit>'
        f'<skyforge:meta>{e(json.dumps(meta, ensure_ascii=False, sort_keys=True))}</skyforge:meta>'
        '</rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end="r"?>'
    )
    return body.encode("utf-8")


def resize_long(rgb: np.ndarray, long: int) -> np.ndarray:
    """Уменьшение до длинной стороны long (Lanczos); меньше — без изменений (не увеличивается)."""
    h, w = rgb.shape[:2]
    if max(h, w) <= long:
        return rgb
    s = long / max(h, w)
    size = (max(1, round(w * s)), max(1, round(h * s)))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS))


def write_jpeg(rgb: np.ndarray, path: Path, meta: dict[str, Any], long: int, quality: int) -> Path:
    img = Image.fromarray(resize_long(np.ascontiguousarray(rgb[..., :3]), long))
    exif = Image.Exif()
    exif[0x010E] = _ascii(meta["description"])     # ImageDescription
    exif[0x013B] = _ascii(meta["credit"])          # Artist
    exif[0x8298] = _ascii(meta["copyright"])       # Copyright
    exif[0x0131] = _ascii(meta["software"])        # Software
    exif.get_ifd(0x8769)[0xA001] = 1               # Exif IFD: ColorSpace = sRGB
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, "JPEG", quality=quality, subsampling=0, optimize=False, progressive=False,
             exif=exif.tobytes(), xmp=_xmp(meta))
    return path


def read_meta(path: Path) -> dict[str, str]:
    """Обратное чтение подписи: title, credit, description, description_ru, copyright."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".png":
        with Image.open(path) as im:
            t = im.text
        return {"title": t.get("Title", ""), "credit": t.get("Author", ""), "description": t.get("Description", ""),
                "description_ru": t.get("Description.ru", ""), "copyright": t.get("Copyright", "")}
    if suf in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            m = json.loads(tf.pages[0].tags["ImageDescription"].value)
        return {k: m.get(k, "") for k in ("title", "credit", "description", "description_ru", "copyright")}
    if suf in (".jpg", ".jpeg"):
        with Image.open(path) as im:
            raw = im.info.get("xmp", b"")
        root = ET.fromstring(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        d = root.find(".//rdf:Description", NS)
        alt = lambda tag, lang="x-default": next((li.text or "" for li in d.findall(f"dc:{tag}/rdf:Alt/rdf:li", NS)
                                                  if li.get(XML_LANG) == lang), "")
        return {"title": alt("title"), "credit": d.findtext("dc:creator/rdf:Seq/rdf:li", "", NS),
                "description": alt("description"), "description_ru": alt("description", "ru"), "copyright": alt("rights")}
    raise SkyforgeError(f"read_meta: неизвестный формат {path.name}")


# --- поиск TIFF принятого результата ---


def find_accepted_tiff(paths: Paths, slug: str, style: str, base: Image.Image) -> tuple[Path, Path]:
    """Прогон, чей PNG совпадает с принятым по пикселям (новые — первыми), и его 16-битный TIFF."""
    want = np.asarray(base)
    root = paths.out_root / slug
    runs = sorted((d for d in root.iterdir() if d.is_dir() and d.name[:4].isdigit()), reverse=True) if root.exists() else []
    for d in runs:
        for png in [*d.glob(f"{slug}_{style}*.png"), *(d / style).glob(f"{slug}_{style}*.png")]:
            tif = png.with_suffix(".tif")
            if not tif.exists():
                continue
            with Image.open(png) as im:
                if im.size != base.size or im.mode != base.mode:
                    continue
                if np.array_equal(np.asarray(im), want):
                    return png, tif
    raise SkyforgeError(f"{slug}/{style}: нет прогона с TIFF, совпадающего с принятым PNG по пикселям — "
                        f"skyforge run {slug} -s {style} --bin <bin принятого> --offline")


# --- публикация и галерея ---


def _style_cfg(paths: Paths, name: str, style: str) -> Config:
    p = paths.root / "configs" / name / f"{style}.yaml"
    if not p.is_file():
        raise SkyforgeError(f"нет конфига {paths.rel(p)}")
    return Config.load(p, style)


def publish_object(paths: Paths, pub: PublishConfig, name: str, style: str | None, day: str) -> list[Path]:
    if name not in pub.objects:
        raise SkyforgeError(f"{name}: нет раздела в configs/publish.yaml (objects.{name})")
    styles = [style] if style else list(pub.objects[name].styles)
    out_files: list[Path] = []
    for st in styles:
        cfg = _style_cfg(paths, name, st)
        base_png = accepted_png(paths, cfg, st)
        with Image.open(base_png) as im:
            im.load()
            base = im.copy()
        arr = np.asarray(base)
        run_png, run_tif = find_accepted_tiff(paths, cfg.slug, st, base)
        import tifffile

        t16 = tifffile.imread(run_tif)
        d8 = np.abs(np.rint(t16[..., :3] / 257.0) - arr[..., :3].astype(np.float64)).max()
        if d8 > 1:
            raise SkyforgeError(f"{cfg.slug}/{st}: TIFF {paths.rel(run_tif)} расходится с принятым PNG на {d8:.0f} уровней")
        f = object_facts(paths, cfg, st, pub, base_png)
        caps = {lang: caption_md(f, pub, lang) for lang in ("ru", "en")}
        bad = find_stop_words(caps["ru"] + caps["en"], pub.stop_words)
        if bad:
            raise SkyforgeError(f"{cfg.slug}/{st}: в подписи слова из стоп-списка: {bad}")
        meta = build_meta(f, pub)
        out = paths.out_root / "publish" / cfg.slug / day
        stem = out / f"{cfg.slug}_{st}"
        files = [write_png(arr, stem.with_name(stem.name + "_master.png"), meta),
                 write_tiff(t16, stem.with_name(stem.name + "_master.tif"), meta)]
        for k, spec in pub.sizes.items():
            q = pub.objects[name].jpeg_quality or spec.quality
            files.append(write_jpeg(arr, stem.with_name(f"{stem.name}_{k}.jpg"), meta, spec.long, q))
        cap = stem.with_name(stem.name + "_caption.md")
        cap.write_text(caps["ru"] + "\n---\n\n" + caps["en"], encoding="utf-8")
        files.append(cap)
        co = pub.objects[name].commons
        if co is not None and co.style == st:
            wt = commons_wikitext(f, pub)
            bad = find_stop_words(wt, pub.stop_words)
            if bad:
                raise SkyforgeError(f"{cfg.slug}/{st}: в тексте для Commons слова из стоп-списка: {bad}")
            cpath = stem.with_name(stem.name + "_commons.txt")
            cpath.write_text(wt, encoding="utf-8")
            files.append(cpath)
        man = {"accepted_png": paths.rel(base_png), "tiff_from": paths.rel(run_tif),
               "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, "meta": meta}
        mpath = stem.with_name(stem.name + "_publish.json")
        mpath.write_text(json.dumps(man, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_files += files + [mpath]
        console.print(f"  {cfg.slug}/{st}: принятый {paths.rel(base_png)}, TIFF из {paths.rel(run_tif.parent)}")
    return out_files


def build_gallery(paths: Paths, pub: PublishConfig) -> Path:
    """docs/GALLERY.md и docs/gallery/<slug>_<style>.jpg — по разделу на объект и стиль из configs/publish.yaml."""
    docs = paths.root / "docs"
    gdir = docs / "gallery"
    gdir.mkdir(parents=True, exist_ok=True)
    spec = pub.sizes["preview"]
    parts = ["# Галерея конфигов Skyforge", "",
             "Каждый раздел — принятый результат: превью, подпись, конфиг, тег и команда, которой получить тот же файл.",
             "Файл создаёт `skyforge gallery` из `configs/publish.yaml`, принятых PNG (`data/baselines/`) и заголовков данных;",
             "руками не правится.", ""]
    for name, po in pub.objects.items():
        for st in po.styles:
            cfg = _style_cfg(paths, name, st)
            base_png = accepted_png(paths, cfg, st)
            f = object_facts(paths, cfg, st, pub, base_png)
            with Image.open(base_png) as im:
                arr = np.asarray(im.convert("RGB"))
            jpg = write_jpeg(arr, gdir / f"{cfg.slug}_{st}.jpg", build_meta(f, pub), spec.long, po.jpeg_quality or spec.quality)
            stt = po.styles[st]
            fl = "; ".join(f"{n} {_um(w, 'ru')} — {color_name(c, 'ru')} ({c})" for n, w, c in f.filters)
            parts += [f"## {f.target} — {f.telescope} {f.instrument}, стиль {st}", "",
                      f"![{f.target}, {st}](gallery/{jpg.name})", "",
                      f"- **Программа:** {f.program}, руководитель — {f.pi}; наблюдения {f.obs_start}; данные открыты {f.release}",
                      f"- **Фильтры:** {fl}",
                      f"- **Указание источника:** {credit_line(f, pub, 'ru')}; лицензия на обработку — {pub.license}",
                      f"- **Конфиг:** [{f.config}](../{f.config}), тег "
                      + (f"[`{f.tag}`]({pub.repo}/tree/{f.tag})" if pub.repo else f"`{f.tag}`") + f", принято {f.accepted}",
                      f"- **Хеш пикселей итога (sha256):** `{f.master_sha256}`",
                      f"- **Что показано:** {stt.shown_ru}", f"- **Ограничения:** {stt.limits_ru}"]
            parts += [f"- **Примечание:** {n}" for n in po.notes_ru]
            parts += ["- **Как получить тот же файл:**", "", "  ```", f"  git checkout {f.tag}",
                      f"  skyforge run {name} -s {st} --bin {f.bin}        # данные из MAST при первом запуске",
                      f"  skyforge publish {name} -s {st}", "  ```", ""]
            console.print(f"  галерея: {cfg.slug}/{st}")
    md = docs / "GALLERY.md"
    md.write_text("\n".join(parts), encoding="utf-8")
    return md
