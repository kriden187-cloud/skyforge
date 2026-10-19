"""HTML-отчёт: все стадии, параметры и копия конфига на одной странице. Картинки — рядом в той же папке."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

STAGES = [
    ("fetch", "1. Данные", "Какие файлы взяты и откуда."),
    ("build", "1½. Сборка из уровня 2",
     "В кадре детектора полос нет; остаток уровней в перекрытиях мал; FWHM звёзд к i2d в допуске (±8% при FWHM < 2,5 px, иначе ±5%), поток к отдельным экспозициям — в 1%."),
    ("inspect", "2. Сырые кадры", "На каждом превью узнаётся объект, пустых и битых кадров нет."),
    ("align", "3. Совмещение", "Звёзды на границах клеток шахматки не смещены, на наложении нет цветных ореолов."),
    ("clean", "3½. Очистка", "Стыки плиток и градиенты ушли в модель фона, туманность осталась; ядро звезды белое."),
    ("stretch", "4. Растяжка", "Пик фона прижат к нулю, хвост гистограммы тянется вправо, детали не выжжены."),
    ("compose", "5. Цвет", "Фон нейтральный, без зелёного или фиолетового оттенка; структура как в пресс-релизе."),
    ("finish", "6. Финиш", "На кропе 100% нет «пластилина» от денойза и светлых ореолов от резкости."),
    ("baseline", "7. Сравнение с принятым",
     "Разность с предыдущим тегом по туманности — только там, где её ждали; фактура в кольце в пределах 10%; кропы без новых артефактов."),
    ("sweep", "Перебор параметров", "Выбери клетку, которая нравится, и перенеси её числа в YAML."),
]

TEMPLATE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<style>
:root{--bg:#0e1014;--panel:#171a21;--line:#2a2f3a;--text:#e6e8ee;--muted:#9aa3b2;--accent:#7fb2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:0 0 4px}.muted{color:var(--muted)}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin:18px 0}
.check{border-left:3px solid var(--accent);padding:4px 10px;margin:8px 0 14px;color:var(--muted)}
.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.imgs a{display:block}.imgs img{width:100%;border-radius:6px;border:1px solid var(--line);background:#000}
.hero img{width:100%;border-radius:8px}
.pair{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.note{margin:4px 0 10px;font-size:14px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:500}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
pre{background:#0a0c10;border:1px solid var(--line);border-radius:6px;padding:12px;overflow:auto;font-size:12.5px}
a{color:var(--accent)}code{color:#c9d4ff}
</style></head><body><main>
<h1>{{ title }}</h1>
<div class="muted">{{ subtitle }}</div>
{% if hero %}<section class="hero"><div class="pair">
{% if reference %}<div><a href="{{ reference }}"><img src="{{ reference }}" alt="референс"></a><div class="muted">референс</div></div>{% endif %}
<div><a href="{{ hero_full }}"><img src="{{ hero }}" alt="итог"></a>{% if reference %}<div class="muted">наш итог</div>{% endif %}</div></div>
<div class="muted">Полный размер: {% for f in finals %}<a href="{{ f }}">{{ f }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</div></section>{% endif %}

{% if crops %}<section><h2>Кропы приёмки, 100%</h2><div class="imgs">{% for c in crops %}
<a href="{{ c.file }}"><img src="{{ c.file }}" alt="{{ c.name }}" style="image-rendering:pixelated"></a>{% endfor %}</div>
<div class="muted">{% for c in crops %}{{ c.name }}{% if not loop.last %} · {% endif %}{% endfor %}</div></section>{% endif %}

{% for s in sections %}<section id="{{ s.key }}"><h2>{{ s.title }}</h2>
<div class="check">Проверка: {{ s.check }}</div>
{% if s.note %}<div class="note">{{ s.note }}</div>{% endif %}
{% if s.seconds is not none %}<div class="muted">{{ s.seconds }} с{% if s.cached %} · из кеша{% endif %}</div>{% endif %}
{% if s.table %}<table><tr>{% for h in s.table.head %}<th>{{ h }}</th>{% endfor %}</tr>
{% for row in s.table.rows %}<tr>{% for c in row %}<td>{{ c|safe }}</td>{% endfor %}</tr>{% endfor %}</table>{% endif %}
{% if s.images %}<div class="imgs">{% for im in s.images %}<a href="{{ im }}"><img src="{{ im }}" alt="{{ im }}" loading="lazy"></a>{% endfor %}</div>{% endif %}
</section>{% endfor %}

<section><h2>Конфиг</h2><div class="muted">Хеш итоговых файлов — для проверки воспроизводимости: тот же конфиг даёт те же хеши.</div>
{% if hashes %}<table><tr><th>файл</th><th>sha256</th></tr>{% for n, h in hashes %}<tr><td>{{ n }}</td><td><code>{{ h }}</code></td></tr>{% endfor %}</table>{% endif %}
<pre>{{ config }}</pre></section>
</main></body></html>
"""


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _table(key: str, info: dict[str, Any]) -> dict[str, Any] | None:
    if key == "build" and info.get("filters"):
        rows = []
        for f, b in info["filters"].items():
            for a in b.get("astrometry", []):
                rows.append([f, a["экспозиция"], a["звёзд"], a["статус"], a["dx"], a["dy"], a["поворот_град"],
                             a["остаток_до"], a["остаток_после"], a["точность"], a["применено_dx"], a["применено_dy"],
                             "да" if a["исключена"] else ""])
        if rows:
            return {"head": ["фильтр", "экспозиция", "звёзд", "решение", "dx", "dy", "поворот, °", "остаток до, px",
                             "после, px", "точность, px", "применено dx", "dy", "исключена"], "rows": rows}
    if key == "fetch" and info.get("products"):
        return {"head": ["фильтр", "откуда", "файл"],
                "rows": [[f, p["source"], p["path"]] for f, p in info["products"].items()]}
    if key == "inspect" and info.get("filters"):
        cols = [("filter", "фильтр"), ("wavelength_um", "λ, мкм"), ("shape", "размер"),
                ("pixel_scale_arcsec", 'пиксель, "'), ("exptime_s", "экспозиция, с"), ("date_obs", "снят"),
                ("nan_percent", "NaN, %"), ("size_mb", "МБ")]
        return {"head": [h for _, h in cols], "rows": [[r.get(k, "") for k, _ in cols] for r in info["filters"]]}
    if key == "align" and info.get("destripe") and "shifts_px" not in info:
        return None
    if key == "align" and "shifts_px" in info:
        rows = [[f, s["dy"], s["dx"]] for f, s in info["shifts_px"].items()]
        rows.insert(0, [f"{info['reference']} (опорный)", 0, 0])
        return {"head": ["фильтр", "сдвиг dy, px", "сдвиг dx, px"], "rows": rows}
    if key == "stretch" and info.get("stats"):
        return {"head": ["фильтр", "чёрная точка", "белая точка", "выжжено, %", "медиана после"],
                "rows": [[f, f"{s['black_value']:.4g}", f"{s['white_value']:.4g}", s["white_clipped_percent"],
                          s["median_after"]] for f, s in info["stats"].items()]}
    if key == "compose" and info.get("colors"):
        return {"head": ["фильтр", "цвет"],
                "rows": [[f, f'<span class="sw" style="background:{c}"></span>{c}'] for f, c in info["colors"].items()]}
    return None


def _note(key: str, info: dict[str, Any]) -> str | None:
    """Размер масок и сила destripe числом: видно, когда что-то вдруг ведёт себя без причины."""
    if key == "build" and info.get("filters"):
        parts = []
        for f, b in info["filters"].items():
            st = b.get("stars", {})
            fc = b.get("fwhm_check", {})
            parts.append(f"{f}: {b['exposures']} эксп., |Δ строк| {b['rows_before']:.3g} → {b['rows_after']:.3g}, "
                         f"остаток уровней {b.get('level_residual_sigma')}σ, FWHM ×{st.get('fwhm_ratio')}"
                         + (f" (в допуске ±{fc['tol_percent']:g}%: {fc['passed']}/{fc['total']})" if fc else "") + ", "
                         f"поток ×{st.get('flux_ratio')} к i2d"
                         + (f" (к отдельным экспозициям: наша ×{st['flux_ratio_vs_single']}, i2d ×{st['i2d_ratio_vs_single']})"
                            if "flux_ratio_vs_single" in st else ""))
        return " · ".join(parts)
    if key == "baseline":
        if info.get("skipped"):
            return f"Сравнение не построено: {info['skipped']}"
        return (f"С {info['baseline']}: средняя |разность| по туманности {info['nebula_diff_mean']:.4f} "
                f"(99-й процентиль {info['nebula_diff_p99']:.3f}); фактура σ в кольце {info['texture_ring_px'][0]}–"
                f"{info['texture_ring_px'][1]} px: {info['texture_sigma_baseline']:.4f} → {info['texture_sigma']:.4f} "
                f"({info['texture_change_percent']:+.1f}%)")
    if key == "align" and info.get("destripe"):
        parts = [f"{f}: строки {d.get('rows_before', d['before']):.3g} → {d.get('rows_after', d['after']):.3g}"
                 + (f", столбцы {d['cols_before']:.3g} → {d['cols_after']:.3g}" if "cols_before" in d else "")
                 for f, d in info["destripe"].items()]
        return "Destripe, среднее |медиана строки − соседней| по пустому небу: " + " · ".join(parts)
    if key != "clean" or "nebula_percent" not in info:
        return None
    note = (f"Маска туманности: {info['nebula_percent']}% кадра · маска звёзд: {info.get('star_mask_percent', '—')}% кадра "
            f"(звёзды ищутся по {info.get('star_detect_filter', '—')}).")
    if info.get("blocks"):
        parts = [f"{f}: {b['jumps_before']:.3g}σ → {b['jumps_after']:.3g}σ" for f, b in info["blocks"].items() if "jumps_before" in b]
        if parts:
            note += " Средний |скачок| фона на границах областей CON (пустое небо): " + " · ".join(parts) + "."
        parts = [f"{f}: {b['before']:.3g} → {b['after']:.3g} (шум {b['noise']:.3g})" for f, b in info["blocks"].items()]
        note += " Разброс медиан блоков по пустому небу: " + " · ".join(parts)
    return note


def build_report(out_dir: Path) -> Path:
    run = json.loads((out_dir / "run.json").read_text(encoding="utf-8")) if (out_dir / "run.json").exists() else {}
    stages = run.get("stages", {})
    sections = []
    for key, title, check in STAGES:
        if key not in stages:
            continue
        info = stages[key]
        images = [im for im in info.get("previews", []) if (out_dir / im).exists()]
        if key == "inspect":
            images = [info.get("sheet", "inspect_sheet.png")]
        sections.append({"key": key, "title": title, "check": check, "seconds": info.get("seconds"),
                         "cached": info.get("cached"), "images": images, "table": _table(key, info),
                         "note": _note(key, info)})

    finals = stages.get("finish", {}).get("files", {})
    final_names = [finals[k] for k in ("png", "tif") if k in finals and (out_dir / finals[k]).exists()]
    hashes = [(n, _sha256(out_dir / n)) for n in final_names]
    config = (out_dir / "config.yaml").read_text(encoding="utf-8") if (out_dir / "config.yaml").exists() else ""
    hero = "finish_preview.png" if (out_dir / "finish_preview.png").exists() else (
        "compose.png" if (out_dir / "compose.png").exists() else None)

    env = Environment(autoescape=select_autoescape(default=True))
    html = env.from_string(TEMPLATE).render(
        title=f"{run.get('target', out_dir.parent.name)} · skyforge",
        subtitle=f"{run.get('mission', '').upper()} {run.get('instrument', '').upper()} · bin {run.get('bin', '?')} · "
                 + (f"стиль {run['style']} · " if run.get("style") else "")
                 + (f"всего {run['total_seconds']} с · " if run.get("total_seconds") is not None else "")
                 + f"{run.get('started', '')} · {out_dir.name}",
        hero=hero, hero_full=final_names[0] if final_names else hero, finals=final_names,
        reference=next((p.name for p in sorted(out_dir.glob("reference.*"))), None),
        crops=stages.get("finish", {}).get("crops", []),
        sections=sections, hashes=hashes, config=config,
    )
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path


COMPARISON = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<style>
:root{--bg:#0e1014;--panel:#171a21;--line:#2a2f3a;--text:#e6e8ee;--muted:#9aa3b2;--accent:#7fb2ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,Segoe UI,sans-serif}
main{max-width:1600px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}.muted{color:var(--muted)}
.row{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:18px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px}
.card img{width:100%;border-radius:6px;background:#000;display:block}
.card h2{font-size:17px;margin:10px 0 2px}
ul{margin:6px 0 0;padding-left:18px;font-size:13px}code{color:#c9d4ff}a{color:var(--accent)}
table{border-collapse:collapse;font-size:13px;margin-top:6px}td{padding:2px 8px 2px 0;color:var(--muted)}
</style></head><body><main>
<h1>{{ title }}</h1><div class="muted">{{ subtitle }}</div>
<div class="row">
{% if reference %}<div class="card"><a href="{{ reference }}"><img src="{{ reference }}" alt="референс"></a>
<h2>Референс</h2><div class="muted">для сравнения палитры и тона</div></div>{% endif %}
{% for s in styles %}<div class="card"><a href="{{ s.name }}/report.html"><img src="{{ s.name }}/{{ s.preview }}" alt="{{ s.name }}"></a>
<h2><a href="{{ s.name }}/report.html">{{ s.name }}</a></h2>
<table><tr><td>время</td><td>{{ s.total }} с</td></tr><tr><td>из кеша</td><td>{{ s.cached or "—" }}</td></tr>
<tr><td>файлы</td><td>{% for f in s.files %}<a href="{{ s.name }}/{{ f }}">{{ f }}</a> {% endfor %}</td></tr></table>
<div class="muted" style="margin-top:8px">Отличия от base:</div>
{% if s.diff %}<ul>{% for d in s.diff %}<li><code>{{ d }}</code></li>{% endfor %}</ul>{% else %}<div class="muted">нет — это base</div>{% endif %}
</div>{% endfor %}
</div>
{% for row in crop_rows %}<h2 style="margin-top:26px">Кроп {{ row.name }}, 100%</h2><div class="row">
{% for c in row.cells %}<div class="card"><a href="{{ c.src }}"><img src="{{ c.src }}" alt="{{ c.style }}" style="image-rendering:pixelated"></a>
<div class="muted">{{ c.style }}</div></div>{% endfor %}</div>{% endfor %}
{% if baseline %}<h2 style="margin-top:26px">Сравнение с принятым ({{ baseline.name }})</h2>
{% for b in baseline.rows %}<div class="card" style="margin-top:12px"><h2>{{ b.style }}</h2><div class="muted">{{ b.note }}</div>
{% for im in b.images %}<a href="{{ im }}"><img src="{{ im }}" style="margin-top:8px"></a>{% endfor %}</div>{% endfor %}{% endif %}
{% if background_check %}<h2 style="margin-top:26px">Фон по каналам (кроп sky)</h2>
<a href="background_check.png"><img src="background_check.png" style="max-width:100%"></a>{% endif %}
</main></body></html>
"""


def _baseline_rows(top: Path, styles: list[str]) -> dict[str, Any] | None:
    rows, name = [], None
    for s in styles:
        info = json.loads((top / s / "run.json").read_text(encoding="utf-8")).get("stages", {}).get("baseline", {})
        if not info or info.get("skipped"):
            continue
        name = info["baseline"]
        rows.append({"style": s, "note": _note("baseline", info),
                     "images": [f"{s}/{im}" for im in info.get("previews", []) if (top / s / im).exists()]})
    return {"name": name, "rows": rows} if rows else None


def build_comparison(top: Path, styles: list[str], diffs: dict[str, list[str]], reference: str | None) -> Path:
    cards = []
    target = ""
    for name in styles:
        run = json.loads((top / name / "run.json").read_text(encoding="utf-8"))
        target = run.get("target", target)
        st = run.get("stages", {})
        files = st.get("finish", {}).get("files", {})
        cards.append({
            "name": name,
            "preview": "finish_preview.png",
            "total": run.get("total_seconds", "?"),
            "cached": ", ".join(k for k, v in st.items() if v.get("cached")),
            "files": [files[k] for k in ("png", "tif") if k in files],
            "diff": diffs.get(name, []),
        })
    crop_rows: dict[str, list[dict[str, str]]] = {}
    for name in styles:
        run = json.loads((top / name / "run.json").read_text(encoding="utf-8"))
        for c in run.get("stages", {}).get("finish", {}).get("crops", []):
            crop_rows.setdefault(c["name"], []).append({"style": name, "src": f"{name}/{c['file']}"})
    env = Environment(autoescape=select_autoescape(default=True))
    html = env.from_string(COMPARISON).render(
        title=f"{target} · стили", subtitle=f"{len(styles)} стиля из одних данных · {top.name}",
        reference=reference, styles=cards,
        crop_rows=[{"name": k, "cells": v} for k, v in crop_rows.items()],
        background_check=(top / "background_check.png").exists(),
        baseline=_baseline_rows(top, styles))
    path = top / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
