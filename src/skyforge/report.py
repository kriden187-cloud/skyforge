"""HTML-отчёт: все стадии, параметры и копия конфига на одной странице. Картинки — рядом в той же папке."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, select_autoescape

STAGES = [
    ("fetch", "1. Данные", "Какие файлы взяты и откуда."),
    ("inspect", "2. Сырые кадры", "На каждом превью узнаётся объект, пустых и битых кадров нет."),
    ("align", "3. Совмещение", "Звёзды на границах клеток шахматки не смещены, на наложении нет цветных ореолов."),
    ("clean", "3½. Очистка", "Стыки плиток и градиенты ушли в модель фона, туманность осталась; ядро звезды белое."),
    ("stretch", "4. Растяжка", "Пик фона прижат к нулю, хвост гистограммы тянется вправо, детали не выжжены."),
    ("compose", "5. Цвет", "Фон нейтральный, без зелёного или фиолетового оттенка; структура как в пресс-релизе."),
    ("finish", "6. Финиш", "На кропе 100% нет «пластилина» от денойза и светлых ореолов от резкости."),
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
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;white-space:nowrap}th{color:var(--muted);font-weight:500}
.sw{display:inline-block;width:14px;height:14px;border-radius:3px;vertical-align:-2px;margin-right:6px}
pre{background:#0a0c10;border:1px solid var(--line);border-radius:6px;padding:12px;overflow:auto;font-size:12.5px}
a{color:var(--accent)}code{color:#c9d4ff}
</style></head><body><main>
<h1>{{ title }}</h1>
<div class="muted">{{ subtitle }}</div>
{% if hero %}<section class="hero"><a href="{{ hero_full }}"><img src="{{ hero }}" alt="итог"></a>
<div class="muted">Полный размер: {% for f in finals %}<a href="{{ f }}">{{ f }}</a>{% if not loop.last %} · {% endif %}{% endfor %}</div></section>{% endif %}

{% for s in sections %}<section id="{{ s.key }}"><h2>{{ s.title }}</h2>
<div class="check">Проверка: {{ s.check }}</div>
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
    if key == "fetch" and info.get("products"):
        return {"head": ["фильтр", "откуда", "файл"],
                "rows": [[f, p["source"], p["path"]] for f, p in info["products"].items()]}
    if key == "inspect" and info.get("filters"):
        cols = [("filter", "фильтр"), ("wavelength_um", "λ, мкм"), ("shape", "размер"),
                ("pixel_scale_arcsec", 'пиксель, "'), ("exptime_s", "экспозиция, с"), ("date_obs", "снят"),
                ("nan_percent", "NaN, %"), ("size_mb", "МБ")]
        return {"head": [h for _, h in cols], "rows": [[r.get(k, "") for k, _ in cols] for r in info["filters"]]}
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
                         "cached": info.get("cached"), "images": images, "table": _table(key, info)})

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
                 f"{run.get('started', '')} · {out_dir.name}",
        hero=hero, hero_full=final_names[0] if final_names else hero, finals=final_names,
        sections=sections, hashes=hashes, config=config,
    )
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
