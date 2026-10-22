"""Сверка одного объекта короткого списка discover с обработками в сети на сегодня (фаза 7: OMC3-N перед публикацией).

Те же запросы и правила отбора, что у `skyforge untouched` (skyforge.untouched: build_queries, sweep, кеш), но для одного
объекта и в свою папку out/discover/<сегодня>/<slug>/: Commons и Flickr — сценарием, поиск в сети — импорт ответов
поиска агента (JSON, как --web-results). Пишет hits.json — находки к разбору (вердикт — после чтения страницы).

Запуск: uv run python scripts/untouched_one.py <дата discover> <цель MAST> <папка> [ответы поиска в сети.json]
"""

from __future__ import annotations

import json
import pathlib
import sys

from skyforge.discover import Candidate
from skyforge.untouched import Cache, import_web, load_names, names_for, sweep


def main() -> None:
    day, target, out = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
    cands = json.loads((pathlib.Path("out/discover") / day / "candidates.json").read_text(encoding="utf-8"))
    c = next(Candidate(**x) for x in cands if x["target"] == target)
    cache = Cache(out / "cache", offline=False)
    if len(sys.argv) > 4:
        print("импортировано ответов поиска в сети:", import_web(pathlib.Path(sys.argv[4]), cache))
    ck = sweep(c, names_for(c, load_names(pathlib.Path("configs/discover_names.yaml"))), cache)
    for q in ck.queries:
        print(f"{q['source']:8} {q['status']:>8} всего {q['total']:>5}, к отбору {q['n']:>3} | {q['query']}")
    (out / "hits.json").write_text(json.dumps(ck.hits, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"находок с именем объекта или номером программы: {len(ck.hits)}")
    for h in ck.hits:
        print(f"  [{h['class']}] {h['platform']} {h.get('author', '')} {h.get('date', '')} {h['url']} — {h['title'][:80]}")


if __name__ == "__main__":
    main()
