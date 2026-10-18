"""Командная строка skyforge. Каждая команда — один этап конвейера."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import Annotated, Callable, Optional

import typer
from dotenv import load_dotenv
from pydantic import ValidationError

from skyforge import cache as cache_mod
from skyforge.config import NON_PROCESSING, Config, SweepAxis
from skyforge.log import SkyforgeError, console
from skyforge.paths import Paths, find_root
from skyforge.pipeline import fetch_stage, prepare_out, run_all, run_stage, run_styles

app = typer.Typer(help="Конвейер обработки снимков JWST и Hubble: от MAST до цветной картинки.",
                  no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
cache_app = typer.Typer(help="Кеш скачанных и промежуточных данных.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

TargetArg = Annotated[str, typer.Argument(help="Имя объекта (\"NGC 3132\") или путь к YAML-конфигу")]
ConfigArg = Annotated[str, typer.Argument(help="YAML-конфиг: путь или имя из configs/ (ngc3132)")]
BinOpt = Annotated[Optional[int], typer.Option("--bin", min=1, help="Переопределить bin из конфига")]
MissionOpt = Annotated[Optional[str], typer.Option("--mission", "-m", help="jwst или hst")]
InstrOpt = Annotated[Optional[str], typer.Option("--instrument", "-i", help="nircam, miri, wfc3_uvis, acs…")]
FiltersOpt = Annotated[Optional[str], typer.Option("--filters", "-f", help="Через запятую: F090W,F187N")]
StyleOpt = Annotated[Optional[str], typer.Option("--style", "-s", help="Стиль из configs/<объект>/ (measured, showcase…)")]


def _paths() -> Paths:
    root = find_root()
    load_dotenv(root / ".env", override=False)
    return Paths(root)


def _config_path(name: str, paths: Paths) -> Path | None:
    p = Path(name)
    for cand in (p, paths.root / "configs" / p, paths.root / "configs" / f"{name}.yaml"):
        if cand.suffix.lower() in (".yaml", ".yml") and cand.is_file():
            return cand
    return None


def _style_dir(name: str, paths: Paths) -> Path | None:
    for cand in (Path(name), paths.root / "configs" / name):
        if cand.is_dir() and (cand / "base.yaml").is_file():
            return cand
    return None


def list_styles(d: Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.yaml") if p.stem != "base")


def _load(name: str, paths: Paths, bin_: int | None = None, style: str | None = None, **over: object) -> Config:
    d = _style_dir(name, paths)
    if d is not None:
        if not style:
            raise SkyforgeError(f"{name} — папка стилей: укажи --style ({', '.join(list_styles(d))}) или --all")
        path = d / f"{style}.yaml"
        if not path.is_file():
            raise SkyforgeError(f"стиля {style} нет в {d}: есть {', '.join(['base', *list_styles(d)])}")
    else:
        if style:
            raise SkyforgeError(f"'{name}' — плоский конфиг, стилей у него нет")
        path = _config_path(name, paths)
        if path is None:
            raise SkyforgeError(f"конфиг '{name}' не найден (ищу путь, configs/{name}.yaml и configs/{name}/)")
    cfg = Config.load(path, style)
    data = cfg.model_dump(mode="json")
    if bin_ is not None:
        data["bin"] = bin_
    data.update({k: v for k, v in over.items() if v is not None})
    return Config.model_validate(data)


def _target(target: str, paths: Paths, mission: str | None, instrument: str | None, filters: str | None) -> Config:
    fl = [f for f in filters.split(",") if f.strip()] if filters else None
    if _config_path(target, paths):
        return _load(target, paths, mission=mission, instrument=instrument, filters=fl)
    return Config(target=target, mission=mission or "jwst", instrument=instrument or "nircam", filters=fl or [])


def _handle(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except SkyforgeError as e:
            console.print(f"[bold red]Ошибка:[/] {e}")
            raise typer.Exit(1) from None
        except ValidationError as e:
            console.print("[bold red]Ошибка в конфиге:[/]")
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                console.print(f"  {loc}: {err['msg']}")
            raise typer.Exit(2) from None
    return wrapper


def _stage(cfg: Config, paths: Paths, name: str, fn: Callable[[Path], dict]) -> Path:
    out = paths.current(cfg.slug, cfg.style)
    prepare_out(cfg, out)
    run_stage(name, lambda: fn(out), out)
    from skyforge.report import build_report

    build_report(out)
    return out


@app.command()
@_handle
def fetch(target: TargetArg, mission: MissionOpt = None, instrument: InstrOpt = None, filters: FiltersOpt = None,
          list_only: Annotated[bool, typer.Option("--list", help="Только показать, что есть в архиве")] = False,
          offline: Annotated[bool, typer.Option(help="Не ходить в сеть, только кеш и inbox")] = False,
          max_gb: Annotated[float, typer.Option(help="Не качать больше стольких ГБ за раз")] = 8.0) -> None:
    """Найти объект в MAST и скачать кадры уровня 3 в data/raw/ (уже скачанное не качается)."""
    paths = _paths()
    cfg = _target(target, paths, mission, instrument, filters)
    if list_only:
        from skyforge.fetch import fetch as do_fetch

        do_fetch(cfg, paths, list_only=True, offline=offline)
        return
    _stage(cfg, paths, "fetch", lambda out: fetch_stage(cfg, paths, offline=offline, max_gb=max_gb))


@app.command()
@_handle
def inspect(target: TargetArg, mission: MissionOpt = None, instrument: InstrOpt = None,
            filters: FiltersOpt = None) -> None:
    """Таблица фильтров и чёрно-белое превью каждого кадра."""
    from skyforge.inspect import inspect as do

    paths = _paths()
    cfg = _target(target, paths, mission, instrument, filters)
    _stage(cfg, paths, "inspect", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def align(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None,
          force: Annotated[bool, typer.Option(help="Пересчитать, даже если кеш актуален")] = False) -> None:
    """Перевести все фильтры на сетку опорного и обрезать до общего поля."""
    from skyforge.align import align as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "align", lambda out: do(cfg, paths, out, force=force))


@app.command()
@_handle
def clean(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Заполнить насыщенные ядра звёзд и вычесть модель фона (стыки мозаики, градиенты)."""
    from skyforge.clean import clean as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "clean", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def stretch(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Вычесть фон, растянуть яркость (asinh/log/sqrt), нормировать в 0–1."""
    from skyforge.stretch import stretch as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "stretch", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def compose(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Раскрасить фильтры по длине волны и смешать в RGB."""
    from skyforge.compose import compose as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "compose", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def finish(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Денойз, резкость, экспорт 16-битного TIFF и PNG."""
    from skyforge.finish import finish as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "finish", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def run(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None,
        all_styles: Annotated[bool, typer.Option("--all", help="Все стили из configs/<объект>/ подряд")] = False,
        offline: Annotated[bool, typer.Option(help="Не ходить в сеть, только кеш и inbox")] = False,
        force_align: Annotated[bool, typer.Option(help="Пересчитать совмещение заново")] = False) -> None:
    """Весь конвейер: fetch → inspect → align → clean → stretch → compose → finish → отчёт в out/<объект>/<дата-время>/."""
    paths = _paths()
    if not all_styles:
        run_all(_load(config, paths, bin_, style), paths, offline=offline, force_align=force_align)
        return
    d = _style_dir(config, paths)
    if d is None:
        raise SkyforgeError(f"--all работает с папкой стилей configs/{config}/ (base.yaml + стили)")
    styles = list_styles(d)
    if not styles:
        raise SkyforgeError(f"в {d} нет стилей, кроме base.yaml")
    cfgs = [_load(config, paths, bin_, s) for s in styles]
    # Сравниваются собранные конфиги (с умолчаниями): видно «stretch.a: 0.02 → 0.1», а не «— → 0.1».
    skip = set(NON_PROCESSING)
    base = {k: v for k, v in Config.load(d / "base.yaml").model_dump(mode="json").items() if k not in skip}
    diffs = {c.style: style_diff(base, {k: v for k, v in Config.load(d / f"{c.style}.yaml").model_dump(mode="json").items()
                                        if k not in skip}) for c in cfgs}
    run_styles(cfgs, paths, diffs, offline=offline, force_align=force_align)


def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))  # пустой словарь — то же, что отсутствие ключа
        else:
            out[key] = v
    return out


def style_diff(base: dict, style: dict) -> list[str]:
    """Что стиль меняет относительно base, по одной строке на параметр."""
    b, s = _flatten(base), _flatten(style)
    lines = []
    for k in sorted(set(b) | set(s)):
        if b.get(k, "—") != s.get(k, "—"):
            lines.append(f"{k}: {b.get(k, '—')} → {s.get(k, '—')}")
    return lines


def _axis(spec: str | None) -> SweepAxis | None:
    if not spec:
        return None
    if "=" not in spec:
        raise SkyforgeError(f"'{spec}': нужно вида stretch.a=0.01,0.02,0.05")
    param, vals = spec.split("=", 1)
    try:
        return SweepAxis(param=param.strip(), values=[float(v) for v in vals.split(",") if v.strip()])
    except ValueError as e:
        raise SkyforgeError(f"'{spec}': значения должны быть числами") from e


@app.command()
@_handle
def sweep(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None,
          x: Annotated[Optional[str], typer.Option(help="Столбцы: stretch.a=0.005,0.02,0.08")] = None,
          y: Annotated[Optional[str], typer.Option(help="Строки: stretch.white=99.5,99.8,99.95")] = None,
          crop: Annotated[Optional[str], typer.Option(help="Кроп 100% вместо всего кадра: X,Y,СТОРОНА в пикселях сетки")] = None) -> None:
    """Сетка вариантов растяжки/цвета на одном листе (нужен выполненный align)."""
    from skyforge.sweep import sweep as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    xa, ya = _axis(x), _axis(y)
    cr = None
    if crop:
        try:
            cx, cy, side = (int(v) for v in crop.split(","))
        except ValueError as e:
            raise SkyforgeError(f"--crop '{crop}': нужно три целых числа X,Y,СТОРОНА") from e
        cr = (cx, cy, side)
    _stage(cfg, paths, "sweep", lambda out: do(cfg, paths, out, xa, ya, cr))


@app.command()
@_handle
def report(folder: Annotated[Optional[Path], typer.Argument(help="Папка в out/ (по умолчанию самая свежая)")] = None) -> None:
    """Пересобрать HTML-отчёт для папки результатов."""
    from skyforge.report import build_report

    paths = _paths()
    if folder is None:
        runs = sorted(paths.out_root.glob("*/*/run.json"), key=lambda p: p.stat().st_mtime)
        if not runs:
            raise SkyforgeError("в out/ нет ни одного запуска")
        folder = runs[-1].parent
    console.print(f"Отчёт: {build_report(folder)}")


@app.command()
@_handle
def baseline(config: ConfigArg,
             name: Annotated[str, typer.Option(help="Имя принятого результата, обычно тег: phase-3c")],
             run_dir: Annotated[Path, typer.Option("--from", help="Папка прогона (стиля или --all со стилями)")]) -> None:
    """Сохранить итоговые PNG прогона как принятые: с ними сравнивает отчёт (report.baseline)."""
    from skyforge.baseline import save_baseline

    paths = _paths()
    d = _style_dir(config, paths)
    slug = Config.load(d / "base.yaml").slug if d else _load(config, paths).slug
    for p in save_baseline(paths, slug, name, run_dir if run_dir.is_absolute() else paths.root / run_dir):
        console.print(f"  {paths.rel(p)}")


@cache_app.command("list")
@_handle
def cache_list() -> None:
    """Сколько места занимают сырые и промежуточные данные."""
    paths = _paths()
    rows = cache_mod.listing(paths)
    total = 0
    for name, size in rows:
        console.print(f"  {size / 1e9:7.2f} ГБ  {name}")
        total += size
    console.print(f"  {total / 1e9:7.2f} ГБ  всего")


@cache_app.command("prune")
@_handle
def cache_prune(target: Annotated[Optional[str], typer.Argument(help="slug объекта (ngc3132); без него — все")] = None,
                raw: Annotated[bool, typer.Option("--raw", help="Удалить и скачанные FITS этого объекта")] = False,
                yes: Annotated[bool, typer.Option("--yes", "-y", help="Не спрашивать подтверждение")] = False) -> None:
    """Удалить промежуточные данные data/work/ (и по --raw — скачанные файлы объекта)."""
    paths = _paths()
    work = paths.work_root / target if target else paths.work_root
    freed = cache_mod.remove_tree(work)
    console.print(f"data/work: освобождено {freed / 1e9:.2f} ГБ")
    if not raw:
        return
    if not target:
        raise SkyforgeError("--raw удаляет только один объект: укажи его slug")
    import json

    victims = []
    for m in sorted((paths.raw / "manifests").glob(f"{target}_*.json")):
        for prod in json.loads(m.read_text(encoding="utf-8")).get("products", {}).values():
            if prod["source"] == "mast":
                victims.append(paths.abs(prod["path"]).parent)
        victims.append(m)
    victims = sorted(set(victims))
    if not victims:
        console.print(f"Скачанных файлов для '{target}' нет")
        return
    size = sum(cache_mod.dir_size(v) if v.is_dir() else v.stat().st_size for v in victims)
    console.print(f"Будут удалены сырые данные {target}: {size / 1e9:.2f} ГБ")
    if not yes and not typer.confirm("Удалить?"):
        return
    for v in victims:
        if v.is_dir():
            cache_mod.remove_tree(v)
        elif v.exists():
            v.unlink()
    console.print("Удалено.")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    app()


if __name__ == "__main__":
    main()
