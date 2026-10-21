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

app = typer.Typer(help="Pipeline for JWST and Hubble images: from the MAST archive to a colour picture.",
                  no_args_is_help=True, add_completion=False, pretty_exceptions_show_locals=False)
cache_app = typer.Typer(help="Cache of downloaded and intermediate data.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")

TargetArg = Annotated[str, typer.Argument(help="Object name (\"NGC 3132\") or path to a YAML config")]
ConfigArg = Annotated[str, typer.Argument(help="YAML config: a path or a name from configs/ (ngc3132)")]
BinOpt = Annotated[Optional[int], typer.Option("--bin", min=1, help="Override bin from the config")]
MissionOpt = Annotated[Optional[str], typer.Option("--mission", "-m", help="jwst or hst")]
InstrOpt = Annotated[Optional[str], typer.Option("--instrument", "-i", help="nircam, miri, wfc3_uvis, acs…")]
FiltersOpt = Annotated[Optional[str], typer.Option("--filters", "-f", help="Comma-separated: F090W,F187N")]
StyleOpt = Annotated[Optional[str], typer.Option("--style", "-s", help="Style from configs/<object>/ (measured, showcase…)")]


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
          list_only: Annotated[bool, typer.Option("--list", help="Only show what the archive has")] = False,
          offline: Annotated[bool, typer.Option(help="No network: cache and inbox only")] = False,
          max_gb: Annotated[float, typer.Option(help="Do not download more than this many GB at once")] = 8.0) -> None:
    """Find the object in MAST and download level-3 frames to data/raw/ (files already on disk are skipped)."""
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
    """Filter table and a greyscale preview of each frame."""
    from skyforge.inspect import inspect as do

    paths = _paths()
    cfg = _target(target, paths, mission, instrument, filters)
    _stage(cfg, paths, "inspect", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def build(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """source: level2 — own mosaic per filter from the _cal.fits exposures, on the i2d grid."""
    from skyforge.level2 import build as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "build", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def align(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None,
          force: Annotated[bool, typer.Option(help="Recompute even if the cache is up to date")] = False) -> None:
    """Reproject all filters onto the reference grid and crop to the common field."""
    from skyforge.align import align as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "align", lambda out: do(cfg, paths, out, force=force))


@app.command()
@_handle
def clean(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Fill saturated star cores and subtract the background model (mosaic seams, gradients)."""
    from skyforge.clean import clean as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "clean", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def stretch(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Subtract the background, stretch the brightness (asinh/log/sqrt), normalise to 0–1."""
    from skyforge.stretch import stretch as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "stretch", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def compose(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Colour the filters by wavelength and blend them into RGB."""
    from skyforge.compose import compose as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "compose", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def finish(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None) -> None:
    """Denoise, sharpen, export a 16-bit TIFF and a PNG."""
    from skyforge.finish import finish as do

    paths = _paths()
    cfg = _load(config, paths, bin_, style)
    _stage(cfg, paths, "finish", lambda out: do(cfg, paths, out))


@app.command()
@_handle
def run(config: ConfigArg, bin_: BinOpt = None, style: StyleOpt = None,
        all_styles: Annotated[bool, typer.Option("--all", help="All styles from configs/<object>/ in turn")] = False,
        offline: Annotated[bool, typer.Option(help="No network: cache and inbox only")] = False,
        force_align: Annotated[bool, typer.Option(help="Recompute the alignment from scratch")] = False) -> None:
    """Whole pipeline: fetch → inspect → align → clean → stretch → compose → finish → report in out/<object>/<date-time>/."""
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
          x: Annotated[Optional[str], typer.Option(help="Columns: stretch.a=0.005,0.02,0.08")] = None,
          y: Annotated[Optional[str], typer.Option(help="Rows: stretch.white=99.5,99.8,99.95")] = None,
          crop: Annotated[Optional[str], typer.Option(help="100% crop instead of the whole frame: X,Y,SIDE in grid pixels")] = None) -> None:
    """Grid of stretch/colour variants on one sheet (needs a completed align)."""
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
def report(folder: Annotated[Optional[Path], typer.Argument(help="Folder in out/ (default: the newest)")] = None) -> None:
    """Rebuild the HTML report for a results folder."""
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
def discover(config: Annotated[Optional[Path], typer.Option("--config", help="Search settings (configs/discover.yaml)")] = None,
             refresh_press: Annotated[bool, typer.Option("--refresh-press",
                                                         help="Rebuild the press-release gallery (network)")] = False,
             since: Annotated[Optional[str], typer.Option(help="Exclusive access ended no earlier than (YYYY-MM-DD)")] = None,
             top: Annotated[Optional[int], typer.Option(help="How many candidates on the sheet")] = None) -> None:
    """Find recently released JWST observations without a press release in MAST: table, candidates.md, preview sheet."""
    from datetime import date

    from skyforge.discover import load_config, run as run_discover

    paths = _paths()
    cfg = load_config(config or paths.root / "configs" / "discover.yaml")
    upd = {k: v for k, v in (("since", date.fromisoformat(since) if since else None), ("top", top)) if v is not None}
    if upd:
        cfg = cfg.model_copy(update=upd)
    run_discover(paths.root, cfg, date.today(), refresh_press=refresh_press)


@app.command()
@_handle
def untouched(day: Annotated[Optional[str], typer.Option("--date", help="discover run to check, out/discover/<YYYY-MM-DD>/ (default: the newest)")] = None,
              shortlist: Annotated[Optional[int], typer.Option(help="How many candidates from the top of the discover list (default: its shortlist)")] = None,
              offline: Annotated[bool, typer.Option(help="No network: answers cached for this date only")] = False,
              web_results: Annotated[Optional[Path], typer.Option(help="JSON with web-search answers to import into the cache")] = None,
              recon: Annotated[Optional[str], typer.Option(help="Comma-separated MAST target names: scouting sheets and recon.md instead of the check")] = None) -> None:
    """Search the web, Wikimedia Commons and Flickr for public processings of the discover shortlist: candidates_untouched.md and .png."""
    from skyforge.untouched import run as run_untouched

    paths = _paths()
    base = paths.root / "out" / "discover"
    if day is None:
        runs = sorted(p.name for p in base.glob("????-??-??") if (p / "candidates.json").exists()) if base.exists() else []
        if not runs:
            raise SkyforgeError("в out/discover/ нет прогонов discover — сначала skyforge discover")
        day = runs[-1]
    if recon:
        from skyforge.untouched import recon as run_recon

        console.print(f"  [green]{run_recon(paths.root, day, [t.strip() for t in recon.split(',') if t.strip()])}[/]")
        return
    out = run_untouched(paths.root, day, shortlist=shortlist, offline=offline, web_results=web_results)
    console.print(f"  [green]{out / 'candidates_untouched.md'}[/]")


@app.command()
@_handle
def baseline(config: ConfigArg,
             name: Annotated[str, typer.Option(help="Name of the accepted result, usually a tag: phase-3c")],
             run_dir: Annotated[Path, typer.Option("--from", help="Run folder (a style folder, or an --all folder with styles)")]) -> None:
    """Save a run's final PNGs as accepted: the report compares against them (report.baseline)."""
    from skyforge.baseline import save_baseline

    paths = _paths()
    d = _style_dir(config, paths)
    slug = Config.load(d / "base.yaml").slug if d else _load(config, paths).slug
    for p in save_baseline(paths, slug, name, run_dir if run_dir.is_absolute() else paths.root / run_dir):
        console.print(f"  {paths.rel(p)}")


@app.command()
@_handle
def publish(target: Annotated[str, typer.Argument(help="Object: folder configs/<object>/ and its section under objects in configs/publish.yaml")],
            style: StyleOpt = None) -> None:
    """Publish the accepted result (report.baseline): master PNG and TIFF, JPEG versions for platforms, RU/EN caption, texts for Wikimedia Commons and Flickr."""
    from datetime import date

    from skyforge.publish import load_publish, publish_object

    paths = _paths()
    pub = load_publish(paths.root / "configs" / "publish.yaml")
    for p in publish_object(paths, pub, target, style, date.today().isoformat()):
        console.print(f"  {paths.rel(p)}")


@app.command()
@_handle
def gallery() -> None:
    """Config gallery: docs/GALLERY.md (English), docs/GALLERY.ru.md and previews in docs/gallery/ — one section per object and style from configs/publish.yaml."""
    from skyforge.publish import build_gallery, load_publish

    paths = _paths()
    console.print(f"  {paths.rel(build_gallery(paths, load_publish(paths.root / 'configs' / 'publish.yaml')))}")


@cache_app.command("list")
@_handle
def cache_list() -> None:
    """How much disk space raw and intermediate data take."""
    paths = _paths()
    rows = cache_mod.listing(paths)
    total = 0
    for name, size in rows:
        console.print(f"  {size / 1e9:7.2f} ГБ  {name}")
        total += size
    console.print(f"  {total / 1e9:7.2f} ГБ  всего")


@cache_app.command("prune")
@_handle
def cache_prune(target: Annotated[Optional[str], typer.Argument(help="Object slug (ngc3132); without it — all objects")] = None,
                raw: Annotated[bool, typer.Option("--raw", help="Also delete the downloaded FITS of this object")] = False,
                yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not ask for confirmation")] = False) -> None:
    """Delete intermediate data in data/work/ (and, with --raw, the object's downloaded files)."""
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
