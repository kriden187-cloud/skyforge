"""Конвейер целиком на синтетическом небе: совмещение по звёздам, воспроизводимость, CLI."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS
from typer.testing import CliRunner

from skyforge.align import align, load_aligned
from skyforge.fetch import local_products
from skyforge.pipeline import run_all

from conftest import marker_radec, star_field


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_inbox_reads_filter_from_header(sky, cfg):
    prods = local_products(cfg, sky)
    assert list(prods) == ["F090W", "F200W", "F470N"]  # по длине волны; F470N из PUPIL
    assert all(p.source == "inbox" for p in prods.values())


@pytest.mark.parametrize("north_up,bin_", [(True, 1), (False, 1), (True, 2)])
def test_align_puts_stars_where_wcs_says(sky, cfg, tmp_path, north_up, bin_):
    c = cfg.model_copy(update={"bin": bin_, "finish": cfg.finish.model_copy(update={"north_up": north_up})})
    info = align(c, sky, tmp_path / "out")
    arrays, wcs, meta = load_aligned(c, sky)
    assert set(arrays) == {"F090W", "F200W", "F470N"}
    assert all(np.isfinite(a).mean() > 0.97 for a in arrays.values())

    checked = 0
    brightest = max(s[2] for s in star_field())
    for ra, dec, flux in star_field():
        if flux == brightest:
            continue  # у неё выбито «насыщенное» ядро, центроид дыры не показатель
        mra, mdec = marker_radec()
        if np.hypot((ra - mra) * np.cos(np.radians(dec)), dec - mdec) * 3600 < 1.0:
            continue  # крылья метки ориентации тянут центроид
        x, y = wcs.world_to_pixel_values(ra, dec)
        xi, yi = int(round(float(x))), int(round(float(y)))
        r = 4
        h, w = meta["shape"]
        if not (r <= xi < w - r and r <= yi < h - r):
            continue
        for f, a in arrays.items():
            win = np.nan_to_num(np.asarray(a[yi - r:yi + r + 1, xi - r:xi + r + 1], dtype=np.float64))
            win = np.clip(win - np.median(win), 0, None)
            yy, xx = np.indices(win.shape)
            cy = (yy * win).sum() / win.sum() + yi - r
            cx = (xx * win).sum() / win.sum() + xi - r
            assert abs(cy - y) < 0.35 and abs(cx - x) < 0.35, f"{f}: звезда смещена на ({cy - y:.2f}, {cx - x:.2f})"
        checked += 1
    assert checked >= 5
    assert all(abs(s["dx"]) < 0.5 and abs(s["dy"]) < 0.5 for s in info["shifts_px"].values())
    if north_up:
        # север вверх, восток влево: RA убывает вправо, Dec растёт вверх
        ra0, dec0 = wcs.pixel_to_world_values(10, 10)
        ra1, _ = wcs.pixel_to_world_values(20, 10)
        _, dec1 = wcs.pixel_to_world_values(10, 20)
        assert ra1 < ra0 and dec1 > dec0


def test_align_cache_is_reused(sky, cfg, tmp_path):
    assert align(cfg, sky, tmp_path / "o")["cached"] is False
    assert align(cfg, sky, tmp_path / "o")["cached"] is True
    c2 = cfg.model_copy(update={"align": cfg.align.model_copy(update={"order": "bicubic"})})
    assert align(c2, sky, tmp_path / "o")["cached"] is False


def test_full_run_is_reproducible_and_raw_untouched(sky, cfg, tmp_path):
    raw_before = {p: sha(p) for p in sky.inbox.rglob("*.fits")}
    a = run_all(cfg, sky, offline=True, out_dir=tmp_path / "run1")
    b = run_all(cfg, sky, offline=True, force_align=True, out_dir=tmp_path / "run2")
    for name in ("testobj.tif", "testobj.png"):
        assert sha(a / name) == sha(b / name), f"{name} отличается между прогонами"
    assert {p: sha(p) for p in sky.inbox.rglob("*.fits")} == raw_before

    for img in ("inspect_sheet.png", "align_check.png", "align_zoom.png", "background_model.png", "star_cores.png",
                "stretch_hist.png",
                "compose.png", "finish_zoom.png", "report.html", "config.yaml"):
        assert (a / img).exists(), img
    run = json.loads((a / "run.json").read_text(encoding="utf-8"))
    assert set(run["stages"]) == {"fetch", "inspect", "align", "clean", "stretch", "compose", "finish"}
    cl = run["stages"]["clean"]["filters"]
    assert cl["F090W"]["cores"] >= 1 and cl["F090W"]["tiles_measured"] == 2

    import tifffile

    t = tifffile.imread(a / "testobj.tif")
    assert t.dtype == np.uint16 and t.ndim == 3 and t.shape[2] == 3
    assert t.max() > 50000  # звёзды доходят до белого


def test_tile_step_removed_in_pipeline(sky, cfg, tmp_path):
    """Ступенька пьедестала между плитками CON уходит на этапе clean."""
    from conftest import NOISE, TILE_STEP
    from skyforge.clean import clean, load_input

    align(cfg, sky, tmp_path / "o")
    raw, _, _ = load_aligned(cfg, sky)
    clean(cfg, sky, tmp_path / "o")
    flat, _ = load_input(cfg, sky)
    from skyforge.align import load_tiles

    def step(arrs, f):
        a, t = np.asarray(arrs[f]), load_tiles(cfg, sky, f)
        sky_px = np.isfinite(a) & (a < np.nanmedian(a) + 2 * NOISE)
        ids = sorted(set(np.unique(t[t >= 0])))
        return np.median(a[sky_px & (t == ids[-1])]) - np.median(a[sky_px & (t == ids[0])])

    for f in ("F090W", "F200W"):
        assert step(raw, f) == pytest.approx(TILE_STEP, abs=0.3)
        assert abs(step(flat, f)) < 0.2 * TILE_STEP


def test_changing_one_number_changes_result(sky, cfg, tmp_path):
    a = run_all(cfg, sky, offline=True, out_dir=tmp_path / "r1")
    c2 = cfg.with_param("stretch.a", 0.2)
    b = run_all(c2, sky, offline=True, out_dir=tmp_path / "r2")
    assert sha(a / "testobj.png") != sha(b / "testobj.png")


def test_cli_stages_and_sweep(sky, cfg, monkeypatch):
    from skyforge.cli import app

    (sky.root / "configs" / "testobj.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    monkeypatch.setenv("SKYFORGE_HOME", str(sky.root))
    runner = CliRunner()
    for cmd in (["fetch", "testobj", "--offline"], ["inspect", "testobj"], ["align", "testobj"], ["clean", "testobj"],
                ["stretch", "testobj"], ["compose", "testobj"], ["finish", "testobj"],
                ["sweep", "testobj", "--x", "stretch.a=0.01,0.1", "--y", "color.saturation=0.8,1.4"],
                ["sweep", "testobj", "--x", "color.weights.F200W=1,0.5,0.2"]):
        res = runner.invoke(app, cmd)
        assert res.exit_code == 0, f"{cmd}: {res.output}"
    cur = sky.current("testobj")
    assert (cur / "sweep.png").exists() and (cur / "testobj.png").exists() and (cur / "report.html").exists()

    # Устаревшая растяжка не должна молча попасть в цвет.
    yaml_text = (sky.root / "configs" / "testobj.yaml").read_text(encoding="utf-8").replace("a: 0.05", "a: 0.07")
    (sky.root / "configs" / "testobj.yaml").write_text(yaml_text, encoding="utf-8")
    res = runner.invoke(app, ["compose", "testobj"])
    assert res.exit_code == 1 and "stretch" in res.output


def test_cli_reports_config_errors(sky, monkeypatch):
    from skyforge.cli import app

    (sky.root / "configs" / "bad.yaml").write_text("target: X\nstretch:\n  blak: 1\n", encoding="utf-8")
    monkeypatch.setenv("SKYFORGE_HOME", str(sky.root))
    res = CliRunner().invoke(app, ["align", "bad"])
    assert res.exit_code == 2 and "stretch.blak" in res.output


@pytest.mark.parametrize("north_up", [True, False])
def test_png_orientation_matches_sky(sky, cfg, tmp_path, north_up):
    """Яркая метка на юго-востоке неба должна лечь в PNG внизу слева (север вверху, восток слева).
    Туманность почти симметрична, поэтому зеркальную картинку не ловили ни тесты, ни приёмка."""
    from PIL import Image
    from scipy import ndimage

    c = cfg.model_copy(update={"finish": cfg.finish.model_copy(update={"north_up": north_up})})
    out = run_all(c, sky, offline=True, out_dir=tmp_path / "o")
    png = np.asarray(Image.open(out / "testobj.png"), dtype=np.float32).mean(axis=-1)
    h, w = png.shape
    y, x = np.unravel_index(np.argmax(ndimage.gaussian_filter(png, 6)), png.shape)
    assert y > 0.6 * h and x < 0.4 * w, f"метка в ({x}, {y}) на картинке {w}×{h}, а должна быть внизу слева"
