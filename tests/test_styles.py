"""Фаза 3c: стили обработки — наследование конфигов, поканальная растяжка, тени по маске, кеш, отчёт."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from skyforge.config import Config, deep_merge, load_raw


def sha(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# --- наследование ---

def test_deep_merge_keeps_untouched_keys():
    base = {"stretch": {"a": 0.1, "white": 99.8}, "color": {"overrides": {"F090W": "#0000ff"}}, "filters": ["A", "B"]}
    out = deep_merge(base, {"stretch": {"a": 0.05}, "color": {"overrides": {"F356W": "#ff9a20"}}, "filters": ["C"]})
    assert out["stretch"] == {"a": 0.05, "white": 99.8}
    assert out["color"]["overrides"] == {"F090W": "#0000ff", "F356W": "#ff9a20"}
    assert out["filters"] == ["C"]  # списки заменяются целиком
    assert base["stretch"]["a"] == 0.1  # исходник не испорчен


def test_extends_chain_and_cycle(tmp_path):
    (tmp_path / "base.yaml").write_text("target: X\nstretch: {a: 0.1, white: 99.8}\n", encoding="utf-8")
    (tmp_path / "mid.yaml").write_text("extends: base\nstretch: {a: 0.05}\n", encoding="utf-8")
    (tmp_path / "top.yaml").write_text("extends: mid\nstretch: {white: 99.9}\n", encoding="utf-8")
    raw = load_raw(tmp_path / "top.yaml")
    assert raw["stretch"] == {"a": 0.05, "white": 99.9} and raw["target"] == "X" and "extends" not in raw
    (tmp_path / "loop1.yaml").write_text("extends: loop2\n", encoding="utf-8")
    (tmp_path / "loop2.yaml").write_text("extends: loop1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="по кругу"):
        load_raw(tmp_path / "loop1.yaml")


def test_style_name_and_reference_do_not_change_digest():
    a = Config.model_validate({"target": "X"})
    b = Config.model_validate({"target": "X", "style": "measured", "reference_image": "ref.jpg"})
    assert a.digest() == b.digest()
    assert a.digest() != Config.model_validate({"target": "X", "stretch": {"a": 0.2}}).digest()


# --- поканальная растяжка ---

def test_stretch_default_section_and_per_filter():
    c = Config.model_validate({"target": "X", "stretch": {"default": {"white": 99.5, "a": 0.2},
                                                          "per_filter": {"F470N": {"a": 0.05}}}})
    assert c.stretch.white == 99.5 and c.stretch.a == 0.2
    assert c.stretch.for_filter("F470N").a == 0.05 and c.stretch.for_filter("F470N").white == 99.5
    with pytest.raises(Exception, match="и в default"):
        Config.model_validate({"target": "X", "stretch": {"a": 0.1, "default": {"a": 0.2}}})


def test_per_filter_changes_only_that_filter():
    from skyforge.stretch import stretch_all

    rng = np.random.default_rng(2)
    arrays = {f: rng.gamma(2, 1, (80, 80)).astype(np.float32) for f in ("F090W", "F200W", "F470N")}
    plain = Config.model_validate({"target": "X"})
    tuned = Config.model_validate({"target": "X", "stretch": {"per_filter": {"F200W": {"white": 99.0, "a": 0.03}}}})
    a, _ = stretch_all(plain, arrays)
    b, _ = stretch_all(tuned, arrays)
    assert not np.array_equal(a["F200W"], b["F200W"])
    assert np.array_equal(a["F090W"], b["F090W"]) and np.array_equal(a["F470N"], b["F470N"])


# --- тени по маске ---

def test_shadows_nebula_scope_keeps_background():
    from skyforge.compose import apply_shadows
    from skyforge.config import ShadowsConfig

    yy, xx = np.indices((100, 100))
    sd = (30 - np.hypot(yy - 50, xx - 50)).astype(np.float32)  # круг радиуса 30: >0 внутри
    rgb = np.full((100, 100, 3), 0.1, dtype=np.float32)
    lifted_all = apply_shadows(rgb, ShadowsConfig(value=4, scope="all"), None)
    neb = apply_shadows(rgb, ShadowsConfig(value=4, scope="nebula", feather_px=20), sd)
    assert np.allclose(neb[50, 50], lifted_all[50, 50])   # в глубине — как при scope all
    assert np.allclose(neb[2, 2], rgb[2, 2])              # фон далеко снаружи не тронут
    profile = neb[50, 50:100, 0]
    assert np.all(np.diff(profile) <= 1e-6)               # к краю спадает плавно, без ступенек
    assert np.max(np.abs(np.diff(profile))) < 0.2 * (lifted_all[50, 50, 0] - 0.1)
    with pytest.raises(Exception, match="маски туманности"):
        apply_shadows(rgb, ShadowsConfig(value=4, scope="nebula"), None)


def test_shadows_number_shorthand():
    c = Config.model_validate({"target": "X", "color": {"shadows": 4}})
    assert c.color.shadows.value == 4 and c.color.shadows.scope == "all"


# --- ориентация ---

def test_export_puts_sky_north_of_grid_on_top(tmp_path):
    """Нулевая строка сетки (как в FITS) — низ картинки."""
    from PIL import Image

    from skyforge.finish import export

    rgb = np.zeros((20, 30, 3), dtype=np.float32)
    rgb[0] = 1.0  # нижняя строка сетки
    files = export(rgb, tmp_path / "x", {"title": "t", "credit": "c", "software": "s"})
    png = np.asarray(Image.open(tmp_path / files["png"]))
    assert png[-1].min() == 255 and png[0].max() == 0


# --- стили целиком: перенос, кеш, отчёт ---

STYLE_A = {"stretch": {"method": "asinh", "black": 1.0, "white": 99.9, "a": 0.05},
           "color": {"saturation": 1.1, "shadows": 3},
           "finish": {"denoise": 0.3, "sharpen": 0.5}}
STYLE_B = {"stretch": {"per_filter": {"F470N": {"a": 0.02}}},
           "color": {"saturation": 1.5, "shadows": {"value": 3, "scope": "nebula", "feather_px": 6}}}


def _write_styles(sky, cfg):
    data = cfg.model_dump(mode="json", exclude={"style", "reference_image"})
    for k in ("stretch", "color", "finish"):
        data.pop(k)
    data["finish"] = {"north_up": True}
    # Синтетический кадр всего ~8″: «звёзды» и «туманность» различаются на мелком масштабе.
    data["stars"].update(max_star_area_arcsec2=0.3, halo_arcsec=0.1)  # уже так в фикстуре; явно — для читаемости
    d = sky.root / "configs" / "testobj"
    d.mkdir()
    (d / "base.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    (d / "alpha.yaml").write_text(yaml.safe_dump({"extends": "base", **STYLE_A}), encoding="utf-8")
    (d / "beta.yaml").write_text(yaml.safe_dump({"extends": "alpha", **STYLE_B}), encoding="utf-8")
    flat = deep_merge(data, STYLE_A)
    (sky.root / "configs" / "flat.yaml").write_text(yaml.safe_dump(flat, allow_unicode=True), encoding="utf-8")


def test_styles_all_reuse_cache_and_transfer_keeps_file(sky, cfg, monkeypatch):
    from skyforge.cli import app

    _write_styles(sky, cfg)
    monkeypatch.setenv("SKYFORGE_HOME", str(sky.root))
    runner = CliRunner()

    res = runner.invoke(app, ["run", "flat", "--offline"])
    assert res.exit_code == 0, res.output
    flat_png = next(sky.out_root.glob("testobj/*/testobj.png"))

    res = runner.invoke(app, ["run", "testobj", "--all", "--offline"])
    assert res.exit_code == 0, res.output
    top = next(p for p in (sky.out_root / "testobj").iterdir() if p.name.endswith("_styles"))
    assert (top / "report.html").exists()
    alpha, beta = top / "alpha", top / "beta"
    # Перенос плоского конфига в base + стиль не меняет файл.
    assert sha(alpha / "testobj_alpha.png") == sha(flat_png)
    assert sha(alpha / "testobj_alpha.png") != sha(beta / "testobj_beta.png")
    # Второй стиль берёт тяжёлые этапы из кеша.
    run_b = json.loads((beta / "run.json").read_text(encoding="utf-8"))
    assert all(run_b["stages"][s].get("cached") for s in ("inspect", "align", "clean"))
    assert run_b["total_seconds"] > 0 and (beta / "shadows_scope.png").exists()
    assert "Маска туманности:" in (beta / "report.html").read_text(encoding="utf-8")
    html = (top / "report.html").read_text(encoding="utf-8")
    assert "alpha/finish_preview.png" in html and "beta/finish_preview.png" in html
    assert "color.saturation: 1.0 → 1.5" in html  # отличия от base (с умолчаниями) у beta, унаследованные через alpha
    assert "stretch.method" not in html            # совпадает с умолчанием — не показывается

    # Тот же стиль ещё раз — тот же файл.
    res = runner.invoke(app, ["run", "testobj", "--style", "beta", "--offline"])
    assert res.exit_code == 0, res.output
    again = next(p for p in (sky.out_root / "testobj").iterdir() if p.name.endswith("_beta"))
    assert sha(again / "testobj_beta.png") == sha(beta / "testobj_beta.png")


def test_style_required_for_style_folder(sky, cfg, monkeypatch):
    from skyforge.cli import app

    _write_styles(sky, cfg)
    monkeypatch.setenv("SKYFORGE_HOME", str(sky.root))
    res = CliRunner().invoke(app, ["align", "testobj"])
    assert res.exit_code == 1 and "--style" in res.output and "alpha" in res.output
