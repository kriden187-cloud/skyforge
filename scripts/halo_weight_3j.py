"""3j: карта веса ореола (целиком и кроп sky3), небо ×5 рядом с phase-3i, совпадение с phase-3i там, где вес 0.

Запуск: uv run python scripts/halo_weight_3j.py <папка> <png варианта> "<параметры варианта>" [<png сравнения> <метка>]
По умолчанию сравнение — baseline phase-3i (3j); в 3k — принятое состояние п.3.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import load_nebula_sd, load_sky_mask, load_star_weight
from skyforge.compose import base_color, halo_weight, load_stretched
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from showcase_variants_3i import apply  # noqa: E402


def main() -> None:
    out, png, spec = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    P = Paths(pathlib.Path(".").resolve())
    cfg = apply(Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1}), spec)
    layers, _ = load_stretched(cfg, P)
    field = stats_field(cfg, P, list(layers.values()))
    sd, sky, sm = load_nebula_sd(cfg, P), load_sky_mask(cfg, P), load_star_weight(cfg, P)
    b, _ = base_color(cfg, layers, sky, sd, field=field)
    if sm is not None:
        sw = sm.weight
    else:  # star_saturation 1 (обесцвечивание выкл.) — маска звёзд та же, из clean
        from skyforge.clean import clean_dir

        p_ = clean_dir(cfg, P) / "star_weight.npy"
        sw = np.load(p_) if p_.exists() else None
    hw = halo_weight(np.clip(b, 0, 1), sd, cfg.color.shadows, sw)
    w_neb = np.clip(0.5 + sd / cfg.color.shadows.feather_px, 0, 1)
    lift = np.maximum(w_neb, (1 - w_neb) * hw)[::-1]                        # картинка: строка 0 — верх
    var = np.asarray(Image.open(png).convert("RGB")).astype(np.float32) / 255
    ref_png = sys.argv[4] if len(sys.argv) > 4 else "data/baselines/ngc3132/phase-3i/showcase_bin1.png"
    ref_lab = sys.argv[5] if len(sys.argv) > 5 else "phase-3i"
    base = np.asarray(Image.open(ref_png).convert("RGB")).astype(np.float32) / 255
    crops = {c.name: c for c in with_resolved_crops(cfg, P).report.crops}
    H, W = var.shape[:2]

    def cut(img, c):
        x0, y0 = int(np.clip(c.x - c.size // 2, 0, W - c.size)), int(np.clip(c.y - c.size // 2, 0, H - c.size))
        return img[y0:y0 + c.size, x0:x0 + c.size]

    hw_img = hw[::-1]
    tiles = [shrink(np.stack([hw_img] * 3, -1), 900), cut(np.stack([hw_img] * 3, -1), crops["sky3"])]
    labels = ["вес подъёма за маской (белое — 1), целиком", "вес на кропе sky3"]
    for n in ("sky", "sky2", "sky3", "sky4"):
        c = crops[n]
        tiles += [np.clip(cut(base, c) * 5, 0, 1), np.clip(cut(var, c) * 5, 0, 1)]
        labels += [f"{n} · {ref_lab} · ×5", f"{n} · вариант · ×5"]
        zero = cut(lift, c) <= 1e-6
        d = np.abs(cut(var, c) - cut(base, c)).max(-1) * 255
        dm = (np.median(cut(var, c) @ np.array([0.2126, 0.7152, 0.0722])) - np.median(cut(base, c) @ np.array([0.2126, 0.7152, 0.0722]))) * 255
        print(f"{n}: небо (медиана яркости) {dm:+.2f} ур.; вес 0 у {zero.mean() * 100:.1f}% кропа; там пикселей ≠ {ref_lab}: {((d[zero] > 0.5).mean() * 100 if zero.any() else 0):.2f}%, "
              f"макс {(d[zero].max() if zero.any() else 0):.0f}; вес > 0,5 у {(cut(hw_img, c) > 0.5).mean() * 100:.1f}%")
    tiles.append(cut(var, crops["sky3"]))
    labels.append("sky3 · вариант · 100%")
    contact_sheet([t[::-1] for t in tiles], labels, out / "07_вес_ореола_и_небо.png", cols=2, tile_inches=4.2,
                  interpolation="nearest", title=f"Вес ореола и небо ×5: {spec}")


if __name__ == "__main__":
    main()
