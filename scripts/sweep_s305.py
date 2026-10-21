"""Фаза 4, Sh2-305: перебор растяжки и цвета showcase — целиком и кропы (волокна, пузырь, глобула, звезда при 100%;
небо и стык ×5). Референса нет: таблица по внутренним критериям — небо R/G/B и доля нулей на пустом небе, выжжено
в туманности (канал ≥ 254), зелень звёзд (медиана G − max(R, B) в ядрах звёзд вне туманности, уровни).

Запуск: uv run python scripts/sweep_s305.py <папка> <имя> <вариант> [<вариант> ...]
Вариант — как в showcase_variants_3i.py ("stretch.white=99.9", несколько через ";"), "" — как в YAML.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask
from skyforge.config import Config
from skyforge.finish import finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
CROPS = [("filaments", 1), ("bubble", 1), ("globule", 1), ("star", 1), ("sky", 5), ("seam", 5)]


def main() -> None:
    out, name, specs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    (out / name).mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    arrays, _ = load_input(base, P)
    load_aligned(base, P)
    sky, sd = load_sky_mask(base, P), load_nebula_sd(base, P)
    ctx = {"field": stats_field(base, P, list(arrays.values())), "sky": sky, "sd": sd, "fin": finish_masks(base, P)}
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    sky_img, neb_img = (sky & full)[::-1], (sd > 0)[::-1]
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    stars_img = ((sw > 0.5) & (sd < 0))[::-1]
    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}

    def cut(img, c):
        s = max(16, c.size // base.bin)
        cx, cy = int(round(c.x / base.bin)), int(round(c.y / base.bin))
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    results, rows = [], ["вариант\tнебо R / G / B, уровни\tв нуле на небе R / G / B, %\tвыжжено в туманности, %"
                         "\tзелень звёзд (G − max(R, B)), уровни"]
    for spec in specs:
        c = apply(base, spec)
        img = render(c, P, arrays, ctx)              # картинка: строка 0 — верх
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / name / f"{(spec or 'yaml').replace(';', '_')}.png")
        lab = spec or "как в YAML"
        results.append((lab, img))
        sk = img[sky_img] * 255
        st = img[stars_img] * 255
        green = np.median(st[:, 1] - np.maximum(st[:, 0], st[:, 2])) if st.size else float("nan")
        rows.append(f"{lab}\t{' / '.join(f'{np.median(sk[:, k]):.1f}' for k in range(3))}"
                    f"\t{' / '.join(f'{(sk[:, k] <= 0.5).mean() * 100:.2f}' for k in range(3))}"
                    f"\t{(img[neb_img].max(-1) >= 254 / 255).mean() * 100:.2f}\t{green:+.1f}")
        print(f"  готово: {lab}", flush=True)
    tiles, labels = [], []
    for lab, im in results:
        tiles.append(shrink(im, 900)[::-1]); labels.append(f"{lab} · целиком")
    for cn, k in CROPS:
        for lab, im in results:
            t = cut(im, crops[cn])
            tiles.append(np.clip(t * k, 0, 1)[::-1]); labels.append(f"{cn} · {lab} · " + (f"×{k}" if k > 1 else "100%"))
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(results), tile_inches=3.6, interpolation="nearest",
                  title=f"Sh2-305 showcase (bin {base.bin}): {name}")
    (out / f"{name}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
