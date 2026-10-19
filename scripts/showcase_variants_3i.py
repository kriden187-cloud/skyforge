"""3i: варианты цвета showcase на кропах рядом с NASA и доли пикселей с каналом ≥ 0,99 по шагам.

Запуск: uv run python scripts/showcase_variants_3i.py <папка> <имя листа> <вариант> [<вариант> ...]
Вариант — параметры через «;»: "color.shadows.by=max;color.saturation_gamut=limit;color.shadows.value=4".
Пустая строка "" — showcase как в YAML. Пишет <имя>.png (кропы inner, edge, core 100%) и <имя>.tsv.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import load_input, load_nebula_sd, load_sky_mask, load_star_weight
from skyforge.compose import apply_shadows, base_color, saturate, tail_color
from skyforge.config import Config
from skyforge.finish import finish_masks, finish_region
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop
from skyforge.stretch import load_masks, stretch_array

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from sweep_white_3i import nasa_on_grid  # noqa: E402


def apply(cfg: Config, spec: str) -> Config:
    for part in filter(None, (p.strip() for p in spec.split(";"))):
        k, v = part.split("=", 1)
        try:
            val: object = float(v)
        except ValueError:
            val = v
        data = cfg.model_dump(mode="json")
        node = data
        keys = k.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = val
        cfg = Config.model_validate(data)
    return cfg


def main() -> None:
    out, name, specs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    P = Paths(pathlib.Path(".").resolve())
    base_cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(base_cfg, P)
    masks = load_masks(base_cfg, P)
    field = stats_field(base_cfg, P, list(arrays.values()))
    sd, sky, sm = load_nebula_sd(base_cfg, P), load_sky_mask(base_cfg, P), load_star_weight(base_cfg, P)
    sw_, sd_, origin = finish_masks(base_cfg.model_copy(update={"finish": base_cfg.finish}), P)
    crops = {c.name: c for c in with_resolved_crops(base_cfg, P).report.crops}
    final = sorted(P.out_root.glob("ngc3132/*_styles"))[-1] / "showcase" / "ngc3132_showcase.png"
    ours = np.asarray(Image.open(final).convert("RGB")).astype(np.float32) / 255
    nasa = nasa_on_grid(ours, str(P.abs(base_cfg.reference_image)), [(1133, 876), (1901, 1701)],
                        [(1491, 1313), (2500, 2388)])
    names = ("inner", "edge", "core")
    cols: dict[str, list] = {n: [(nasa, "NASA")] for n in names}
    lines = ["вариант\tшаг\tинner_R\tинner_G\tинner_B"]
    for spec in specs:
        c = apply(base_cfg, spec)
        layers = {f: stretch_array(a, c.stretch.for_filter(f), masks=masks, field=field)[0] for f, a in arrays.items()}
        b, _ = base_color(c, layers, sky, sd, field=field)
        x0 = np.clip(b, 0, 1)
        x1 = apply_shadows(x0, c.color.shadows, sd)
        x2 = saturate(x1, c.color.saturation, c.color.saturation_gamut)
        comp = tail_color(c, b, layers, sm, sd)
        label = spec or "как в YAML"
        inner = display_crop(comp.shape, crops["inner"].x, crops["inner"].y, crops["inner"].size, 1)
        fin_inner = finish_region(comp, c.finish, sw_, sd_, origin, inner)
        for step, img in (("до shadows", x0[inner]), ("после shadows", x1[inner]), ("после saturate", x2[inner]),
                          ("после finish", fin_inner)):
            fr = [(img[..., k] >= 0.99).mean() * 100 for k in range(3)]
            lines.append(f"{label}\t{step}\t" + "\t".join(f"{v:.1f}" for v in fr))
            print(f"{label} | {step}: inner R/G/B ≥ 0,99 = {fr[0]:.1f} / {fr[1]:.1f} / {fr[2]:.1f}%")
        for n in names:
            sl = display_crop(comp.shape, crops[n].x, crops[n].y, crops[n].size, 1)
            img = fin_inner if n == "inner" else finish_region(comp, c.finish, sw_, sd_, origin, sl)
            cols[n].append((img, label))
    imgs, labels = [], []
    for n in names:
        sl = display_crop(ours.shape, crops[n].x, crops[n].y, crops[n].size, 1)
        for i, (img, lab) in enumerate(cols[n]):
            imgs.append(img[sl] if i == 0 else img)
            labels.append(f"{n} · {lab}")
    contact_sheet(imgs, labels, out / f"{name}.png", cols=len(specs) + 1, tile_inches=4.0, interpolation="nearest",
                  title="showcase 100%: NASA и варианты; доли каналов ≥ 0,99 — в таблице")
    (out / f"{name}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
