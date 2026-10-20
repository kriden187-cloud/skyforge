"""3k, п.4: сетка «прозрачности» оболочки и середины для владельца — color.saturation × color.shadows.value при
двух finish.local_contrast.amount; кропы inner и edge при 100%, NASA первым в каждой строке. Финиш — по всему кадру
(совпадает с конвейером байт в байт). В подписи — разброс
яркости внутри кропа (σ, шкала 0–1) и у NASA. Отдельный лист: денойз 0 / текущий / вдвое сильнее на кропе edge.

Запуск: uv run python scripts/transparency_grid_3k.py <папка> <png текущего showcase (для сетки NASA)>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.align import stats_field, with_resolved_crops
from skyforge.clean import load_nebula_sd, load_saturated_stars, load_sky_mask
from skyforge.compose import base_color, load_stretched, tail_color
from skyforge.config import Config
from skyforge.finish import finish_array, finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, display_crop

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from showcase_variants_3i import apply  # noqa: E402
from sweep_white_3i import nasa_on_grid  # noqa: E402

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
SATS, SHADOWS, AMOUNTS, DENOISE = (1.0, 1.2, 1.4), (2, 3, 4), (0.3, 0.6), (0.0, None, None)


def sig(img: np.ndarray) -> float:
    return float(np.std(np.clip(img, 0, 1) @ LUMA))


def main() -> None:
    out, cur_png = pathlib.Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    layers, _ = load_stretched(base, P)
    field = stats_field(base, P, list(layers.values()))
    sd, sky = load_nebula_sd(base, P), load_sky_mask(base, P)
    sat_stars = load_saturated_stars(base, P)
    sw_, sd_, origin = finish_masks(base, P)
    b, _ = base_color(base, layers, sky, sd, field=field)
    ours = np.asarray(Image.open(cur_png).convert("RGB")).astype(np.float32) / 255
    nasa = nasa_on_grid(ours, str(P.abs(base.reference_image)))          # сетка: строка 0 — низ
    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}
    sls = {n: display_crop(b.shape, crops[n].x, crops[n].y, crops[n].size, 1) for n in ("inner", "edge")}
    nasa_c = {n: nasa[sl] for n, sl in sls.items()}
    cur = (base.color.saturation, base.color.shadows.value, base.finish.local_contrast.amount)
    tiles = {n: [] for n in sls}
    labels = {n: [] for n in sls}
    rows = ["кроп\tamount\tsaturation\tshadows\tσ яркости\tσ NASA"]
    for amount in AMOUNTS:
        for sat in SATS:
            for n in sls:
                tiles[n].append(nasa_c[n])
                labels[n].append(f"NASA · σ {sig(nasa_c[n]):.3f}")
            for sv in SHADOWS:
                c = apply(base, f"color.saturation={sat};color.shadows.value={sv};finish.local_contrast.amount={amount}")
                comp = tail_color(c, b, layers, None, sd, sat_stars)
                full = finish_array(comp, c.finish, sw_, sd_, origin)   # весь кадр: финиш по кропу с запасом не совпадал с конвейером
                mark = " (сейчас)" if (sat, sv, amount) == cur else ""
                for n, sl in sls.items():
                    fin = full[sl]
                    s = sig(fin)
                    tiles[n].append(fin)
                    labels[n].append(f"контраст {amount:g} · насыщ. {sat:g} · тени {sv:g}{mark} · σ {s:.3f}")
                    rows.append(f"{n}\t{amount:g}\t{sat:g}\t{sv:g}\t{s:.4f}\t{sig(nasa_c[n]):.4f}")
            print(f"  amount {amount}, saturation {sat}: готово", flush=True)
    for n in sls:
        contact_sheet(tiles[n], labels[n], out / f"25_прозрачность_{n}.png", cols=4, tile_inches=3.6, interpolation="nearest",
                      title=f"{n} 100%: локальный контраст (amount) × насыщенность × тени — NASA первым в строке; σ — разброс яркости в кропе")
    # денойз на кропе edge при текущем цвете
    comp = tail_color(base, b, layers, None, sd, sat_stars)
    dn = base.finish.denoise
    t, lab = [nasa_c["edge"]], [f"NASA · σ {sig(nasa_c['edge']):.3f}"]
    for v, name in ((0.0, "денойз 0"), (dn, f"денойз {dn:g} (сейчас)"), (min(1.0, 2 * dn), f"денойз {min(1.0, 2 * dn):g} (вдвое)")):
        c = apply(base, f"finish.denoise={v}")
        fin = finish_array(comp, c.finish, sw_, sd_, origin)[sls["edge"]]
        t.append(fin)
        lab.append(f"{name} · σ {sig(fin):.3f}")
        rows.append(f"edge денойз\t\t\t{v:g}\t{sig(fin):.4f}\t{sig(nasa_c['edge']):.4f}")
    contact_sheet(t, lab, out / "26_денойз_edge.png", cols=4, tile_inches=4.2, interpolation="nearest",
                  title="edge 100%: денойз 0 / текущий / вдвое сильнее — тонкие волокна оболочки")
    (out / "25_прозрачность.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows[-4:]))


if __name__ == "__main__":
    main()
