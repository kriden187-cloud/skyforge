"""Фаза 4, шаг 4: слабый свет между волокнами Sh2-305 — перебор (bin 4) с Тевено первой клеткой.

Лист: целиком; bubble, filaments 100%; sky 100% и ×5, seam ×5, край маски туманности ×5 (точка на границе маски в слабой
части поля); при shadows.outside > 0 — карта веса подъёма за маской (halo_weight).
Таблица: выжжено, нули неба по каналам, зелень звёзд, разброс яркости в bubble и filaments относительно phase-4,
скачок яркости поперёк края маски (внутри 1–4 px против снаружи 1–4 px, уровни), мера достоверности — корреляция
сглаженной (σ 16 px при bin 1) яркости итога и слоя F335M после растяжки phase-4 вне маски туманности без звёзд.

Запуск: uv run python scripts/faint_s305.py <папка> <имя> <вариант> [<вариант> ...]   ("" — phase-4 как в YAML)
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

from skyforge.align import load_aligned, stats_field, with_resolved_crops
from skyforge.clean import clean_dir, load_input, load_nebula_sd, load_sky_mask, load_star_weight_map
from skyforge.compose import LUMA, halo_weight
from skyforge.config import Config
from skyforge.finish import finish_masks
from skyforge.paths import Paths
from skyforge.preview import contact_sheet, shrink
from skyforge.stretch import load_masks, stretch_array

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from showcase_variants_3i import apply  # noqa: E402
from sweep_3b import render  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
THEV = pathlib.Path("data/reference/sh2305/thevenot_lw_commons.jpg")


def thevenot_on_grid(ours_bin1_lum: np.ndarray, binf: int) -> tuple[np.ndarray, str]:
    """Тевено → наша сетка bin 1 (картинка, строка 0 — верх): поворот на 90° по часовой, сдвиг по корреляции
    (scripts/refgrid_s305.py — шаг 7: кадр с F182M сдвинут, готовые сдвиги не годятся); затем блоки binf."""
    from refgrid_s305 import ref_on_grid

    return ref_on_grid(THEV, -1, ours_bin1_lum, binf)


def _thevenot_on_grid_phase4(ours_bin1_lum: np.ndarray, binf: int) -> tuple[np.ndarray, str]:
    """Прежнее совмещение (до шага 7): перебор готовых сдвигов (−80, −48) px по sources.md."""
    t = np.asarray(Image.open(THEV).convert("RGB"), dtype=np.float32) / 255
    t = np.rot90(t, -1)
    H, W = ours_bin1_lum.shape
    best = None
    small_o = ours_bin1_lum[::8, ::8]
    for dx, dy in ((-80, -48), (-48, -80), (80, 48), (48, 80), (-80, 48), (80, -48), (-48, 80), (48, -80)):
        canvas = np.zeros((H, W, 3), np.float32)
        ys, xs = max(0, -dy), max(0, -dx)                 # ours[y, x] = theirs[y + dy', x + dx'] с dx' = −dx
        yd, xd = max(0, dy), max(0, dx)
        h = min(H - yd, t.shape[0] - ys)
        w = min(W - xd, t.shape[1] - xs)
        canvas[yd:yd + h, xd:xd + w] = t[ys:ys + h, xs:xs + w]
        hp = lambda a: a - ndimage.uniform_filter(a, 9)
        a, b = hp(small_o), hp((canvas @ LUMA)[::8, ::8])
        c = float((a * b).mean() / (a.std() * b.std() + 1e-9))
        if best is None or c > best[0]:
            best = (c, (dx, dy), canvas)
    c, sh, canvas = best
    if binf > 1:
        Hb, Wb = H // binf, W // binf
        canvas = canvas[:Hb * binf, :Wb * binf].reshape(Hb, binf, Wb, binf, 3).mean((1, 3))
    return canvas, f"сдвиг {sh}, корреляция мелких деталей {c:.2f}"


def main() -> None:
    out, name, specs = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    (out / name).mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    base = Config.load("configs/sh2305/showcase.yaml", "showcase")
    arrays, _ = load_input(base, P)
    load_aligned(base, P)
    sky, sd = load_sky_mask(base, P), load_nebula_sd(base, P)
    field = stats_field(base, P, list(arrays.values()))
    ctx = {"field": field, "sky": sky, "sd": sd, "fin": finish_masks(base, P)}
    H, W = sd.shape
    full = np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
    sw = np.load(clean_dir(base, P) / "star_weight.npy")
    stars = sw > 0.01
    sky_img, neb_img = (sky & full)[::-1], (sd > 0)[::-1]
    star_img = ((sw > 0.5) & (sd < 0))[::-1]
    sd_img = sd[::-1]
    out_img = ((sd < 0) & full & ~ndimage.binary_dilation(stars, iterations=3))[::-1]
    f335 = stretch_array(arrays["F335M"], base.stretch.for_filter("F335M"), masks=load_masks(base, P), field=field)[0][::-1]
    sig = 16 / base.bin

    def smooth_valid(a, m):
        num = ndimage.gaussian_filter(np.where(m, a, 0).astype(np.float32), sig)
        den = ndimage.gaussian_filter(m.astype(np.float32), sig)
        return num / np.maximum(den, 1e-6)

    ref_s = smooth_valid(np.nan_to_num(f335), out_img)

    def fidelity(img):
        a = smooth_valid(img @ LUMA, out_img)[out_img]
        b = ref_s[out_img]
        return float(np.corrcoef(a, b)[0, 1])

    crops = {c.name: c for c in with_resolved_crops(base, P).report.crops}

    def cut(img, c=None, center=None, size=None):
        s = max(16, (c.size if c else size) // base.bin)
        cx, cy = (int(round(c.x / base.bin)), int(round(c.y / base.bin))) if c else center
        y0, x0 = int(np.clip(cy - s // 2, 0, H - s)), int(np.clip(cx - s // 2, 0, W - s))
        return img[y0:y0 + s, x0:x0 + s]

    # точка на краю маски туманности в слабой части: граница, где сглаженный F335M ниже медианы границы, далеко от звёзд
    edge = (np.abs(sd_img) < 1.5) & ~ndimage.binary_dilation(stars[::-1], iterations=20)
    ys, xs = np.nonzero(edge)
    vals = ref_s[ys, xs]
    k = np.argsort(np.abs(vals - np.percentile(vals, 25)))[0]
    edge_c = (int(xs[k]), int(ys[k]))
    band_in, band_out = (sd_img > 1) & (sd_img < 4), (sd_img < -1) & (sd_img > -4)

    rows = ["вариант\tвыжжено, %\tнулей неба R/G/B, %\tнебо R/G/B\tзелень звёзд\tσ bubble / filaments, % от phase-4"
            "\tскачок на краю маски, уровни\tдостоверность (корр. с F335M вне маски)"]
    thev_full = None
    results = []
    sigma_ref = None
    for spec in specs:
        c = apply(base, spec)
        img = render(c, P, arrays, ctx)
        Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(out / name / f"{(spec or 'phase-4').replace(';', '_')}.png")
        lab = spec or "phase-4 (как в YAML)"
        hw = None
        sh = c.color.shadows
        if sh.outside > 0 and sh.halo_k > 0:
            hw = halo_weight(img[::-1].copy(), sd, sh, load_star_weight_map(c, P))[::-1]
        results.append((lab, img, hw))
        lum = img @ LUMA * 255
        sb = [float(cut(lum, crops[n]).std()) for n in ("bubble", "filaments")]
        if sigma_ref is None:
            sigma_ref = sb
        sk, st = img[sky_img] * 255, img[star_img] * 255
        green = float(np.median(st[:, 1] - np.maximum(st[:, 0], st[:, 2])))
        jump = float(np.median(lum[band_in]) - np.median(lum[band_out]))
        rows.append(f"{lab}\t{(img[neb_img].max(-1) >= 254 / 255).mean() * 100:.2f}"
                    f"\t{' / '.join(f'{(sk[:, i] <= 0.5).mean() * 100:.2f}' for i in range(3))}"
                    f"\t{' / '.join(f'{np.median(sk[:, i]):.1f}' for i in range(3))}\t{green:+.1f}"
                    f"\t{sb[0] / sigma_ref[0] * 100:.0f} / {sb[1] / sigma_ref[1] * 100:.0f}\t{jump:+.1f}\t{fidelity(img):.3f}")
        print(f"  готово: {lab}", flush=True)
        if thev_full is None:
            ours1 = np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((W * base.bin, H * base.bin)),
                               dtype=np.float32) / 255 @ LUMA
            thev_full, treg = thevenot_on_grid(ours1, base.bin)
            thev = thev_full[:H, :W]
            rows.append(f"Тевено (Commons, {treg})\t{(thev[neb_img].max(-1) >= 254 / 255).mean() * 100:.2f}\t—\t"
                        f"{' / '.join(f'{np.median(thev[sky_img][:, i] * 255):.1f}' for i in range(3))}\t—\t"
                        f"{cut(thev @ LUMA * 255, crops['bubble']).std() / sigma_ref[0] * 100:.0f} / "
                        f"{cut(thev @ LUMA * 255, crops['filaments']).std() / sigma_ref[1] * 100:.0f}\t"
                        f"{np.median((thev @ LUMA * 255)[band_in]) - np.median((thev @ LUMA * 255)[band_out]):+.1f}\t"
                        f"{fidelity(thev):.3f}")
    allc = [("Тевено", thev, None)] + results
    tiles, labels = [], []
    rowspec = [("целиком", None, 1), ("bubble · 100%", "bubble", 1), ("filaments · 100%", "filaments", 1),
               ("sky · 100%", "sky", 1), ("sky · ×5", "sky", 5), ("seam · ×5", "seam", 5), ("край маски · ×5", "edge", 5)]
    for rl, cn, kk in rowspec:
        for lab, im, _ in allc:
            if cn is None:
                t = shrink(im, 700)
            elif cn == "edge":
                t = cut(im, center=edge_c, size=400)
            else:
                t = cut(im, crops[cn])
            tiles.append(np.clip(t * kk, 0, 1)[::-1]); labels.append(f"{rl} · {lab}")
    if any(hw is not None for _, _, hw in results):
        for lab, _, hw in allc:
            tiles.append(np.repeat(shrink(hw if hw is not None else np.zeros((H, W), np.float32), 700)[::-1, :, None], 3, -1))
            labels.append(f"вес подъёма за маской · {lab}")
    contact_sheet(tiles, labels, out / f"{name}.png", cols=len(allc), tile_inches=3.4, interpolation="nearest",
                  title=f"Sh2-305: слабый свет между волокнами — {name} (bin {base.bin}); Тевено первой клеткой")
    (out / f"{name}.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
