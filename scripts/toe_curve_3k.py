"""3k, п.1: кривая носка неба по фильтрам — данные → уровни после растяжки, рядом пьедестал без носка и phase-3j
(процентиль black). Пишет график и таблицу: где носок сходится с пьедесталом (разность < 1%) и с прямой.

Запуск: uv run python scripts/toe_curve_3k.py <папка>
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib
import numpy as np

from skyforge.align import stats_field
from skyforge.clean import load_input
from skyforge.config import Config
from skyforge.paths import Paths
from skyforge.stretch import curve, load_masks, masked_sample, sky_toe, stretch_array

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    P = Paths(pathlib.Path(".").resolve())
    cfg = Config.load("configs/ngc3132/showcase.yaml", "showcase").model_copy(update={"bin": 1})
    arrays, _ = load_input(cfg, P)
    masks = load_masks(cfg, P)
    field = stats_field(cfg, P, list(arrays.values()))
    c_ped = cfg.with_param("stretch.sky_level", None)
    c_3j = c_ped.with_param("stretch.black_mode", "percentile")
    fig, ax = plt.subplots(2, 6, figsize=(30, 10))
    rows = ["фильтр\tσ неба (данные)\tнебо: 3j / пьедестал / носок, уровни\tp носка"
            "\tнад небом, где носок отличается от пьедестала < 1%\tгде отличается от 3j < 1%"
            "\t99% туманности: 3j / пьедестал / носок"]
    for i, (f, a) in enumerate(arrays.items()):
        s = cfg.stretch.for_filter(f)
        y_t, st = stretch_array(a, s, masks=masks, field=field)
        _, st_p = stretch_array(a, c_ped.stretch.for_filter(f), masks=masks, field=field)
        _, st_3 = stretch_array(a, c_3j.stretch.for_filter(f), masks=masks, field=field)
        sky_v = masked_sample(field.values(a), field.mask(masks[0]))
        med = float(np.median(sky_v)); sig = float(1.4826 * np.median(np.abs(sky_v - med)))
        lo, hi, lo3, hi3 = st["black_value"], st["white_value"], st_3["black_value"], st_3["white_value"]
        v = np.linspace(lo, hi, 400001, dtype=np.float64)
        u = np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32)
        toe = st["sky_toe"]
        yt = curve(sky_toe(u, toe["u_sky"], toe["u_target"])[0], s.method, s.a) * 255
        yp = curve(u, s.method, s.a) * 255
        y3 = curve(np.clip((v - lo3) / (hi3 - lo3), 0, 1), s.method, s.a) * 255
        sky_t = float(np.interp(med, v, yt))
        line = np.interp(med, v, yp) - (np.interp(med, v, yp) - sky_t)  # прямая наклона 1 после asinh через (небо, 6)
        z = (v - med) / sig                                               # данные в σ неба от медианы
        above = v >= med
        rel_p = np.abs(yp - yt) / np.maximum(yt, 1e-6)
        rel_3 = np.abs(y3 - yt) / np.maximum(yt, 1e-6)
        def first_ok(rel):
            bad = np.where(above & (rel >= 0.01))[0]
            j = bad[-1] + 1 if bad.size else np.argmax(above)
            return (f"{yt[j] - sky_t:.1f} уровня над небом ({z[j]:.0f}σ)" if j < len(v) else "нет")
        neb = field.values(a)[field.mask(masks[1])]
        p99 = float(np.percentile(neb[np.isfinite(neb)], 99))
        rows.append(f"{f}\t{sig:.4g}\t{np.interp(med, v, y3):.1f} / {np.interp(med, v, yp):.1f} / {sky_t:.1f}\t{toe['p']:.2f}"
                    f"\t{first_ok(rel_p)}\t{first_ok(rel_3)}"
                    f"\t{np.interp(p99, v, y3):.1f} / {np.interp(p99, v, yp):.1f} / {np.interp(p99, v, yt):.1f}")
        win = (z >= -6) & (z <= 15)
        a0 = ax[0, i]
        a0.plot(z[win], y3[win], label="phase-3j (процентиль)", color="0.6")
        a0.plot(z[win], yp[win], label="пьедестал k3", color="tab:orange")
        a0.plot(z[win], yt[win], label=f"носок → {s.sky_level:g}", color="tab:blue", lw=2)
        a0.plot(z[win], (yp - (np.interp(med, v, yp) - sky_t))[win], "--", color="tab:red", lw=0.8,
                label="прямая наклона 1 после asinh")
        a0.axhline(0, color="k", lw=0.5); a0.axvline(0, color="k", lw=0.5, ls=":")
        a0.set_ylim(-20, max(60, float(yp[win].max()) * 1.05)); a0.set_title(f"{f}: у неба (x — σ неба от медианы)")
        a0.set_xlabel("данные, σ неба от медианы"); a0.set_ylabel("уровни из 255"); a0.grid(alpha=0.3)
        if i == 0:
            a0.legend(fontsize=8)
        a1 = ax[1, i]
        a1.plot(v, y3, color="0.6"); a1.plot(v, yp, color="tab:orange"); a1.plot(v, yt, color="tab:blue")
        a1.set_xscale("symlog", linthresh=max(sig, 1e-6)); a1.set_title(f"{f}: весь диапазон")
        a1.set_xlabel("данные (symlog)"); a1.grid(alpha=0.3)
        del line
    fig.suptitle("Носок неба: данные → уровни. Синяя — носок (п.1), оранжевая — пьедестал k3 без носка, серая — phase-3j")
    fig.tight_layout(); fig.savefig(out / "10_кривая_носка.png", dpi=60); plt.close(fig)
    text = "\n".join(rows) + "\n"
    (out / "10_кривая_носка.tsv").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
