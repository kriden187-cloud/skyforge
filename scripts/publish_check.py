"""Фаза 5: проверка публикации — лист на объект и стиль и таблица тона.

Лист: верхний ряд — мастер и три версии (web, social, preview) в одинаковом размере показа; нижний — кроп 600 px
мастера при 100% и то же место неба в веб-версии при 100% (меньше пикселей) и её же, увеличенной до размера кропа
мастера (видно, что потеряно при уменьшении). Тон: каждая JPEG-версия против мастера, уменьшенного усреднением (BOX)
до того же размера, — медиана и 95-й процентиль |разности| по каналам, уровни.

Запуск: uv run python scripts/publish_check.py <папка out/publish> <дата> <папка листов>
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from PIL import Image

from skyforge.preview import contact_sheet

Image.MAX_IMAGE_PIXELS = None
CROP = 600


def main() -> None:
    root, day, out = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rows = ["объект_стиль\tверсия\tразмер\tмедиана |Δ| R/G/B\tсдвиг тона (знаковая медиана Δ) R/G/B\tp95 |Δ| R/G/B"
            "\tпорог медианы |Δ| ≤ 1"]
    for master in sorted(root.glob(f"*/{day}/*_master.png")):
        stem = master.name[: -len("_master.png")]
        m = np.asarray(Image.open(master).convert("RGB"))
        H, W = m.shape[:2]
        vers = {k: np.asarray(Image.open(master.with_name(f"{stem}_{k}.jpg")).convert("RGB")) for k in ("web", "social", "preview")}
        tiles, labels = [np.asarray(Image.fromarray(m).resize(vers["web"].shape[1::-1], Image.LANCZOS)).astype(np.float32) / 255], \
            [f"мастер {W}×{H} (показан уменьшенным)"]
        for k, v in vers.items():
            ref = np.asarray(Image.fromarray(m).resize((v.shape[1], v.shape[0]), Image.BOX), dtype=np.float32)
            d = np.abs(v.astype(np.float32) - ref)
            med = [float(np.median(d[..., c])) for c in range(3)]
            p95 = [float(np.percentile(d[..., c], 95)) for c in range(3)]
            bias = [float(np.median(v[..., c].astype(np.float32) - ref[..., c])) for c in range(3)]
            rows.append(f"{stem}\t{k}\t{v.shape[1]}×{v.shape[0]}\t{' / '.join(f'{x:.0f}' for x in med)}\t"
                        f"{' / '.join(f'{x:+.0f}' for x in bias)}\t{' / '.join(f'{x:.0f}' for x in p95)}\t≤ 1 — {'да' if max(med) <= 1 else 'НЕТ'}")
            tiles.append(np.asarray(Image.fromarray(v).resize(vers["web"].shape[1::-1], Image.LANCZOS)).astype(np.float32) / 255)
            labels.append(f"{k} {v.shape[1]}×{v.shape[0]}\nмедиана |Δ| к мастеру {' / '.join(f'{x:.0f}' for x in med)}")
        cy, cx = H // 2, W // 2
        mc = m[cy - CROP // 2:cy + CROP // 2, cx - CROP // 2:cx + CROP // 2]
        web = vers["web"]
        s = web.shape[1] / W
        wy0, wx0, ws = int(round((cy - CROP // 2) * s)), int(round((cx - CROP // 2) * s)), max(1, int(round(CROP * s)))
        wc = web[wy0:wy0 + ws, wx0:wx0 + ws]
        canvas = np.zeros_like(mc)
        canvas[:wc.shape[0], :wc.shape[1]] = wc
        tiles += [mc.astype(np.float32) / 255, canvas.astype(np.float32) / 255,
                  np.asarray(Image.fromarray(wc).resize((CROP, CROP), Image.NEAREST)).astype(np.float32) / 255,
                  np.zeros_like(mc, dtype=np.float32)]
        labels += [f"мастер · центр {CROP} px при 100%", f"web · то же место при 100% ({ws} px, масштаб {s:.2f})",
                   f"web · то же место, увеличено ×{1 / s:.1f} (nearest)", ""]
        # contact_sheet рисует origin=lower: картинки (строка 0 — верх) переворачиваются
        contact_sheet([t[::-1] for t in tiles], labels, out / f"90_{stem}.png", cols=4, tile_inches=4.5,
                      interpolation="nearest", title=f"{stem}: мастер и версии; кроп 100% мастера против веб-версии")
        print(f"  {stem}", flush=True)
    (out / "90_тон_версий.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
