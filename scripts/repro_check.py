"""Аудит 2026-10-27, этап 3: воспроизводимость принятых результатов из сырых данных на их коммитах (в этой папке).

Для каждого объекта по манифесту sources/<slug>.json: sha256 исходников в data/raw сверяются заново (до переключения);
рабочая папка переключается на коммит из манифеста (окружение одно: uv.lock и pyproject.toml на всех принятых коммитах те
же); кеш data/work/<slug> на время прогона переименовывается в <slug>__aside, прогон холодный; финал тем же стилем и bin,
что baseline, с --offline; итог сравнивается с data/baselines по пикселям и по sha256 файла. Новый кеш остаётся как
<slug>__repro_<коммит> (не удаляется), прежний возвращается на место. В конце и при любой ошибке — git checkout master.
Расхождение пикселей — остановка с картой разности.

Запуск (из корня проекта, рабочее дерево без изменений отслеживаемых файлов):
    uv run python scripts/repro_check.py <объект> [<объект> ...]
Результаты — out/audit_9/repro/<объект>.json и out/audit_9/repro/results.tsv.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
RDIR = ROOT / "out" / "audit_9" / "repro"
WORK = ROOT / "data" / "work"
ENV = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
ENV["PYTHONIOENCODING"] = "utf-8"
VERSIONS = ("import sys, numpy, scipy, astropy, reproject, PIL, skimage, tifffile; "
            "print(sys.version.split()[0], numpy.__version__, scipy.__version__, astropy.__version__, "
            "reproject.__version__, PIL.__version__, skimage.__version__, tifffile.__version__)")


def sh(cmd: list[str], log: Path | None = None) -> str:
    r = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log is not None:
        log.write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise SystemExit(f"ошибка {r.returncode}: {' '.join(cmd)}\n{r.stderr[-3000:]}")
    return r.stdout


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while chunk := fh.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def final_png(run_dir: Path, slug: str, style: str) -> Path:
    rj = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    png = rj.get("stages", {}).get("finish", {}).get("files", {}).get("png")
    if png:
        return run_dir / png
    c = sorted(run_dir.glob(f"{slug}_{style}*.png"))
    if not c:
        raise SystemExit(f"в {run_dir} нет итогового PNG")
    return c[0]


def png_chunks(p: Path) -> list[tuple[str, int, str]]:
    """Чанки PNG: тип, длина, sha256 содержимого — чтобы назвать, чем отличаются файлы при тех же пикселях."""
    b = p.read_bytes()
    i, out = 8, []
    while i < len(b):
        n = int.from_bytes(b[i:i + 4], "big")
        t = b[i + 4:i + 8].decode("latin-1")
        out.append((t, n, hashlib.sha256(b[i + 8:i + 8 + n]).hexdigest()[:12]))
        i += 12 + n
    return out


def check_object(lock: dict) -> None:
    slug = lock["slug"]
    short = lock["commit"][:7]
    res: dict = {"slug": slug, "tag": lock["tag"], "commit": lock["commit"]}
    t0 = time.perf_counter()
    sh(["git", "checkout", "-q", "--detach", lock["commit"]])
    res["versions"] = sh(["uv", "run", "--frozen", "python", "-c", VERSIONS]).strip()
    aside, repro = WORK / f"{slug}__aside", WORK / f"{slug}__repro_{short}"
    if aside.exists() or repro.exists():
        raise SystemExit(f"{aside} или {repro} уже есть — разберись вручную")
    had_cache = (WORK / slug).exists()
    if had_cache:
        (WORK / slug).rename(aside)
    res["runs"] = []
    try:
        for b in lock["baseline"]:
            m = re.match(r"(.+)_bin(\d+)\.png$", Path(b["path"]).name)
            style, binn = m.group(1), int(m.group(2))
            t = time.perf_counter()
            sh(["uv", "run", "--frozen", "skyforge", "run", slug, "-s", style, "--bin", str(binn), "--offline"],
               RDIR / f"{slug}_{style}_run.log")
            sec = round(time.perf_counter() - t, 1)
            run_dir = sorted((ROOT / "out" / slug).glob(f"*_{style}"))[-1]
            png = final_png(run_dir, slug, style)
            base = ROOT / b["path"]
            a, c = np.asarray(Image.open(png)), np.asarray(Image.open(base))
            same_pix = a.shape == c.shape and a.dtype == c.dtype and np.array_equal(a, c)
            fs = sha256(png)
            same_file = fs == b["file_sha256"]
            r = {"style": style, "bin": binn, "run_dir": str(run_dir.relative_to(ROOT)), "png": png.name, "seconds": sec,
                 "pixels_equal": bool(same_pix), "file_equal": bool(same_file), "file_sha256": fs}
            if not same_file:
                ca, cb = png_chunks(png), png_chunks(base)
                r["chunks_run"], r["chunks_baseline"] = ca, cb
                r["differs"] = sorted({x[0] for x in ca if x not in cb} | {x[0] for x in cb if x not in ca})
            if not same_pix:
                if a.shape == c.shape:
                    d = np.abs(a.astype(np.int32) - c.astype(np.int32))
                    d = d.max(axis=-1) if d.ndim == 3 else d
                    r["diff_max"], r["diff_fraction"] = int(d.max()), float((d > 0).mean())
                    Image.fromarray(np.clip(d * 50, 0, 255).astype(np.uint8)).save(RDIR / f"{slug}_{style}_diff.png")
                else:
                    r["shape_run"], r["shape_baseline"] = list(a.shape), list(c.shape)
            res["runs"].append(r)
            print(f"{slug} {style} bin{binn} @ {short}: пиксели {'то же' if same_pix else 'ОТЛИЧИЕ'}, файл "
                  f"{'тот же' if same_file else 'другой (' + ', '.join(r.get('differs', [])) + ')'}, {sec:.0f} с",
                  flush=True)
            if not same_pix:
                raise SystemExit(f"{slug} {style}: пиксели не совпали — остановка (карта разности в {RDIR})")
    finally:
        res["total_s"] = round(time.perf_counter() - t0, 1)
        (RDIR / f"{slug}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        if (WORK / slug).exists():
            (WORK / slug).rename(repro)       # холодный кеш прогона — рядом, не удаляется
        if had_cache:
            aside.rename(WORK / slug)
    with open(RDIR / "results.tsv", "a", encoding="utf-8") as fh:
        for r in res["runs"]:
            fh.write("\t".join(map(str, [slug, lock["tag"], lock["commit"][:12], r["style"], r["bin"], r["pixels_equal"],
                                         r["file_equal"], ",".join(r.get("differs", [])), r["seconds"], res["total_s"],
                                         res["versions"]])) + "\n")


def main() -> None:
    RDIR.mkdir(parents=True, exist_ok=True)
    if sh(["git", "status", "--porcelain", "--untracked-files=no"]).strip():
        raise SystemExit("в рабочем дереве изменены отслеживаемые файлы — переключать коммиты нельзя")
    locks = [json.loads((ROOT / "sources" / f"{s}.json").read_text(encoding="utf-8")) for s in sys.argv[1:]]
    # исходники — заново по sha256, до любого переключения
    for lock in locks:
        t = time.perf_counter()
        bad = [e["path"] for e in lock["files"] if sha256(ROOT / e["path"]) != e["sha256"]]
        if bad:
            raise SystemExit(f"{lock['slug']}: исходники не совпадают с манифестом: {bad[:5]}")
        print(f"{lock['slug']}: {len(lock['files'])} исходников совпадают по sha256 ({time.perf_counter() - t:.0f} с)",
              flush=True)
    try:
        for lock in locks:
            check_object(lock)
    finally:
        sh(["git", "checkout", "-q", "master"])


if __name__ == "__main__":
    main()
