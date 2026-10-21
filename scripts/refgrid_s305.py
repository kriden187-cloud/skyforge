"""Фаза 4: чужая обработка Sh2-305 (data/reference/sh2305/) на нашей сетке bin 1 — поворот и целый сдвиг по корреляции.

Сдвиг ищется без заранее заданных значений: грубо — взаимная корреляция мелких деталей на картинках, уменьшенных в 8 раз
(FFT), точно — перебор ±8 px при bin 1 в центральном окне. Кадр нашей сборки меняется (шаг 7: с F182M он сдвинут на
92 строки), поэтому прежний перебор восьми готовых сдвигов (−80, −48) больше не годится.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image
from scipy import ndimage, signal

Image.MAX_IMAGE_PIXELS = None
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _hp(a: np.ndarray, k: int) -> np.ndarray:
    a = a - ndimage.uniform_filter(a, k)
    return a / (a.std() + 1e-9)


def _block(a: np.ndarray, f: int) -> np.ndarray:
    h, w = a.shape[0] // f, a.shape[1] // f
    return a[:h * f, :w * f].reshape(h, f, w, f).mean((1, 3))


def place(t: np.ndarray, shape: tuple[int, int], dy: int, dx: int) -> np.ndarray:
    """ours[y, x] = theirs[y − dy, x − dx]: картинка t на холсте shape, пусто — 0."""
    H, W = shape
    canvas = np.zeros((H, W, 3), np.float32)
    ys, xs, yd, xd = max(0, -dy), max(0, -dx), max(0, dy), max(0, dx)
    h, w = min(H - yd, t.shape[0] - ys), min(W - xd, t.shape[1] - xs)
    if h > 0 and w > 0:
        canvas[yd:yd + h, xd:xd + w] = t[ys:ys + h, xs:xs + w]
    return canvas


def ref_on_grid(path: str | pathlib.Path, rot_k: int, ours_lum: np.ndarray, binf: int = 1) -> tuple[np.ndarray, str]:
    """Референс → наша сетка bin 1 (картинка, строка 0 — верх). rot_k — np.rot90 (−1 — 90° по часовой).
    Возвращает холст (блоки binf) и подпись совмещения."""
    t = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    t = np.rot90(t, rot_k)
    H, W = ours_lum.shape
    f = 8
    a = _hp(_block(ours_lum, f), 9)
    b = _hp(_block(t @ LUMA, f), 9)
    cc = signal.fftconvolve(a, b[::-1, ::-1], mode="full")
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    dy0, dx0 = (py - (b.shape[0] - 1)) * f, (px - (b.shape[1] - 1)) * f
    # уточнение при bin 1 в центральном окне 1536 px
    s = 1536
    cy, cx = H // 2, W // 2
    o = _hp(ours_lum[cy - s // 2:cy + s // 2, cx - s // 2:cx + s // 2], 9)
    tl = t @ LUMA
    best = (-2.0, dy0, dx0)
    for ddy in range(-f, f + 1):
        for ddx in range(-f, f + 1):
            dy, dx = dy0 + ddy, dx0 + ddx
            y0, x0 = cy - s // 2 - dy, cx - s // 2 - dx
            if y0 < 0 or x0 < 0 or y0 + s > tl.shape[0] or x0 + s > tl.shape[1]:
                continue
            c = float((o * _hp(tl[y0:y0 + s, x0:x0 + s], 9)).mean())
            if c > best[0]:
                best = (c, dy, dx)
    c, dy, dx = best
    canvas = place(t, (H, W), dy, dx)
    if binf > 1:
        Hb, Wb = H // binf, W // binf
        canvas = canvas[:Hb * binf, :Wb * binf].reshape(Hb, binf, Wb, binf, 3).mean((1, 3))
    return canvas, f"сдвиг (x {dx}, y {dy}), корреляция мелких деталей {c:.2f}"
