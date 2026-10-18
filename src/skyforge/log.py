"""Общая консоль и замер времени этапов."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from rich.console import Console

console = Console(highlight=False)


class SkyforgeError(Exception):
    """Ошибка, понятная пользователю: печатается без трейсбэка."""


@contextmanager
def timed(info: dict) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        info["seconds"] = round(time.perf_counter() - t0, 2)
