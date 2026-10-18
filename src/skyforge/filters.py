"""Имена фильтров и длины волн.

В MAST узкий фильтр NIRCam записан парой с широким блокирующим: "F444W;F405N".
Настоящий фильтр — узкий (N), затем средний (M), затем широкий (W).
"""

from __future__ import annotations

import re

_FILTER_RE = re.compile(r"^F\d{3,4}[A-Z0-9]*$")
# Чем уже полоса, тем она главнее в паре фильтр+пьюпил.
_WIDTH_RANK = {"N": 0, "M": 1, "W2": 2, "W": 3, "LP": 4, "X": 5}


def normalize_filter_name(name: str) -> str:
    name = name.strip().upper()
    if not _FILTER_RE.match(name):
        raise ValueError(f"'{name}' не похоже на имя фильтра (ожидается вроде F090W, F187N)")
    return name


def _width_rank(name: str) -> int:
    suffix = re.sub(r"^F\d{3,4}", "", name)
    return _WIDTH_RANK.get(suffix, 9)


def effective_filter(raw: str) -> str | None:
    """'F444W;F405N' → 'F405N', 'CLEAR;F090W' → 'F090W', 'CLEAR' → None."""
    parts = [p.strip().upper() for p in re.split(r"[;,+\s-]+", raw or "") if p.strip()]
    names = [p for p in parts if _FILTER_RE.match(p)]
    if not names:
        return None
    return min(names, key=_width_rank)


def wavelength_um(name: str, mission: str, instrument: str = "") -> float:
    """Центральная длина волны по имени фильтра.

    JWST: цифры — сотые доли микрона (F090W = 0.90, F1130W = 11.3).
    HST UVIS/ACS: цифры — нанометры (F656N = 0.656 мкм).
    HST WFC3/IR и NICMOS: цифры — сотые микрона (F160W = 1.60).
    """
    m = re.match(r"F(\d{3,4})", name.upper())
    if not m:
        raise ValueError(f"не могу определить длину волны фильтра '{name}'")
    n = int(m.group(1))
    if mission.lower() == "jwst":
        return n / 100
    inst = instrument.upper()
    if "IR" in inst or "NICMOS" in inst:
        return n / 100
    return n / 1000


def sort_by_wavelength(names: list[str], mission: str, instrument: str = "") -> list[str]:
    return sorted(names, key=lambda f: wavelength_um(f, mission, instrument))
