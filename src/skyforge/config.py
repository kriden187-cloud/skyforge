"""Схема YAML-конфига. Все параметры обработки живут здесь, в коде этапов чисел нет.

Значения по умолчанию в схеме — документированная часть конфига: если поле не
указано в YAML, берётся отсюда, и итоговый конфиг целиком копируется в out/.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from skyforge.filters import normalize_filter_name


class _Strict(BaseModel):
    # Опечатка в имени поля должна ругаться, а не молча игнорироваться.
    model_config = ConfigDict(extra="forbid")


class AlignConfig(_Strict):
    order: Literal["nearest-neighbor", "bilinear", "biquadratic", "bicubic"] = "bilinear"
    # Доля непустых пикселей на краю обрезки: 1.0 — ни одного NaN на границе.
    crop_min_coverage: float = Field(0.99, gt=0, le=1)


class StarsConfig(_Strict):
    # Насыщенные ядра в i2d — дыры NaN с WHT=0 внутри поля (слоя DQ в продуктах уровня 3 нет).
    fill_cores: bool = False
    max_core_arcsec: float = Field(2.0, gt=0, description="дыры шире этого — не ядра, а пропуски мозаики")


class BackgroundConfig(_Strict):
    subtract: bool = False
    # Шаг 1: постоянная поправка на каждую область CON/CTX (свой набор экспозиций) — убирает
    # ступеньки между плитками мозаики, которые гладкая модель только размазывает.
    tiles: bool = True
    tile_min_area_arcsec2: float = Field(1.0, gt=0, description="меньше свободной площади — поправка от соседа")
    tile_mask_dilate_arcsec: float = Field(4.0, ge=0, description="отступ от туманности при оценке плиток")
    # Шаг 2: гладкая модель по блокам — остаточные градиенты.
    smooth: bool = True
    # Значения по умолчанию подобраны на NGC 3132: блок 4" и маска 1.5" съедали ореол оболочки.
    box_arcsec: float = Field(8.0, gt=0, description="размер блока сетки фона")
    filter_size: int = Field(3, ge=1, description="медианное сглаживание сетки, в блоках (нечётное)")
    mask_sigma: float = Field(1.5, gt=0, description="порог маски источников над фоном, в σ")
    mask_smooth_arcsec: float = Field(1.0, ge=0, description="сглаживание перед порогом: ловит слабую туманность")
    mask_dilate_arcsec: float = Field(3.0, ge=0, description="расширение маски вокруг источников")
    min_free: float = Field(0.3, gt=0, le=1, description="доля свободных от маски пикселей, чтобы блок считался")

    @field_validator("filter_size")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("background.filter_size должен быть нечётным")
        return v


class StretchConfig(_Strict):
    method: Literal["asinh", "log", "sqrt", "linear"] = "asinh"
    black: float = Field(0.5, ge=0, lt=100, description="процентиль фона → 0")
    white: float = Field(99.8, gt=0, le=100, description="процентиль ярких деталей → 1")
    a: float = Field(0.02, gt=0, description="сила растяжки: меньше — сильнее")
    # Переопределения для отдельных фильтров: {F187N: {white: 99.9}}
    per_filter: dict[str, dict[str, float]] = {}

    @model_validator(mode="after")
    def _check(self) -> StretchConfig:
        if self.black >= self.white:
            raise ValueError(f"stretch.black ({self.black}) должен быть меньше white ({self.white})")
        allowed = {"black", "white", "a"}
        for f, params in self.per_filter.items():
            extra = set(params) - allowed
            if extra:
                raise ValueError(f"stretch.per_filter.{f}: неизвестные поля {sorted(extra)}")
        self.per_filter = {normalize_filter_name(k): v for k, v in self.per_filter.items()}
        return self

    def for_filter(self, name: str) -> StretchConfig:
        params = self.per_filter.get(name)
        if not params:
            return self
        return self.model_copy(update={**params, "per_filter": {}})


class ColorConfig(_Strict):
    mode: Literal["wavelength", "manual"] = "wavelength"
    # Оттенок (градусы HSV) для самой короткой и самой длинной волны: 240 синий, 0 красный.
    hue_range: tuple[float, float] = (240.0, 0.0)
    # average — сбалансированная сумма (все слои по 1 дают белый); screen — как в Photoshop,
    # ярче, но при 4+ фильтрах сводит туманность в белый и жёлтый.
    blend: Literal["average", "screen"] = "average"
    saturation: float = Field(1.0, ge=0)
    # Подъём теней после смешивания (asinh по яркости): меньше — сильнее, null — выкл.
    lift: float | None = Field(None, gt=0)
    overrides: dict[str, str] = {}
    weights: dict[str, float] = {}
    neutral_background: bool = True
    background_percentile: float = Field(10.0, gt=0, lt=100)

    @field_validator("overrides")
    @classmethod
    def _hex(cls, v: dict[str, str]) -> dict[str, str]:
        out = {}
        for k, color in v.items():
            if not re.fullmatch(r"#?[0-9a-fA-F]{6}", color):
                raise ValueError(f"color.overrides.{k}: '{color}' — нужен цвет вида #ff8040")
            out[normalize_filter_name(k)] = "#" + color.lstrip("#").lower()
        return out

    @field_validator("weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        return {normalize_filter_name(k): w for k, w in v.items()}


class FinishConfig(_Strict):
    denoise: float = Field(0.0, ge=0, le=1, description="0 — выкл, 1 — максимум")
    denoise_iters: int = Field(100, ge=1)
    sharpen: float = Field(0.0, ge=0, description="сила нерезкой маски, 0 — выкл")
    sharpen_radius: float = Field(2.0, gt=0, description="радиус нерезкой маски в пикселях")
    # Применяется на этапе align: поворот на север делается одной интерполяцией вместе с совмещением.
    north_up: bool = True


class PreviewConfig(_Strict):
    max_size: int = Field(1600, ge=200, description="длинная сторона превью, px")
    checker_cell: int = Field(256, ge=8, description="клетка шахматки на полном разрешении, px")
    zoom: int = Field(400, ge=64, description="сторона кропа 100%, px")
    core_zoom: int = Field(48, ge=8, description="сторона кропа ядра звезды, px")


class SweepAxis(_Strict):
    param: str
    values: list[float]


class SweepConfig(_Strict):
    x: SweepAxis = SweepAxis(param="stretch.a", values=[0.005, 0.02, 0.08])
    y: SweepAxis = SweepAxis(param="stretch.white", values=[99.5, 99.8, 99.95])


class Config(_Strict):
    target: str
    slug: str | None = None  # имя папок; по умолчанию из target: "NGC 3132" → ngc3132
    coords: str | None = None  # "RA Dec" в градусах, если имя не резолвится
    radius: str = "3 arcmin"
    mission: Literal["jwst", "hst"] = "jwst"
    instrument: str = "nircam"
    proposal_id: str | None = None
    filters: list[str] = []
    allow_missing: bool = True
    calib_level: int = Field(3, ge=1, le=4)
    reference: str | None = None
    bin: int = Field(1, ge=1)
    align: AlignConfig = AlignConfig()
    stars: StarsConfig = StarsConfig()
    background: BackgroundConfig = BackgroundConfig()
    stretch: StretchConfig = StretchConfig()
    color: ColorConfig = ColorConfig()
    finish: FinishConfig = FinishConfig()
    preview: PreviewConfig = PreviewConfig()
    sweep: SweepConfig = SweepConfig()
    credit: str = "NASA, ESA, CSA, STScI (MAST)"
    author: str = "Denis"

    @field_validator("filters")
    @classmethod
    def _filters(cls, v: list[str]) -> list[str]:
        out = [normalize_filter_name(f) for f in v]
        dup = {f for f in out if out.count(f) > 1}
        if dup:
            raise ValueError(f"фильтры повторяются: {sorted(dup)}")
        return out

    @field_validator("mission", "instrument", mode="before")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    @model_validator(mode="after")
    def _check(self) -> Config:
        if self.reference:
            self.reference = normalize_filter_name(self.reference)
            if self.filters and self.reference not in self.filters:
                raise ValueError(f"reference {self.reference} нет в filters {self.filters}")
        if not self.slug:
            self.slug = slugify(self.target)
        if self.color.mode == "manual":
            missing = [f for f in self.filters if f not in self.color.overrides]
            if missing:
                raise ValueError(f"color.mode: manual, но нет цвета в overrides для {missing}")
        return self

    # --- загрузка и сохранение ---

    @classmethod
    def load(cls, path: str | Path) -> Config:
        path = Path(path)
        with path.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    def digest(self, *sections: str) -> str:
        """Хеш выбранных секций конфига — ключ кеша для промежуточных данных."""
        data = self.model_dump(mode="json")
        if sections:
            data = {k: data[k] for k in sections}
        blob = json.dumps(data, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def with_param(self, dotted: str, value: float) -> Config:
        """Копия конфига с одним изменённым параметром: 'stretch.a' → 0.05."""
        data = self.model_dump(mode="json")
        node = data
        keys = dotted.split(".")
        for k in keys[:-1]:
            if k not in node or not isinstance(node[k], dict):
                raise ValueError(f"нет такого параметра: {dotted}")
            node = node[k]
        # В словарях по фильтрам (color.weights.F212N) ключа может ещё не быть.
        open_dict = len(keys) >= 2 and keys[-2] == "weights"
        if keys[-1] not in node and not open_dict:
            raise ValueError(f"нет такого параметра: {dotted}")
        node[keys[-1]] = value
        return Config.model_validate(data)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not slug:
        raise ValueError(f"не получается сделать имя папки из '{name}', задай slug в конфиге")
    return slug
