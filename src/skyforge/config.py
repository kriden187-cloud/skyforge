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


class DestripeConfig(_Strict):
    # Полосы 1/f детекторов NIRCam: смещения строк внутри каждой области CON (до совмещения).
    enabled: bool = True
    columns: bool = True  # после строк — то же по столбцам внутри каждой области CON
    # 0 — вычитать медиану строки целиком (уровень области тоже уходит в ноль, как в задании);
    # N — только отклонение от скользящей медианы по N строкам: уровень и плавный ход фона остаются.
    window_rows: int = Field(201, ge=0, description="окно скользящей медианы, строк; 0 — вычитать медиану целиком")
    mask_grow_px: int = Field(5, ge=0, description="расширение маски туманности и звёзд")
    # Дифракционные лучи длиннее ореола в маске звёзд: луч вдоль столбца тянет медиану столбца,
    # и destripe вычитает его как полосу (тёмная полоса вдоль луча). Пиксели ярче фона на pixel_sigma
    # шума (с расширением pixel_grow_px) в медианы не идут.
    pixel_sigma: float = Field(3.0, gt=0, description="порог яркого пикселя над фоном, в σ шума")
    pixel_grow_px: int = Field(2, ge=0, description="расширение маски ярких пикселей")
    # Линия, у которой свободна лишь часть длины внутри области (остальное под маской), даёт медиану
    # только этой части — не уровень линии; такие линии получают значение от соседей.
    min_free_fraction: float = Field(0.5, ge=0, le=1, description="доля свободных пикселей линии внутри области")
    min_pixels: int = Field(50, ge=1, description="меньше свободных пикселей в строке — медиана от соседних строк")
    min_region_rows: int = Field(20, ge=1,
                                 description="область CON меньше min_pixels×min_region_rows пикселей приписывается соседней")
    preview_filter: str | None = None  # для destripe_check.png; по умолчанию — самый полосатый фильтр


class CleanConfig(_Strict):
    destripe: DestripeConfig = DestripeConfig()


class ReportCrop(_Strict):
    # Координаты — центр кропа на итоговой картинке при bin 1 (x вправо, y вниз, как в просмотрщике).
    name: str
    x: int
    y: int
    size: int = Field(400, ge=16)


class ReportConfig(_Strict):
    crops: list[ReportCrop] = []
    # Принятый результат для сравнения: data/baselines/<slug>/<baseline>/<стиль>_bin<N>.png
    # (сохраняется командой skyforge baseline). Отчёт строит карту разности, фактуру и кропы рядом.
    baseline: str | None = None
    texture_ring: tuple[int, int] = (30, 200)  # кольцо вокруг центральной звезды (кроп core), px при bin 1


class StarsConfig(_Strict):
    # Насыщенные ядра в i2d — дыры NaN с WHT=0 внутри поля (слоя DQ в продуктах уровня 3 нет).
    fill_cores: bool = False
    max_core_arcsec: float = Field(2.0, gt=0, description="дыры шире этого — не ядра, а пропуски мозаики")
    # Маска звёзд для color.star_saturation: компактные яркие источники, туманность исключена.
    mask_sigma: float = Field(3.0, gt=0, description="порог пикселей звезды над фоном, в σ")
    peak_sigma: float = Field(20.0, gt=0, description="пик слабее — не звезда (шум, слабые галактики)")
    max_star_area_arcsec2: float = Field(20.0, gt=0, description="больше — уже туманность")
    halo_arcsec: float = Field(1.5, ge=0, description="ореол вокруг звезды, входящий в маску")
    local_window_arcsec: float = Field(6.0, gt=0, description="окно местного фона под звездой (шире звезды с ореолом)")
    nebula_level: float = Field(0.05, gt=0, le=1,
                                description="яркость местного фона (0–1 итоговой картинки), с которой под звездой туманность")
    # Звёзды ищутся в одном фильтре: по умолчанию самом коротковолновом — там у звезды самый резкий
    # пик, а узлы туманности и фоновые галактики слабые. Туманность исключается по всем фильтрам.
    detect_filter: str | None = None


class BackgroundConfig(_Strict):
    subtract: bool = False
    # Шаг 1: постоянная поправка на каждую область CON/CTX (свой набор экспозиций) — убирает
    # ступеньки между плитками мозаики, которые гладкая модель только размазывает.
    tiles: bool = True
    tile_min_area_arcsec2: float = Field(1.0, gt=0, description="меньше свободной площади — поправка от соседа")
    tile_mask_dilate_arcsec: float = Field(4.0, ge=0, description="отступ от туманности при оценке плиток")
    # Область без своего неба (целиком под маской) получает поправку сшивкой по границе с измеренными соседями.
    tile_band_px: int = Field(8, ge=1, description="полоса по обе стороны границы областей для сшивки")
    # В сшивке не участвуют яркие пиксели (лучи звёзд, яркая туманность): внутри туманности ступенька
    # на границе сбивается её структурой — такие области получают поправку ближайшей измеренной, как раньше.
    tile_pixel_sigma: float = Field(3.0, gt=0, description="яркий пиксель для сшивки, в σ шума фона")
    # Своя медиана — только если свободного неба достаточно и по площади, и по доле области: узкая полоса
    # дизеринга у ореола, где свободен край, иначе получит уровень этого края, а не свой.
    tile_min_free_fraction: float = Field(0.3, ge=0, le=1, description="доля свободных пикселей для своей медианы")
    block_px: int = Field(64, ge=8, description="блок для метрики остаточных ступенек фона (медианы блоков)")
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
    # masks — black считается по маске пустого неба, white — по маске туманности (этап clean):
    # очистка фона и обрезка кадра не двигают растяжку. frame — по всему кадру, как до фазы 3d.
    anchor: Literal["masks", "frame"] = "masks"
    black: float = Field(0.5, ge=0, lt=100, description="процентиль фона → 0")
    white: float = Field(99.8, gt=0, le=100, description="процентиль ярких деталей → 1")
    a: float = Field(0.02, gt=0, description="сила растяжки: меньше — сильнее")
    # Переопределения для отдельных фильтров: {F187N: {white: 99.9}}; неуказанное берётся из общих полей.
    per_filter: dict[str, dict[str, float]] = {}

    @model_validator(mode="before")
    @classmethod
    def _default_section(cls, data: object) -> object:
        """stretch.default: {black, white, a} — то же, что общие поля stretch, просто нагляднее рядом с per_filter."""
        if isinstance(data, dict) and "default" in data:
            data = dict(data)
            default = data.pop("default") or {}
            both = sorted(set(default) & set(data) - {"per_filter"})
            if both:
                raise ValueError(f"stretch: {both} заданы и в default, и рядом — оставь одно место")
            data.update(default)
        return data

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


class ShadowsConfig(_Strict):
    # Подъём теней после смешивания: asinh(k·яркость)/asinh(k), оттенок не меняется.
    # БОЛЬШЕ — СИЛЬНЕЕ, 0 — выкл. (Прежний lift: value = 1/lift, lift 0.25 → 4.)
    value: float = Field(0.0, ge=0)
    # all — по всему кадру; nebula — только внутри маски туманности (этап clean), фон остаётся на чёрной точке.
    scope: Literal["all", "nebula"] = "all"
    feather_px: float = Field(20.0, gt=0, description="ширина мягкого края маски, px сетки")


class ColorConfig(_Strict):
    mode: Literal["wavelength", "manual"] = "wavelength"
    # Оттенок (градусы HSV) для самой короткой и самой длинной волны: 240 синий, 0 красный.
    hue_range: tuple[float, float] = (240.0, 0.0)
    # average — сбалансированная сумма (все слои по 1 дают белый); screen — как в Photoshop,
    # ярче, но при 4+ фильтрах сводит туманность в белый и жёлтый.
    blend: Literal["average", "screen"] = "average"
    saturation: float = Field(1.0, ge=0)
    # Подъём теней: число (= value) или {value, scope, feather_px}.
    shadows: ShadowsConfig = ShadowsConfig()
    # Насыщенность звёзд поля внутри маски звёзд (stars.*): 1 — как есть, 0 — серые с той же яркостью.
    star_saturation: float = Field(1.0, ge=0, le=1)
    overrides: dict[str, str] = {}
    weights: dict[str, float] = {}
    # Нейтральный фон: чёрная точка по самым тёмным пикселям, затем медианы R, G, B по маске пустого
    # неба (не туманность и не звёзды, плюс sky_grow_px) приводятся к общему уровню — среднему из трёх.
    neutral_background: bool = True
    background_percentile: float = Field(10.0, gt=0, lt=100)
    sky_grow_px: int = Field(10, ge=0, description="отступ маски пустого неба от туманности и звёзд")
    # Поправка каналов применяется с весом: 1 вне маски туманности, 0 внутри, мягкий край sky_feather_px.
    sky_feather_px: float = Field(20.0, gt=0, description="мягкий край маски туманности для поправки каналов, px")

    @model_validator(mode="before")
    @classmethod
    def _renamed(cls, data: object) -> object:
        if isinstance(data, dict) and "lift" in data:
            lift = data["lift"]
            hint = f"shadows: {1 / lift:g}" if lift else "shadows: 0"
            raise ValueError(f"color.lift переименован в color.shadows с обратной шкалой (больше — сильнее): "
                             f"вместо lift: {lift} напиши {hint}")
        if isinstance(data, dict) and isinstance(data.get("shadows"), (int, float)):
            data = {**data, "shadows": {"value": data["shadows"]}}
        return data

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
    clean: CleanConfig = CleanConfig()
    stars: StarsConfig = StarsConfig()
    background: BackgroundConfig = BackgroundConfig()
    stretch: StretchConfig = StretchConfig()
    color: ColorConfig = ColorConfig()
    finish: FinishConfig = FinishConfig()
    preview: PreviewConfig = PreviewConfig()
    sweep: SweepConfig = SweepConfig()
    report: ReportConfig = ReportConfig()
    credit: str = "NASA, ESA, CSA, STScI (MAST)"
    author: str = "Denis"
    # Картинка для сравнения (пресс-релиз): в sweep и в отчёте по стилям — в соседней клетке.
    reference_image: str | None = None
    # Имя стиля (файл в configs/<объект>/); задаётся загрузчиком, не пишется руками.
    style: str | None = None

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
    def load(cls, path: str | Path, style: str | None = None) -> Config:
        raw = load_raw(Path(path))
        if style:
            raw["style"] = style
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json")
        return yaml.safe_dump(data, allow_unicode=True, sort_keys=False)

    def digest(self, *sections: str) -> str:
        """Хеш обработки (или выбранных секций) — в метаданные файла и ключи кеша."""
        data = self.model_dump(mode="json")
        if sections:
            data = {k: data[k] for k in sections}
        else:
            data = {k: v for k, v in data.items() if k not in NON_PROCESSING}
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


# Поля, которые не влияют на пиксели: в хеш обработки не входят, иначе одна и та же картинка
# из плоского конфига и из стиля measured дала бы разные файлы (хеш пишется в метаданные).
NON_PROCESSING = ("style", "reference_image", "sweep", "preview", "report")


def deep_merge(base: dict, over: dict) -> dict:
    """Словари сливаются в глубину, всё остальное (числа, строки, списки) заменяется целиком."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_raw(path: Path, _chain: tuple[Path, ...] = ()) -> dict:
    """YAML с наследованием: `extends: base` — файл base.yaml рядом; цепочки разрешены, циклы — нет."""
    path = path.resolve()
    if path in _chain:
        names = " → ".join(p.stem for p in (*_chain, path))
        raise ValueError(f"конфиги наследуют друг друга по кругу: {names}")
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    parent = raw.pop("extends", None)
    if not parent:
        return raw
    parent_path = path.parent / (parent if str(parent).endswith((".yaml", ".yml")) else f"{parent}.yaml")
    if not parent_path.is_file():
        raise ValueError(f"{path.name}: extends: {parent} — нет файла {parent_path}")
    return deep_merge(load_raw(parent_path, (*_chain, path)), raw)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", name.lower())
    if not slug:
        raise ValueError(f"не получается сделать имя папки из '{name}', задай slug в конфиге")
    return slug
