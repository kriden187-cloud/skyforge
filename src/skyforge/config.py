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
    # intersection — кадр обрезается до поля, где есть все фильтры; union — объединение покрытий: сетка
    # опорного кадра расширяется на целое число пикселей, где фильтра нет — NaN, цвет из имеющихся.
    crop: Literal["intersection", "union"] = "intersection"
    # union: группы фильтров (обычно по каналам R, G, B) — кадр обрезается до прямоугольника, где в каждой
    # группе есть хотя бы один фильтр; пусто — до рамки вокруг всех данных.
    channel_groups: list[list[str]] = []
    channel_inset_px: int = Field(8, ge=0, description="отступ внутрь от границы покрытия каждой группы, px при bin 1")
    # union: на краях кадра есть все фильтры (полоса, где одного не хватает, — свой баланс цвета, срезается)
    require_all: bool = False
    # union, source: level2: на краях кадра у каждого коротковолнового фильтра (< 2,5 мкм) не меньше стольких
    # экспозиций (слой NEXP своей сборки); 0 — не проверять. Граница числа экспозиций — ступенька шума и цвета.
    min_exposures_sw: int = Field(0, ge=0)
    # Доля непустых пикселей на краю обрезки: 1.0 — ни одного NaN на границе.
    crop_min_coverage: float = Field(0.99, gt=0, le=1)
    # 3b (Hubble): trim — обрезка как выше; target — наибольший прямоугольник без пустот, внутри которого лежит сам
    # объект (L-образное поле WFPC2, мозаики с вырезами); координаты — coords или заголовок файла.
    crop_mode: Literal["trim", "target"] = "trim"
    # Повернуть сетку так, чтобы рамка вокруг реальных данных была наименьшей. Важнее finish.north_up.
    auto_rotate: bool = False


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


class CosmicsConfig(_Strict):
    # 3b: следы космических лучей, которые не убрало сложение (у Hubble — одиночные экспозиции).
    # Луч — резкий выброс, которого нет в остальных фильтрах на том же месте неба.
    clean: bool = False
    sigma: float = Field(5.0, gt=0, description="выброс над мелкой структурой, в σ шума")
    sharpness: float = Field(2.0, gt=0, description="во сколько раз выброс резче окружающей структуры")
    others_sigma: float = Field(3.0, gt=0, description="порог «источник есть» в остальных фильтрах, в σ")
    protect_arcsec: float = Field(0.15, ge=0, description="допуск совмещения при сверке с остальными фильтрами")
    grow_px: int = Field(1, ge=0, description="расширить след на столько пикселей")


class GapsConfig(_Strict):
    # 3b: тонкие пустые полосы внутри поля — зазоры между чипами WFPC2 в одиночной экспозиции.
    fill: bool = False
    max_width_arcsec: float = Field(0.5, gt=0, description="шире — не зазор, а край поля")


class CleanConfig(_Strict):
    destripe: DestripeConfig = DestripeConfig()
    cosmics: CosmicsConfig = CosmicsConfig()
    gaps: GapsConfig = GapsConfig()


class ReportCrop(_Strict):
    # Центр кропа: x, y — на итоговой картинке при bin 1 (x вправо, y вниз, как в просмотрщике) или
    # ra, dec (градусы) — на небе: такие кропы при любой смене сетки (bin, union, другая обрезка)
    # пересчитываются в пиксели автоматически и показывают то же место.
    name: str
    x: int | None = None
    y: int | None = None
    ra: float | None = None
    dec: float | None = None
    size: int = Field(400, ge=16)

    @model_validator(mode="after")
    def _center(self) -> "ReportCrop":
        if (self.ra is None) != (self.dec is None) or (self.x is None) != (self.y is None) or \
                (self.ra is None and self.x is None):
            raise ValueError(f"report.crops.{self.name}: нужен центр — x и y или ra и dec")
        return self


class ReportConfig(_Strict):
    crops: list[ReportCrop] = []
    # Принятый результат для сравнения: data/baselines/<slug>/<baseline>/<стиль>_bin<N>.png
    # (сохраняется командой skyforge baseline). Отчёт строит карту разности, фактуру и кропы рядом.
    baseline: str | None = None
    texture_ring: tuple[int, int] = (30, 200)  # кольцо вокруг центральной звезды (кроп core), px при bin 1


class Level2Config(_Strict):
    """Своя мозаика из отдельных экспозиций уровня 2 (_cal.fits) — source: level2."""
    # destripe в системе детектора: строки — по каждому усилителю (полосы по 512 столбцов), затем столбцы.
    amplifiers: int = Field(4, ge=1)
    mask_grow_px: int = Field(5, ge=0, description="расширение маски туманности и звёзд из i2d в кадре экспозиции")
    min_pixels: int = Field(50, ge=1, description="меньше свободных пикселей в линии — значение от соседних линий")
    min_free_fraction: float = Field(0.3, ge=0, le=1, description="доля свободных пикселей линии")
    # Окно скользящей медианы, как в 3d: вычитается быстрая часть (полосы), плавный ореол остаётся.
    window: int = Field(201, ge=0)
    columns: bool = True
    pixel_sigma: float = Field(3.0, gt=0, description="яркие пиксели (лучи) — в σ шума, не идут в медианы")
    # DQ: какие биты значат «плохой пиксель» (1 — DO_NOT_USE; он ставится и для насыщенных и отбракованных).
    dq_bad: int = Field(1, ge=0)
    # Сборка: взвешенное среднее (веса 1/ERR²) с отбрасыванием выбросов по стеку.
    clip_sigma: float = Field(4.0, gt=0, description="выброс: |x − медиана стека| > clip·ERR + наклон·|∇медианы|")
    clip_gradient: float = Field(1.0, ge=0, description="допуск на крутизну (края звёзд при разной выборке пикселей)")
    match_levels: bool = True  # одна константа фона на экспозицию по парам перекрытий (МНК)
    match_per_amp: bool = False  # константа на экспозицию и усилитель (4 полосы по 512 столбцов), если крест остаётся
    # Сколько фильтров собирать одновременно (отдельные процессы). Результат тот же, в ключ кеша не входит.
    # Память: фильтр коротких волн — до ~5 ГБ.
    workers: int = Field(1, ge=1, le=16)
    # 3k: волна фона вдоль оси y детектора — общая для усилителей (шум считывания, лист 17): одна кривая на
    # экспозицию, ломаная с узлами каждые curve_knot_rows строк детектора. Один МНК по перекрытиям со всеми соседями
    # вместе с константами на усилитель, по всему свободному небу (общий ореол при сравнении с соседями сокращается);
    # кривые стянуты к нулю (априорный разброс curve_prior, σ неба стека), средняя кривая экспозиции — ноль (иначе
    # путается с константами), средняя по фильтру в каждом узле — ноль. Затем константы сводятся как раньше.
    match_curve_y: bool = False
    curve_knot_rows: int = Field(256, ge=16, description="шаг узлов кривой, строк детектора")
    curve_prior: float = Field(1.0, gt=0, description="σ неба стека: меньше — сильнее стягивает кривые к нулю")
    curve_bin_px: int = Field(64, ge=8, description="полоса строк сетки для медианы разности в перекрытии")
    # Веса стека: rnoise — 1/VAR_RNOISE (шум считывания, как ivm у drizzle); err — 1/ERR² (в ERR есть пуассоновский
    # шум самой звезды: в ядре яркий отсчёт получает меньший вес, и поток звёзд занижается — у F090W на 2%).
    weights: Literal["rnoise", "err"] = "rnoise"
    # Астрометрическая подстройка экспозиций по звёздам i2d (аналог tweakreg). Правила 3g (решение Дениса):
    # исправлять, если точность сдвига (остаток/√N) < tweak_max_precision; поворот — только при ≥ tweak_rot_min_stars
    # звёздах и |поворот| ≥ tweak_rot_min_deg, иначе только сдвиг. Короткие волны: детекторы одной экспозиции
    # сдвигаются вместе — детектор, отличающийся от медианы группы больше tweak_group_tol, получает медиану.
    # Длинные волны (без соседей): точность ≥ порога или остаток после > tweak_lw_max_residual — не исправлять;
    # если при этом остаток до > tweak_exclude_above — экспозиция исключается из стека.
    tweak: bool = True
    tweak_min_stars: int = Field(8, ge=3)
    tweak_max_stars: int = Field(30, ge=3)
    tweak_max_precision: float = Field(0.03, gt=0, description="px: точность сдвига rms/√N, ниже которой исправлять")
    tweak_rot_min_stars: int = Field(20, ge=4)
    tweak_rot_min_deg: float = Field(0.002, ge=0)
    tweak_group_tol: float = Field(0.05, gt=0, description="px: отличие детектора от медианы своей экспозиции")
    tweak_lw_max_residual: float = Field(0.2, gt=0, description="px: длинные волны — остаток после, выше которого не исправлять")
    tweak_exclude_above: float = Field(0.3, gt=0, description="px: остаток до у неисправленной — исключить из стека")
    overlap_block_px: int = Field(64, ge=8, description="блок для метрики и карты остатков в перекрытиях")
    # Допуск FWHM звёзд к i2d (приёмка 3g): у недодискретизированных фильтров (медиана FWHM i2d < star_fwhm_small_px)
    # гауссова подгонка FWHM чувствительна к субпиксельной фазе — допуск шире. На сборку не влияют.
    star_fwhm_small_px: float = Field(2.5, gt=0)
    star_fwhm_tol_small_percent: float = Field(8.0, gt=0)
    star_fwhm_tol_percent: float = Field(5.0, gt=0)


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
    # Звёзды с насыщенным ядром (3i): дыра насыщения не меньше saturated_min_core_px хотя бы в saturated_min_filters
    # фильтрах. Для NGC 3132 — центральная звезда, звезда внизу справа и звезда в оболочке слева.
    saturated_min_core_px: int = Field(50, ge=1)
    saturated_min_filters: int = Field(2, ge=1)

    def mask_key(self) -> dict:
        """Поля, от которых зависят маски для destripe и сборки уровня 2 (без поиска звёзд с насыщенным ядром —
        он на этапе clean и в ключи build/align не входит)."""
        return self.model_dump(exclude={"saturated_min_core_px", "saturated_min_filters"})


class BackgroundConfig(_Strict):
    subtract: bool = False
    # photutils — модель Background2D (как в 3d); blocks — свои блоки по свободному небу, внутри туманности
    # модель продолжается от соседних блоков, а не меряется (3g, для своей мозаики уровня 2).
    model: Literal["photutils", "blocks"] = "photutils"
    blocks_px: int = Field(256, ge=16, description="блок модели blocks")
    blocks_mask_grow_px: int = Field(10, ge=0, description="расширение маски туманности и звёзд для модели blocks")
    blocks_min_free: float = Field(0.5, gt=0, le=1, description="доля свободных пикселей, чтобы блок мерился")
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
    # constant — уровень на область (принято в 3d); plane — плоскость a + b·x + c·y (опыт 3e: ступеньки
    # не снимает, на областях со свободным краем наклон улетает — см. docs/PLAN.md).
    tile_model: Literal["constant", "plane"] = "constant"
    tile_min_extent_arcsec: float = Field(4.0, gt=0, description="свободное небо области по каждой оси для своего наклона")
    # Своя медиана — только если свободного неба достаточно и по площади, и по доле области: узкая полоса
    # дизеринга у ореола, где свободен край, иначе получит уровень этого края, а не свой.
    tile_min_free_fraction: float = Field(0.3, ge=0, le=1, description="доля свободных пикселей для своей медианы")
    block_px: int = Field(64, ge=8, description="блок для метрики остаточных ступенек фона (медианы блоков)")
    # 3k: константа на каждую связную область одинакового числа экспозиций (слой NEXP своей сборки) — по
    # скачкам на границах (пары пикселей до nexp_band_px по обе стороны, продолжение к границе, как сшивка 3d).
    # Опора — самая большая область (поправка 0); туманность получает константу своей области.
    nexp_levels: bool = False
    nexp_band_px: int = Field(30, ge=3, description="полоса по обе стороны границы областей NEXP для скачка")
    nexp_smooth_px: int = Field(15, ge=1, description="медиана по NEXP до разметки областей (крапины отбраковки)")
    nexp_min_area_px: int = Field(2000, ge=0, description="меньше — клочок, поправка от ближайшей области")
    # Шаг 2: гладкая модель по блокам — остаточные градиенты.
    smooth: bool = True
    # Значения по умолчанию подобраны на NGC 3132: блок 4" и маска 1.5" съедали ореол оболочки.
    box_arcsec: float = Field(8.0, gt=0, description="размер блока сетки фона")
    filter_size: int = Field(3, ge=1, description="медианное сглаживание сетки, в блоках (нечётное)")
    mask_sigma: float = Field(1.5, gt=0, description="порог маски источников над фоном, в σ")
    mask_smooth_arcsec: float = Field(1.0, ge=0, description="сглаживание перед порогом: ловит слабую туманность")
    mask_dilate_arcsec: float = Field(3.0, ge=0, description="расширение маски вокруг источников")
    min_free: float = Field(0.3, gt=0, le=1, description="доля свободных от маски пикселей, чтобы блок считался")
    # Маска туманности (3h): ищется по кадру, где звёзды заменены местной медианой фона, расширяется диском.
    # Квадратное расширение склеивало яркие звёзды со спайками и соседей в «туманность» — прямоугольные окна,
    # в которых не работали поправка цвета неба, shadows и растяжка.
    nebula_dilate_arcsec: float | None = Field(None, ge=0, description="радиус диска; None — mask_dilate_arcsec")
    nebula_star_fraction: float = Field(0.5, gt=0, le=1, description="компонента, где звёзд больше этой доли, — не туманность")
    nebula_drop_saturated: bool = True  # компоненты отдельно от главной с насыщенным ядром внутри — яркая звезда, не туманность

    @field_validator("filter_size")
    @classmethod
    def _odd(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("background.filter_size должен быть нечётным")
        return v


class SaturatedStarsConfig(_Strict):
    # 3i: звёзды с насыщенным ядром (clean) отдельно от туманности до растяжки — skyforge.satstars.
    enabled: bool = False
    r_in_px: float = Field(100.0, gt=0, description="слой звезды целиком до этого радиуса, px при bin 1")
    r_out_px: float = Field(150.0, gt=0, description="к этому радиусу слой плавно уходит в ноль, px при bin 1")
    profile_percentile: float = Field(30.0, gt=0, lt=100, description="процентиль по кольцу для профиля звезды")
    color_ring_px: tuple[float, float] = (20.0, 60.0)  # кольцо, по которому берётся цвет звезды (вне заполненных ядер)
    a: float = Field(0.01, gt=0, description="asinh слоя звезды: белая точка — ядро")
    # Внутри core_r_px туманность не восстанавливается: она ровная, с уровнем кольца снаружи, звезда — всё выше
    # неё (сумма фильтров, один цвет). 30 px: у центральной звезды NGC 3132 разброс профиля по кольцу падает
    # ниже половины белой точки туманности на 30 px, у двух других — на 16 px. 0 — только радиальная модель.
    core_r_px: float = Field(30.0, ge=0)
    core_feather_px: float = Field(10.0, gt=0, description="переход от ядра к радиальной модели, px")


class StretchConfig(_Strict):
    method: Literal["asinh", "log", "sqrt", "linear"] = "asinh"
    # masks — black считается по маске пустого неба, white — по маске туманности (этап clean):
    # очистка фона и обрезка кадра не двигают растяжку. frame — по всему кадру, как до фазы 3d.
    anchor: Literal["masks", "frame"] = "masks"
    black: float = Field(0.5, ge=0, lt=100, description="процентиль фона → 0")
    # 3k: pedestal — чёрная точка = медиана неба − black_k·σ неба самых шумных мест (98-й процентиль σ по блокам
    # noise_block_px): небо лежит на пьедестале, шум симметричен и не обрезается в 0 (обрезка шума разной силы
    # давала разный цвет неба по областям числа экспозиций). percentile — процентиль black, как раньше.
    black_mode: Literal["percentile", "pedestal"] = "percentile"
    black_k: float = Field(3.0, gt=0)
    noise_block_px: int = Field(128, ge=16)
    # 3k: носок — медиана неба после растяжки ставится на sky_level (уровни из 255), всё ниже сжимается в 0…sky_level
    # плавной монотонной кривой без обрезки. Носок — в данных до asinh: выше неба отображение линейно по данным
    # (как обычная чёрная точка), туманность не темнеет. None — выкл. Только при anchor masks.
    sky_level: float | None = Field(None, gt=0, lt=255)
    # 3k п.3: гладкая модель крупного хода неба для показа — полином порядка sky_poly_order по x и y на фильтр,
    # подгонка по пустому небу вне файла sky_poly_exclude (туманность с кольцом ореола и круг у центра, FITS с WCS),
    # вычитается до растяжки (skyforge.skymodel). 0 — выкл.
    sky_poly_order: int = Field(0, ge=0, le=4)
    sky_poly_exclude: str | None = None
    # поправка гасится к туманности: вес 1 дальше sky_poly_taper_px от маски туманности clean, 0 на её краю и внутри
    # (полином, продолженный под туманность, иначе менял её яркость: 0,004 при допуске 0,003). 0 — без гашения.
    sky_poly_taper_px: float = Field(0.0, ge=0)
    white: float = Field(99.8, gt=0, le=100, description="процентиль ярких деталей → 1")
    a: float = Field(0.02, gt=0, description="сила растяжки: меньше — сильнее")
    # Переопределения для отдельных фильтров: {F187N: {white: 99.9}}; неуказанное берётся из общих полей.
    per_filter: dict[str, dict[str, float]] = {}
    # 3i: в выборку для белой точки по туманности не идут звёзды — маска звёзд clean и диски exclude_radius_px
    # вокруг звёзд с насыщенным ядром (их крылья в синих фильтрах в сотни раз ярче туманности рядом).
    exclude_stars: bool = True
    saturated_stars: SaturatedStarsConfig = SaturatedStarsConfig()

    exclude_radius_px: float = Field(150.0, ge=0, description="радиус диска вокруг звезды с насыщенным ядром, px при bin 1")

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
    # luma — множитель по яркости, пиксель с каналом за 1 делится целиком (канал ровно 1: в яркой цветной
    # области рисунок стирается); max — множитель по самому яркому каналу: канал за 1 не выходит (3i).
    by: Literal["luma", "max"] = "luma"
    # all — по всему кадру; nebula — только внутри маски туманности (этап clean), фон остаётся на чёрной точке.
    scope: Literal["all", "nebula"] = "all"
    feather_px: float = Field(20.0, gt=0, description="ширина мягкого края маски, px сетки")
    # scope nebula: подъём за маской туманности (внешний ореол, 3j) — своей силой; внутри маски — value, как раньше.
    outside: float = Field(0.0, ge=0, description="сила подъёма за маской туманности, 0 — выкл")
    # Подъём за маской — с весом от самого сигнала (3j): сглаженная яркость (звёзды вырезаны) над уровнем неба,
    # делённая на halo_k·σ сглаженного неба, в 0–1. Где сигнал на уровне неба (и его ступенек), подъёма нет;
    # без новой границы. halo_k 0 — подъём за маской ровный.
    halo_k: float = Field(0.0, ge=0)
    halo_smooth_px: float = Field(30.0, gt=0, description="сглаживание яркости для веса ореола, px")
    # 3k: подъём сигнала над небом — уровень неба stretch.sky_level вычитается, сигнал поднимается, уровень
    # возвращается: небо внутри маски и вне её на одном уровне, контур маски не проступает.
    above_sky: bool = False


class StarChromaConfig(_Strict):
    # 3j: цвет звёзд с насыщенным ядром (clean): яркость пикселя (самый яркий канал) не меняется, цвет (доли
    # каналов) сглаживается по соседям — пёстрая крупа на лучах от разной дифракции фильтров уходит; белые
    # обрезанные пиксели ядра цвета не имеют, в расчёт не идут и получают цвет кольца крыльев.
    enabled: bool = False
    r_in_px: float = Field(80.0, gt=0, description="до этого радиуса от звезды — целиком")
    r_out_px: float = Field(140.0, gt=0, description="к этому радиусу — плавно в ноль")
    sigma_px: float = Field(5.0, gt=0, description="сглаживание цвета, px")
    core_sigma_px: float = Field(15.0, gt=0, description="сглаживание цвета для обрезанного ядра, px")
    # Только звёзды вне маски туманности: у звёзд внутри неё лучи отличаются от туманности цветом, а не яркостью,
    # и сглаженный цвет растворяет звезду в туманности (3j).
    outside_nebula_only: bool = True


class ColorConfig(_Strict):
    mode: Literal["wavelength", "manual"] = "wavelength"
    # Оттенок (градусы HSV) для самой короткой и самой длинной волны: 240 синий, 0 красный.
    hue_range: tuple[float, float] = (240.0, 0.0)
    # average — сбалансированная сумма (все слои по 1 дают белый); screen — как в Photoshop,
    # ярче, но при 4+ фильтрах сводит туманность в белый и жёлтый.
    blend: Literal["average", "screen"] = "average"
    saturation: float = Field(1.0, ge=0)
    # clip — прирост цветности и обрезка в 0–1; limit — прирост ограничивается в пикселе так, чтобы ни один
    # канал не вышел за 0–1 (оттенок и рисунок в ярких цветных областях сохраняются, 3i).
    saturation_gamut: Literal["clip", "limit"] = "clip"
    star_chroma: StarChromaConfig = StarChromaConfig()
    # Подъём теней: число (= value) или {value, scope, feather_px}.
    shadows: ShadowsConfig = ShadowsConfig()
    # Насыщенность звёзд поля внутри маски звёзд (stars.*): 1 — как есть, 0 — серые с той же яркостью.
    star_saturation: float = Field(1.0, ge=0, le=1)
    overrides: dict[str, str] = {}
    weights: dict[str, float] = {}
    # Нейтральный фон: чёрная точка по самым тёмным пикселям, затем цвет пустого неба (не туманность и не
    # звёзды, плюс sky_grow_px) к серому. map — по блокам background_map_px: в каждом блоке медианы R, G, B
    # приводятся к их среднему, карта гладкая и продолжена под маску от соседей; поправка только цветовая
    # (сумма каналов в пикселе не меняется). constant — одна поправка на кадр (3d–3g). off — ничего.
    neutral_background: Literal["map", "constant", "off"] = "map"
    background_map_px: int = Field(256, ge=16, description="блок цветовой карты фона, px")
    background_map_min_sky: float = Field(0.3, gt=0, le=1, description="доля пустого неба, чтобы блок мерился")
    background_percentile: float = Field(10.0, gt=0, lt=100)
    # 3k: True — нейтрализация снимает только цветовой сдвиг фона между каналами, уровень неба (пьедестал из
    # растяжки) не опускается в 0. False — фон каждого канала в 0, как раньше.
    background_keep_level: bool = False
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

    @field_validator("neutral_background", mode="before")
    @classmethod
    def _neutral_bool(cls, v: object) -> object:
        # до 3h было true/false: true — одна поправка на кадр
        if v is True:
            return "constant"
        if v is False:
            return "off"
        return v

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


class FinishStarsConfig(_Strict):
    # Разделение звёзд и туманности (3i): под маской звёзд (clean, расширенной с мягким краем feather_px)
    # слой «без звёзд» заполняется местной медианой; звёзды = кадр − без звёзд. scale 1, shrink 0 — как раньше.
    scale: float = Field(1.0, ge=0, le=1, description="яркость слоя звёзд")
    shrink: float = Field(0.0, ge=0, description="радиус уменьшения ядра (минимум по диску), px")
    feather_px: float = Field(6.0, gt=0, description="мягкий край маски звёзд, px")
    fill_window_px: int = Field(48, ge=8, description="окно местной медианы под звездой, px")


class LocalContrastConfig(_Strict):
    # Локальный контраст слоя без звёзд внутри маски туманности (мягкий край feather_px, вне маски — ноль):
    # самый яркий канал + amount·(он − размытая копия с радиусом radius), подъём плавно сходится к 1.
    radius: float = Field(80.0, gt=0, description="радиус размытой копии (σ гаусса), px")
    amount: float = Field(0.0, ge=0, description="сила, 0 — выкл")
    feather_px: float = Field(30.0, gt=0)


class FinishConfig(_Strict):
    denoise: float = Field(0.0, ge=0, le=1, description="0 — выкл, 1 — максимум")
    denoise_iters: int = Field(100, ge=1)
    sharpen: float = Field(0.0, ge=0, description="сила нерезкой маски, 0 — выкл")
    sharpen_radius: float = Field(2.0, gt=0, description="радиус нерезкой маски в пикселях")
    stars: FinishStarsConfig = FinishStarsConfig()
    local_contrast: LocalContrastConfig = LocalContrastConfig()
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
    # level3 — готовые мозаики i2d/drc; level2 — своя сборка из отдельных экспозиций (_cal.fits, только JWST).
    source: Literal["level3", "level2"] = "level3"
    level2: Level2Config = Level2Config()
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
        # В словарях по фильтрам (color.weights.F212N, stretch.per_filter.F090W.white) ключа может ещё не быть.
        per_filter = len(keys) >= 3 and keys[-3] == "per_filter"
        for i, k in enumerate(keys[:-1]):
            if per_filter and i == len(keys) - 2 and k not in node:
                node[k] = {}
            if k not in node or not isinstance(node[k], dict):
                raise ValueError(f"нет такого параметра: {dotted}")
            node = node[k]
        open_dict = (len(keys) >= 2 and keys[-2] == "weights") or per_filter
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
