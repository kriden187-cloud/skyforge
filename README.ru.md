# Skyforge

Конвейер, который превращает открытые научные данные телескопов JWST и Hubble из архива MAST в цветную картинку —
воспроизводимо: каждый результат задаётся YAML-конфигом, повторный прогон даёт тот же файл байт в байт.

English: [README.md](README.md).

![Sh2-305, JWST NIRCam](docs/gallery/sh2305_showcase.jpg)

## Что внутри

- **Загрузка** из MAST (`skyforge fetch`), совмещение фильтров на общей сетке (`align`), очистка — ядра насыщенных звёзд,
  швы и следы космических лучей (`clean`), растяжка с опорой на пустое небо (`stretch`), смешивание в цвет (`compose`),
  финиш и экспорт 16-битного TIFF и PNG с метаданными (`finish`). Весь путь — `skyforge run <объект> -s <стиль>`.
- **Поиск объектов** (`skyforge discover`): недавно открытые наблюдения NIRCam, сверка с пресс-релизами ESA/Webb и NASA/STScI.
- **Публикация** (`skyforge publish`, `skyforge gallery`): мастер PNG/TIFF принятого результата, версии под площадки,
  подпись на русском и английском, галерея конфигов `docs/GALLERY.ru.md` (английская — `docs/GALLERY.md`).
- **Приёмка по картинкам**: каждый этап пишет превью и отчёт; принятые результаты хранятся как baseline, отчёт
  показывает разность с ними.

## Объекты

| Объект | Данные | Конфиг | Тег |
|---|---|---|---|
| NGC 3132 (Южное кольцо) | JWST NIRCam, программа 2733 (ERO), 6 фильтров, сборка из экспозиций уровня 2 | `configs/ngc3132/` (стили measured и showcase) | `phase-3k` |
| NGC 3132 | Hubble WFPC2 1995, программа 6221, [O III], Hα, [N II] | `configs/ngc3132hst/` | `phase-3b` |
| Sh2-305 | JWST NIRCam, программа 4547, F182M и 4 длинноволновых фильтра | `configs/sh2305/` | `phase-4` |
| W3 (Westerhout 3) | JWST NIRCam, программа 5437, поле W3-JWST-2: F140M, F182M, F210M и 5 длинноволновых, оба модуля детектора | `configs/w3_2/` | `phase-6` |

Превью, подписи и команды воспроизведения — в [docs/GALLERY.ru.md](docs/GALLERY.ru.md). План проекта — [docs/PLAN.md](docs/PLAN.md),
история решений и ограничения — [docs/HANDOFF.md](docs/HANDOFF.md). Краткий обзор на английском —
[docs/OVERVIEW.en.md](docs/OVERVIEW.en.md), ключи YAML — [configs/README.md](configs/README.md) (на английском).

## Быстрый старт

```bash
uv sync
uv run skyforge --help
uv run pytest
uv run skyforge run sh2305 -s showcase --bin 4      # если данных нет на диске, скачает из MAST (~45 ГБ для Sh2-305)
uv run skyforge publish sh2305 -s showcase
```

Данные (`data/`) и результаты (`out/`) в git не хранятся, кроме референсов и листов приёмки. Если MAST не открывается
напрямую, прокси задаётся в `.env` (пример — `.env.example`).

## Стили

Один объект — одни данные, несколько обработок. В `configs/<объект>/` лежит `base.yaml` (объект, фильтры, очистка, маски) и стили, которые меняют только нужное:

```yaml
# configs/ngc3132/showcase.yaml
extends: measured          # или base; словари сливаются в глубину
stretch:
  per_filter: { F470N: { white: 99.95, a: 0.05 } }
color:
  shadows: { value: 4, scope: nebula }   # подъём теней только в маске туманности
```

```
uv run skyforge run ngc3132 --style measured        # один стиль → out/ngc3132/<дата>_measured/
uv run skyforge run ngc3132 --all --bin 1           # все стили → out/ngc3132/<дата>_styles/report.html
uv run skyforge sweep ngc3132 --style showcase --x color.weights.F212N=0.15,0.08   # референс — в левой верхней клетке
```

Тяжёлые этапы (`inspect`, `align`, `clean`) общие и кешируются в `data/work/`: второй стиль подряд считается за секунды. Растяжка, цвет и финиш у каждого стиля свои.

## По шагам

| Команда | Что делает | Что смотреть |
| --- | --- | --- |
| `skyforge fetch "NGC 3132" --list` | что есть в архиве: фильтры, экспозиции, эксклюзив | таблица в консоли |
| `skyforge fetch ngc3132` | скачать фильтры из конфига в `data/raw/`; при `source: level2` — ещё и экспозиции `_cal.fits` в `data/raw/<объект>/l2/` | — |
| `skyforge build ngc3132 --style measured` | `source: level2`: своя мозаика по каждому фильтру из экспозиций — destripe в кадре детектора, сведение уровней, стек на сетке i2d | `l2_destripe.png`, `l2_overlaps.png`, `l2_stars.png` |
| `skyforge inspect ngc3132` | таблица и превью сырых кадров | `inspect_sheet.png` |
| `skyforge align ngc3132` | общая сетка, север вверх, обрезка | `align_check.png`, `align_zoom.png` |
| `skyforge clean ngc3132` | ядра звёзд, пьедесталы плиток мозаики, градиенты фона, маска звёзд | `background_model.png`, `star_cores.png`, `star_mask.png` |
| `skyforge stretch ngc3132` | фон → 0, яркие детали → 1, asinh | `stretch_hist.png` |
| `skyforge compose ngc3132` | цвет по длине волны, смешивание | `compose.png` |
| `skyforge finish ngc3132` | денойз, резкость, TIFF/PNG | `finish_zoom.png` |
| `skyforge sweep ngc3132` | 9 вариантов растяжки на одном листе | `sweep.png` |
| `skyforge sweep ngc3132 --x color.weights.F212N=1,0.7,0.5,0.3` | одна строка вариантов по одному параметру | `sweep.png` |
| `skyforge sweep ngc3132 --x color.star_saturation=1,0.5,0.2 --crop 600,450,500` | то же на кропе 100% (X,Y,сторона) | `sweep.png` |
| `skyforge report` | пересобрать HTML для последнего запуска | `report.html` |
| `skyforge baseline ngc3132 --name phase-3d --from out/ngc3132/<папка>` | сохранить итог как принятый: с ним сравнивает отчёт (`report.baseline`) | `baseline_diff.png`, `crops_compare.png` |
| `skyforge cache list` / `prune` | место на диске, чистка `data/work/` | — |
| `skyforge discover` | недавно открытые наблюдения NIRCam, сверка с пресс-релизами, 10 кандидатов с превью | `candidates.png`, `candidates.md` |
| `skyforge untouched` | сверка короткого списка discover с публичными обработками (поиск в сети, Wikimedia Commons, Flickr); `--recon` — разведка выбранных | `candidates_untouched.md`, `candidates_untouched.png`, `recon/` |
| `skyforge publish sh2305` | мастер PNG/TIFF принятого результата, JPEG web / social / preview, подпись RU/EN | `out/publish/<объект>/<дата>/` |
| `skyforge gallery` | галерея конфигов: превью, подпись, команда воспроизведения | `docs/GALLERY.md`, `docs/GALLERY.ru.md` |

Отдельные команды пишут в `out/<объект>/current/` (у стилей — `current/<стиль>/`), `run` — в новую папку с датой. Для папки стилей нужен `--style`; вместо имени можно указать путь к любому YAML.

Без сети: положи FITS в `data/inbox/<объект>/` (например `data/inbox/ngc3132/`), фильтр определится по заголовку, и `skyforge run ngc3132 --offline` возьмёт их.

## Тесты

```
uv run pytest
```

Синтетическое небо из трёх фильтров с разной сеткой: проверяется, что после совмещения звёзды стоят там, где их предсказывает WCS, что два прогона дают байт-в-байт одинаковые файлы и что сырые файлы не меняются.

## Лицензии и указание источника

- **Код** — MIT (`LICENSE`).
- **Наши картинки** (`out/publish/`, `docs/gallery/`, листы приёмки) — CC BY-SA 4.0, автор Denis Krivosheev.
  Указание источника данных: «NASA, ESA, CSA, STScI» для JWST, «NASA, ESA, STScI» для Hubble; обработка — Denis Krivosheev.
- **Данные** — архив MAST (STScI); на них распространяются условия NASA, ESA, CSA и STScI.
- **Чужие изображения для сравнения** в `data/reference/`: снимок Hubble Heritage 1998 (STScI-1998-39, ESA/Hubble opo9839a,
  CC BY 4.0, Hubble Heritage Team (STScI/AURA/NASA/ESA)); обработки Sh2-305 Мелины Тевено и Марка Маккохрейна
  (Wikimedia Commons, CC BY-SA 4.0) — авторы и ссылки в `data/reference/sh2305/sources.md`.
