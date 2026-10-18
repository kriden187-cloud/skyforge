# skyforge

Консольный конвейер: название объекта → кадры JWST/Hubble из MAST → цветная картинка (PNG + 16-битный TIFF) и HTML-отчёт со всеми промежуточными стадиями. План и архитектура: [docs/PLAN.md](docs/PLAN.md).

## Установка

```
uv sync
copy .env.example .env      # если MAST открывается только через прокси
```

## Быстрый старт

```
uv run skyforge run ngc3132                 # всё целиком, bin 4 из конфига → out/ngc3132/<дата-время>/report.html
uv run skyforge run ngc3132 --bin 1         # финал в полном разрешении
```

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
uv run skyforge sweep ngc3132 --style showcase --x color.weights.F212N=0.15,0.08   # с референсом в первой клетке
```

Тяжёлые этапы (`inspect`, `align`, `clean`) общие и кешируются в `data/work/`: второй стиль подряд считается за секунды. Растяжка, цвет и финиш у каждого стиля свои.

## По шагам

| Команда | Что делает | Что смотреть |
| --- | --- | --- |
| `skyforge fetch "NGC 3132" --list` | что есть в архиве: фильтры, экспозиции, эксклюзив | таблица в консоли |
| `skyforge fetch ngc3132` | скачать фильтры из конфига в `data/raw/` | — |
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
| `skyforge cache list` / `prune` | место на диске, чистка `data/work/` | — |

Отдельные команды пишут в `out/<объект>/current/` (у стилей — `current/<стиль>/`), `run` — в новую папку с датой. Для папки стилей нужен `--style`; вместо имени можно указать путь к любому YAML.

Без сети: положи FITS в `data/inbox/<объект>/` (например `data/inbox/ngc3132/`), фильтр определится по заголовку, и `skyforge run ngc3132 --offline` возьмёт их.

## Тесты

```
uv run pytest
```

Синтетическое небо из трёх фильтров с разной сеткой: проверяется, что после совмещения звёзды стоят там, где их предсказывает WCS, что два прогона дают байт-в-байт одинаковые файлы и что сырые файлы не меняются.
