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

## По шагам

| Команда | Что делает | Что смотреть |
| --- | --- | --- |
| `skyforge fetch "NGC 3132" --list` | что есть в архиве: фильтры, экспозиции, эксклюзив | таблица в консоли |
| `skyforge fetch ngc3132` | скачать фильтры из конфига в `data/raw/` | — |
| `skyforge inspect ngc3132` | таблица и превью сырых кадров | `inspect_sheet.png` |
| `skyforge align ngc3132` | общая сетка, север вверх, обрезка | `align_check.png`, `align_zoom.png` |
| `skyforge clean ngc3132` | ядра звёзд, пьедесталы плиток мозаики, градиенты фона | `background_model.png`, `star_cores.png` |
| `skyforge stretch ngc3132` | фон → 0, яркие детали → 1, asinh | `stretch_hist.png` |
| `skyforge compose ngc3132` | цвет по длине волны, смешивание | `compose.png` |
| `skyforge finish ngc3132` | денойз, резкость, TIFF/PNG | `finish_zoom.png` |
| `skyforge sweep ngc3132` | 9 вариантов растяжки на одном листе | `sweep.png` |
| `skyforge sweep ngc3132 --x color.weights.F212N=1,0.7,0.5,0.3` | одна строка вариантов по одному параметру | `sweep.png` |
| `skyforge report` | пересобрать HTML для последнего запуска | `report.html` |
| `skyforge cache list` / `prune` | место на диске, чистка `data/work/` | — |

Отдельные команды пишут в `out/<объект>/current/`, `run` — в новую папку с датой. Вместо `ngc3132` можно указать путь к любому YAML.

Без сети: положи FITS в `data/inbox/<объект>/` (например `data/inbox/ngc3132/`), фильтр определится по заголовку, и `skyforge run ngc3132 --offline` возьмёт их.

## Тесты

```
uv run pytest
```

Синтетическое небо из трёх фильтров с разной сеткой: проверяется, что после совмещения звёзды стоят там, где их предсказывает WCS, что два прогона дают байт-в-байт одинаковые файлы и что сырые файлы не меняются.
