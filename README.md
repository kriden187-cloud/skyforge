# Skyforge

A pipeline that turns open science data from the JWST and Hubble telescopes in the MAST archive into a colour image,
reproducibly: every result is defined by a YAML config, and a rerun gives the same file byte for byte.

![Sh2-305, JWST NIRCam](docs/gallery/sh2305_showcase.jpg)

## What's inside

- **Download** from MAST (`skyforge fetch`), filter alignment on a common grid (`align`), cleaning of saturated star
  cores, seams and cosmic-ray trails (`clean`), a stretch anchored to empty sky (`stretch`), blending into colour
  (`compose`), finishing and export of a 16-bit TIFF and a PNG with metadata (`finish`). The whole path is
  `skyforge run <object> -s <style>`.
- **Object search** (`skyforge discover`): recently released NIRCam observations, checked against ESA/Webb and
  NASA/STScI press releases.
- **Publishing** (`skyforge publish`, `skyforge gallery`): master PNG/TIFF of the accepted result, versions for
  platforms, a caption in Russian and English, the config gallery `docs/GALLERY.md`.
- **Acceptance by images**: every stage writes previews and a report; accepted results are kept as a baseline, and the
  report shows the difference from it.

How the pipeline is built and what it promises — [docs/OVERVIEW.en.md](docs/OVERVIEW.en.md).

## Objects

| Object | Data | Config | Tag |
|---|---|---|---|
| NGC 3132 (Southern Ring) | JWST NIRCam, program 2733 (ERO), 6 filters, built from level-2 exposures | `configs/ngc3132/` (styles measured and showcase) | `phase-3k` |
| NGC 3132 | Hubble WFPC2 1995, program 6221, [O III], Hα, [N II] | `configs/ngc3132hst/` | `phase-3b` |
| Sh2-305 | JWST NIRCam, program 4547, F182M and 4 long-wavelength filters | `configs/sh2305/` | `phase-4` |

Previews, captions and reproduction commands are in [docs/GALLERY.md](docs/GALLERY.md). The project plan is
[docs/PLAN.md](docs/PLAN.md), the history of decisions and limitations is [docs/HANDOFF.md](docs/HANDOFF.md) (both in
Russian). The YAML keys are explained in [configs/README.md](configs/README.md).

## Quick start

```bash
uv sync
uv run skyforge --help
uv run pytest
uv run skyforge run sh2305 -s showcase --bin 4      # downloads from MAST if the data is not on disk (~45 GB for Sh2-305)
uv run skyforge publish sh2305 -s showcase
```

Data (`data/`) and results (`out/`) are not stored in git, except references and acceptance sheets. If MAST cannot be
reached directly, a proxy is set in `.env` (example: `.env.example`).

## Styles

One object means one set of data and several processings. `configs/<object>/` holds `base.yaml` (object, filters,
cleaning, masks) and styles that change only what they need:

```yaml
# configs/ngc3132/showcase.yaml
extends: measured          # or base; dictionaries are merged deeply
stretch:
  per_filter: { F470N: { white: 99.95, a: 0.05 } }
color:
  shadows: { value: 4, scope: nebula }   # shadow lift only inside the nebula mask
```

```
uv run skyforge run ngc3132 --style measured        # one style → out/ngc3132/<date>_measured/
uv run skyforge run ngc3132 --all --bin 1           # all styles → out/ngc3132/<date>_styles/report.html
uv run skyforge sweep ngc3132 --style showcase --x color.weights.F212N=0.15,0.08   # the reference is in the top-left cell
```

The heavy stages (`inspect`, `align`, `clean`) are shared and cached in `data/work/`: a second style in a row takes
seconds. Stretch, colour and finish are each style's own.

## Step by step

| Command | What it does | What to look at |
| --- | --- | --- |
| `skyforge fetch "NGC 3132" --list` | what the archive has: filters, exposures, exclusive access | table in the console |
| `skyforge fetch ngc3132` | download the filters from the config to `data/raw/`; with `source: level2`, also the `_cal.fits` exposures to `data/raw/<object>/l2/` | — |
| `skyforge build ngc3132 --style measured` | `source: level2`: own mosaic per filter from the exposures — destripe in the detector frame, level matching, stacking on the i2d grid | `l2_destripe.png`, `l2_overlaps.png`, `l2_stars.png` |
| `skyforge inspect ngc3132` | table and previews of the raw frames | `inspect_sheet.png` |
| `skyforge align ngc3132` | common grid, north up, cropping | `align_check.png`, `align_zoom.png` |
| `skyforge clean ngc3132` | star cores, mosaic tile pedestals, background gradients, star mask | `background_model.png`, `star_cores.png`, `star_mask.png` |
| `skyforge stretch ngc3132` | background → 0, bright details → 1, asinh | `stretch_hist.png` |
| `skyforge compose ngc3132` | colour by wavelength, blending | `compose.png` |
| `skyforge finish ngc3132` | denoise, sharpening, TIFF/PNG | `finish_zoom.png` |
| `skyforge sweep ngc3132` | 9 stretch variants on one sheet | `sweep.png` |
| `skyforge sweep ngc3132 --x color.weights.F212N=1,0.7,0.5,0.3` | one row of variants along one parameter | `sweep.png` |
| `skyforge sweep ngc3132 --x color.star_saturation=1,0.5,0.2 --crop 600,450,500` | the same on a 100% crop (X,Y,side) | `sweep.png` |
| `skyforge report` | rebuild the HTML for the latest run | `report.html` |
| `skyforge baseline ngc3132 --name phase-3d --from out/ngc3132/<folder>` | save the result as accepted: the report compares against it (`report.baseline`) | `baseline_diff.png`, `crops_compare.png` |
| `skyforge cache list` / `prune` | disk space, cleanup of `data/work/` | — |
| `skyforge discover` | recently released NIRCam observations, check against press releases, 10 candidates with previews | `candidates.png`, `candidates.md` |
| `skyforge publish sh2305` | master PNG/TIFF of the accepted result, JPEG web / social / preview, RU/EN caption | `out/publish/<object>/<date>/` |
| `skyforge gallery` | config gallery: preview, caption, reproduction command | `docs/GALLERY.md`, `docs/GALLERY.ru.md` |

Individual commands write to `out/<object>/current/` (for styles, `current/<style>/`); `run` writes to a new dated
folder. A folder with styles needs `--style`; instead of a name you can give a path to any YAML.

Without a network: put the FITS files into `data/inbox/<object>/` (for example `data/inbox/ngc3132/`), the filter is
read from the header, and `skyforge run ngc3132 --offline` will use them.

## Tests

```
uv run pytest
```

A synthetic sky of three filters on different grids: the tests check that after alignment the stars sit where the WCS
predicts them, that two runs give byte-identical files and that the raw files are not changed.

## Licenses and credit

- **Code** — MIT (`LICENSE`).
- **Our images** (`out/publish/`, `docs/gallery/`, acceptance sheets) — CC BY-SA 4.0, author Denis Krivosheev.
  Data credit: "NASA, ESA, CSA, STScI" for JWST, "NASA, ESA, STScI" for Hubble; processing — Denis Krivosheev.
- **Data** — the MAST archive (STScI); the terms of NASA, ESA, CSA and STScI apply to it.
- **Third-party images for comparison** in `data/reference/`: the 1998 Hubble Heritage image (STScI-1998-39,
  ESA/Hubble opo9839a, CC BY 4.0, Hubble Heritage Team (STScI/AURA/NASA/ESA)); processings of Sh2-305 by Melina Thévenot
  and Mark McCaughrean (Wikimedia Commons, CC BY-SA 4.0) — authors and links in `data/reference/sh2305/sources.md`.

---

Документация по-русски: [README.ru.md](README.ru.md)
