# Skyforge overview

Skyforge turns public JWST and Hubble science frames from the MAST archive into colour images. Everything that
shapes a picture is written down in a YAML config (see [configs/README.md](../configs/README.md)), so a result can be
reproduced, reviewed and argued about. The full plan and the decision log are in Russian:
[PLAN.md](PLAN.md), [HANDOFF.md](HANDOFF.md).

## How the pipeline is built

`skyforge run <object> -s <style>` runs the stages below in order. Each stage can also be run alone; the heavy ones
(`inspect`, `align`, `clean`) are cached in `data/work/` and shared between styles of the same object.

| Stage | What it does | What it leaves behind |
|---|---|---|
| `fetch` | Finds the observations in MAST and downloads the filters listed in the config (level 3, or level-2 exposures with `source: level2`). Files already on disk are skipped. | FITS in `data/raw/`, a manifest in `data/raw/manifests/` |
| `build` | Only for `source: level2`: own mosaic per filter from the `_cal.fits` exposures — destripe in the detector frame, level matching, stacking on the i2d grid. | mosaics in `data/work/`, `l2_destripe.png`, `l2_overlaps.png`, `l2_stars.png` |
| `inspect` | Reads the headers, builds a filter table and a greyscale preview of every frame. | `inspect_sheet.png`, `inspect_<filter>.png` |
| `align` | Reprojects all filters onto the reference grid in one interpolation (north up if asked), crops to the common field. | `align_check.png`, `align_zoom.png`, `coverage.png` |
| `clean` | Fills saturated star cores, removes mosaic seams and cosmic-ray trails, subtracts the background model, builds the star, nebula and empty-sky masks. | `background_model.png`, `star_cores.png`, `star_mask.png`, `nebula_mask.png` |
| `stretch` | Maps each filter to 0–1: the black point from empty sky, the white point from the nebula, asinh (or log, sqrt, linear) in between. | `stretch_hist.png`, `stretch_<filter>.png` |
| `compose` | Gives each filter a colour (by wavelength or by hand), blends the layers into RGB, neutralises the sky colour. | `compose.png`, `compose_layers.png`, `compose_legend.png` |
| `finish` | Optional denoise, sharpening, star layer and local contrast; writes a 16-bit TIFF and a PNG with metadata. | `<object>_<style>.tif`, `<object>_<style>.png`, `finish_zoom.png` |
| report | Collects every sheet into one HTML page with 100% crops, sha256 of the final files and the difference from the accepted result. | `report.html`, `run.json`, `config.yaml` (the merged config of the run) |

After acceptance, `skyforge publish` makes the master PNG/TIFF, JPEG versions for platforms and RU/EN captions in
`out/publish/<object>/<date>/`, and `skyforge gallery` rebuilds [GALLERY.md](GALLERY.md).

## Principles

- **Reproducibility.** A result is defined by its config and a git tag. The same config on the same data gives
  byte-identical files; the processing hash goes into the image metadata, and the gallery lists the sha256 of the
  final pixels together with the command that recreates them.
- **Raw data is untouchable.** Downloaded FITS files are made read-only and are never written to; every
  intermediate product goes to `data/work/`, every result to `out/`. A test checks that a run leaves the raw files
  unchanged.
- **Acceptance by images.** No stage is judged by numbers alone. Each stage writes sheets and crops at 100%; an
  accepted result is saved as a baseline (`skyforge baseline`), and every later report shows the pixel difference
  from it, so an unintended change is visible at once.
- **The sky comes from a measured empty region.** The black point, the sky level and the sky colour are measured on
  pixels that are neither star nor nebula — the empty-sky mask, or, where the nebula fills the frame (Sh2-305), the
  darkest blocks of the reference filter. They are never set by eye.

## Known limitations

- **NGC 3132, JWST measured:** background steps along exposure-count boundaries up to ~0.6σ; cores of three saturated stars are white; bright-star spikes are outside the star mask.
- **NGC 3132, JWST showcase:** the same steps; three background details visible at ×2–×5; outer halo fainter than in the NASA release; one field star with a greenish core.
- **NGC 3132, Hubble WFPC2:** final at bin 2; 65 cosmic-ray traces on the sky; 0.25% zeros in the blue sky channel; seams ≤ 6 px filled linearly.
- **Sh2-305, JWST:** F182M 1/f stripes visible at ×5; the bright star's core is saturated; F212N not included; PAH filaments are nearly one hue.
- **W3, JWST (field W3-JWST-2):** no empty sky in the field (the sky reference is the faintest nebula); short-wave detector-gap strips filled from a neighbouring filter are noisier at 3×, and about a dozen 4″ squares where the gaps cross differ in tone; short-wave 1/f stripes on the sky at 5×.
- **Object search:** `skyforge discover` checks press releases; `skyforge untouched` checks web search, Wikimedia Commons and Flickr (Reddit, AstroBin and X are not reachable from the pipeline; on a known object it found 6 of 7 public processings).
