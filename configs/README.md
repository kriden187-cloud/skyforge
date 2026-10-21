# Config keys

Each object has a folder `configs/<object>/` with `base.yaml` (object, filters, alignment, cleaning, masks) and style
files that start with `extends: base` (or another style) and override only what they change; dictionaries are merged
deeply. The full list of keys with defaults and limits is in `src/skyforge/config.py`; unknown keys are an error. The
comments inside the object configs are the processing notes and stay in Russian. Below are the four sections that
shape the look of a picture, in pipeline order.

## `background` — background model (stage `clean`)

| Key | Meaning |
|---|---|
| `subtract` | Subtract the background model at all (`false` keeps the frame as delivered). |
| `model` | `photutils` — smooth Background2D grid; `blocks` — own blocks over free sky, continued under the nebula from neighbours (`blocks_px`, `blocks_mask_grow_px`, `blocks_min_free`). |
| `tiles` | A constant correction per mosaic region with its own exposure set; removes steps between tiles. |
| `nexp_levels` | A constant per region of equal exposure count (own level-2 mosaic). |
| `smooth` | A smooth model of residual gradients (`box_arcsec`, `filter_size`, `mask_sigma`, `mask_dilate_arcsec`). |
| `nebula_dilate_arcsec` | Radius by which the nebula mask is grown. The nebula, star and empty-sky masks built here are used by every later stage. |
| `nebula_mask_exclude` | Filters that do not take part in building the nebula mask (e.g. a striped filter). |

## `stretch` — from data values to 0–1 (stage `stretch`)

| Key | Meaning |
|---|---|
| `method` | `asinh` (default), `log`, `sqrt` or `linear`. |
| `anchor` | `masks` — black point from the empty-sky mask, white point from the nebula mask; `frame` — whole frame. |
| `black`, `white` | Percentiles mapped to 0 and 1. |
| `a` | Strength of the asinh curve: smaller is stronger. |
| `black_mode`, `black_k` | `percentile` — the `black` percentile; `pedestal` — sky median minus `black_k`·σ, so sky noise is not clipped to 0. |
| `sky_level` | Puts the sky median at this level (out of 255) with a smooth toe below it; `sky_toe_max_p` limits the toe. |
| `sky_reference` | Where the sky is measured: `mask` (empty-sky mask) or `darkest_blocks` (the darkest blocks of the reference filter, for nebulae that fill the frame). |
| `sky_poly_order`, `sky_poly_exclude`, `sky_poly_taper_px` | A smooth sky polynomial fitted outside an exclusion file, faded out towards the nebula. |
| `per_filter` | Per-filter overrides of `black`, `white`, `a`, `black_k`: `{F470N: {white: 99.9, a: 0.05}}`. |
| `saturated_stars` | Stars with saturated cores as a separate layer before the stretch (`enabled`, radii, own `a`). |

## `color` — filters into RGB (stage `compose`)

| Key | Meaning |
|---|---|
| `mode` | `wavelength` — hues spread over `hue_range` from short to long wavelength; `manual` — colours from `overrides`. |
| `overrides` | Colour per filter, `"#rrggbb"`. |
| `weights` | Brightness weight per filter, e.g. `{F212N: 0.25}`. |
| `blend` | `average` (all layers at 1 give white) or `screen` (brighter). |
| `saturation`, `saturation_gamut` | Saturation gain; `limit` keeps every channel inside 0–1 instead of clipping. |
| `saturation_above_sky` | Saturation grows only above the sky level, over this many levels. |
| `shadows` | Shadow lift: a number or `{value, scope: nebula, by: max, above_sky: true}`; larger is stronger, 0 is off. |
| `star_saturation`, `star_saturation_scope` | Saturation of field stars (1 as is, 0 grey); `sky` touches only stars outside the nebula. |
| `star_chroma` | Smooths the colour of saturated-star spikes, brightness unchanged. |
| `neutral_background` | Sky colour to grey: `map` (by blocks), `constant` (one correction) or `off`. |
| `background_keep_level` | Remove only the colour shift of the sky, keep its level from the stretch. |

## `finish` — final touches and export (stage `finish`)

| Key | Meaning |
|---|---|
| `denoise` | 0 off … 1 maximum. |
| `sharpen`, `sharpen_radius` | Unsharp mask strength and radius in pixels. |
| `stars` | Star layer: `scale` (brightness), `shrink` (core reduction radius, px). |
| `local_contrast` | Local contrast inside the nebula mask: `radius` (px), `amount` (0 off). |
| `north_up` | Rotate north up; done in `align` within the same interpolation. |
| `nodata_fill` | Pixels outside every filter: `black` or `sky` (flat sky level). |
