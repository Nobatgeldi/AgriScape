# AgriScape Pipeline — How the Data Flows, Step by Step

This document explains **where the satellite data comes from**, **how each band
is stored on disk**, and **how every stage reads, processes, and converts it**
into the final Unreal Engine weightmaps. It is meant to be read top-to-bottom by
someone who wants to understand the whole system, not just run it.

---

## Table of contents

1. [The big picture (data flow)](#1-the-big-picture-data-flow)
2. [Background: how geospatial rasters are stored](#2-background-how-geospatial-rasters-are-stored)
3. [Stage 0 — Data acquisition (`download.py`)](#3-stage-0--data-acquisition-downloadpy)
4. [Stage 1–2 — NDVI stack (`ndvi_stack.py`)](#4-stage-12--ndvi-stack-ndvi_stackpy)
5. [Stage 2 (enriched) — Feature stack (`build_features.py`)](#5-stage-2-enriched--feature-stack-build_featurespy)
6. [Stage 2 (optional) — Deep features (`deep_features.py`)](#6-stage-2-optional--deep-features-deep_featurespy)
7. [Stage 3 — Ground truth & classification (`classify.py`)](#7-stage-3--ground-truth--classification-classifypy)
8. [Stage 4 — Weightmap export (`export_weightmaps.py`)](#8-stage-4--weightmap-export-export_weightmapspy)
9. [File & data-type inventory](#9-file--data-type-inventory)
10. [Glossary](#10-glossary)

---

## 1. The big picture (data flow)

```
                 Copernicus / Microsoft Planetary Computer (STAC API)
                                     │  Sentinel-2 L2A scenes
                                     ▼
   download.py  ──►  scenes/<date>/B02..B12,SCL .tif        (raw reflectance per band, per date)
                                     │
   ndvi_stack.py / build_features.py ──►  feature_stack.tif (one multi-band raster: features per pixel)
                                     │                        + feature_stack.json (band names)
        deep_features.py (optional) ─┤  appends deep_pca_* bands
                                     ▼
   classify.py  ──►  classified.tif (1 band, class id per pixel)  + classified.json (id→name)
        ▲                            │
        │ ground_truth.gpkg          ▼
        │ (labeled polygons)  export_weightmaps.py ──► weightmaps/weightmap_<class>.png  (Unreal)
```

Every arrow is a file on disk. Nothing is kept only in memory between stages, so
you can inspect, resume, or swap out any stage independently.

---

## 2. Background: how geospatial rasters are stored

Before the stages, three concepts explain *how bands are stored*:

### A raster band = a 2D grid of numbers
A single Sentinel-2 band (say Red) is just a 2D array — e.g. `2364 × 2306`
values, one per pixel. Each value is a **Digital Number (DN)**: for L2A surface
reflectance the DN is `reflectance × 10000` (so a DN of `2500` means 0.25
reflectance). The array on its own has no idea *where on Earth* it sits — that
comes from the two pieces of georeferencing below.

### CRS — Coordinate Reference System
The **CRS** says what coordinate system the pixels live in. AgriScape works in
**UTM** (e.g. `EPSG:32636` = UTM zone 36N), where coordinates are in **metres**.
Metres matter because a 10 m Sentinel-2 pixel is literally 10 units wide, which
makes distances, windows, and buffers simple. `download.py` asks the server for
`crs="utm"`, so it auto-picks the correct UTM zone for your area.

### Affine transform — pixel ↔ ground mapping
The **affine transform** is 6 numbers that convert a pixel `(row, col)` into a
ground coordinate `(x, y)` in the CRS. For a north-up 10 m image it encodes:
"pixel (0,0)'s top-left corner is at (x0, y0); each column steps +10 m east;
each row steps −10 m north (`predictor` note: y decreases downward)". CRS +
transform together fully georeference the 2D array.

### The container: GeoTIFF
All rasters here are **GeoTIFF** (`.tif`) files. A GeoTIFF stores one or more
band arrays **plus** a header (the *profile*) holding the georeferencing and
storage options. In `rasterio`, the profile is a dict with keys like:

| profile key  | meaning                                    | typical value here            |
|--------------|--------------------------------------------|-------------------------------|
| `driver`     | file format                                | `GTiff`                       |
| `width`/`height` | columns / rows                         | e.g. 2364 / 2306              |
| `count`      | number of bands packed in the file         | 1 (SCL) … 28 (feature stack)  |
| `dtype`      | number type of each pixel                   | `float32`, `uint8`, `uint16`  |
| `crs`        | coordinate reference system                 | `EPSG:32636`                  |
| `transform`  | affine pixel→ground mapping                 | 6-number Affine               |
| `nodata`     | value meaning "no data"                     | `NaN` for floats, unset else  |
| `compress`   | lossless compression                        | `deflate`                     |
| `predictor`  | helps compression on float/int data         | `3` (float), `1` (byte)       |

**Key idea:** a "stack" in AgriScape is one GeoTIFF where **each band is a
feature**. Band 1 might be `ndvi_2024-04-15`, band 2 `red_2024-04-15`, etc. The
classifier reads the pixel *down through all bands* to get that pixel's feature
vector. A sidecar `*.json` file lists the band names in order, because GeoTIFF
band descriptions alone are easy to lose.

---

## 3. Stage 0 — Data acquisition (`download.py`)

**Goal:** turn "an area + some date ranges" into folders of raw band GeoTIFFs.

### Where the data comes from
Sentinel-2 is an ESA mission; its imagery is free and open. We pull the
**L2A** product (Level-2A = *bottom-of-atmosphere surface reflectance*, already
atmospherically corrected and ready to analyse) from the **Microsoft Planetary
Computer** via its **STAC API** (`https://planetarycomputer.microsoft.com/api/stac/v1`,
collection `sentinel-2-l2a`). STAC = *SpatioTemporal Asset Catalog*, a standard
JSON API for searching Earth-observation scenes by area, time, and metadata.
(The manual alternative is the Copernicus Data Space Ecosystem browser, which
gives the same `.jp2` bands — the pipeline accepts those too.)

### The area (AOI) and the search
- The AOI is a bounding box in **lon/lat (WGS84, `EPSG:4326`)**. It can be given
  directly (`--bbox W S E N`) or read from a vector file (`--aoi`, default
  `ExampleArea.gpkg`). `bbox_from_vector()` reads the polygon and reprojects its
  extent to lon/lat if needed.
- For each requested **window** (e.g. `2024-04-01/2024-04-30`) `find_best_item()`
  runs a STAC search filtered by bbox, date range, and `eo:cloud_cover < max`,
  then keeps the **single least-cloudy scene**.

### The bands we request
`BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL"]`:

| Asset | Meaning              | Native resolution | Used for                         |
|-------|----------------------|-------------------|----------------------------------|
| B02   | Blue (~490 nm)       | 10 m              | spectral feature                 |
| B03   | Green (~560 nm)      | 10 m              | spectral feature, RGB for ResNet |
| B04   | Red (~665 nm)        | 10 m              | NDVI + spectral + RGB            |
| B08   | NIR (~842 nm)        | 10 m              | NDVI + spectral                  |
| B11   | SWIR-1 (~1610 nm)    | 20 m              | spectral feature                 |
| B12   | SWIR-2 (~2190 nm)    | 20 m              | spectral feature                 |
| SCL   | Scene Classification | 20 m              | cloud/shadow masking             |

Only B04/B08/SCL are strictly required (they make NDVI); the rest enrich the
features later.

### How the bands are loaded and stored
`download_scene()` calls **`odc.stac.load`** with `resolution=10` and
`crs="utm"`. This does the heavy lifting:
- **Reprojects/resamples every asset onto one common 10 m UTM grid.** The 20 m
  bands (B11, B12, SCL) are resampled up to 10 m so all bands share the exact
  same grid — no misalignment later.
- Applies the STAC nodata metadata and returns each band as a **`float32`**
  array (float so nodata can be `NaN`). The *values are still raw DN* (0–10000
  for reflectance, 0–11 for SCL) — no reflectance scaling is applied yet.
- `chunks={"x": 2048, "y": 2048}` makes the load **Dask-backed**, so a large AOI
  streams in tiles instead of exploding memory.

Each band is then written with `rioxarray`'s `.rio.to_raster(..., compress="deflate")`
into this layout:

```
scenes/
    2024-04-15/  B02.tif B03.tif B04.tif B08.tif B11.tif B12.tif SCL.tif
    2024-07-29/  ...
    2024-09-27/  ...
```

One folder per acquisition date (named from the scene's timestamp), one GeoTIFF
per band. Every file is a single-band `float32` GeoTIFF on the shared 10 m UTM
grid. **This is the on-disk "raw band store" the rest of the pipeline reads.**

> Manual downloads differ only in dtype: Copernicus `.jp2` bands are native
> `uint16`. The code finds bands by name regardless of extension
> (`find_band_file` tries `.jp2/.tif/.tiff`), and the reflectance scaling later
> (`/10000`) is correct for both.

---

## 4. Stage 1–2 — NDVI stack (`ndvi_stack.py`)

**Goal:** collapse each date's Red+NIR into one **NDVI** band and stack the dates
into a single multi-band raster (band *i* = NDVI for date *i*).

### Why NDVI
NDVI (Normalized Difference Vegetation Index) measures greenness:

```
NDVI = (NIR − Red) / (NIR + Red)
```

Healthy vegetation reflects lots of NIR and absorbs Red → NDVI near 1; bare
soil/water → near 0 or negative. Crucially, dividing (a ratio) makes NDVI
**scale-invariant**, so it does not matter that the input is raw DN.

### Processing, per date (`load_masked_ndvi`)
1. **Read** `B04` (Red) and `B08` (NIR) arrays.
2. **Compute NDVI** in `float32` with a divide-by-zero guard (`compute_ndvi`):
   pixels where `NIR+Red == 0` become `NaN` instead of crashing.
3. **Cloud-mask with SCL** (`SCL_BAD_VALUES = {0,1,3,8,9,10,11}`): SCL labels
   every pixel (0 no-data, 3 cloud shadow, 8/9 cloud, 10 cirrus, 11 snow, …).
   Any pixel whose SCL is in the bad set is set to `NaN` — we refuse to trust
   cloudy pixels. SCL is resampled to the Red grid with **nearest-neighbour**
   (never interpolate class labels).

### Aligning dates onto one grid (`align_to_reference`)
The **first date's grid becomes the reference**. If a later date has a different
CRS, transform, or size (e.g. it came from a neighbouring UTM tile), it is
**reprojected** onto the reference grid with `rasterio.warp.reproject`
(bilinear, `NaN` nodata). This guarantees every band in the final stack is
pixel-for-pixel aligned.

### Filling cloud gaps (`fill_gaps_median`)
After stacking all dates into a `(n_dates, H, W)` array, some pixels are still
`NaN` (clouded on that date). Each such pixel is filled with the **median of its
own values across the other dates** — a simple, robust temporal gap-fill. Pixels
that are `NaN` on *every* date (outside the scene footprint) become `0`.
Optional `--smooth savgol` additionally runs a Savitzky-Golay filter along time
for cleaner phenology (needs ≥4 dates).

### How it is stored
Written as one **`float32` GeoTIFF**, `count = n_dates`, `compress="deflate"`,
`predictor=3` (predictor 3 = floating-point differencing, which makes smooth
float rasters compress much better). Each band gets a description (the date), and
a sidecar `ndvi_stack.json` records `{"band_dates": [...], "n_bands": n}`.

---

## 5. Stage 2 (enriched) — Feature stack (`build_features.py`)

**Goal:** instead of NDVI alone, give the classifier real context. Produces one
`float32` multi-band GeoTIFF combining **three feature groups**. This is the
default stack for classification.

### Loading & converting bands (`load_band_aligned`)
For each date and each present band:
- Read the band, cast to `float32`, and **convert DN → reflectance by dividing
  by 10000** (so values sit in ~0–1, a tidy physical range). NDVI is still
  computed as a ratio, so scaling does not change it.
- **Align** to the reference grid (same reprojection logic as Stage 1).
- **Mask** clouds via SCL (bad pixels → `NaN`).

Missing bands are skipped gracefully (an old B04/B08-only download still works —
you just get fewer spectral features).

### Group 1 — Spectral
The raw reflectance of every band present (`blue, green, red, nir, swir1, swir2`)
**for each date**, plus that date's NDVI. This is the "raw material" — the model
sees the actual multi-spectral signature, not just a derived index.

### Group 2 — Spatial (texture) — `local_std`
For each date's NDVI, compute the **local standard deviation in a window**
(default 5×5). Implemented fast with `scipy.ndimage.uniform_filter`:
`std = sqrt(mean(x²) − mean(x)²)` over the window. Homogeneous fields → low
texture; field edges, roads, mixed pixels → high texture. This is a scalable
stand-in for GLCM texture. **It is computed on the gap-filled (NaN-free) NDVI**,
and on the *whole image* (not per tile) so there are no tile-seam artifacts.

### Group 3 — Temporal — `temporal_features`
Per-pixel summary of the NDVI **time series**:
`ndvi_mean, ndvi_min, ndvi_max, ndvi_amplitude (max−min)`, and `ndvi_slope`
(the ordinary-least-squares trend of NDVI vs. date index, computed per pixel in
closed form). These capture *phenology*: e.g. wheat peaks early then drops,
maize peaks late — the slope/amplitude encode that shape in a few numbers.
`ndvi_slope` needs ≥2 dates.

### How it is stored
All feature arrays are concatenated in a **stable order** into a single
`float32` GeoTIFF (`compress="deflate"`, `predictor=3`). A sidecar
`feature_stack.json` lists every band name (`features`, `n_features`,
`band_dates`) so you always know which band is which. Example for 3 dates,
6 bands present:

```
blue_2024-04-15, green_…, red_…, nir_…, swir1_…, swir2_…, ndvi_…, ndvi_std5_…,   (×3 dates)
ndvi_mean, ndvi_min, ndvi_max, ndvi_amplitude, ndvi_slope
```

---

## 6. Stage 2 (optional) — Deep features (`deep_features.py`)

**Goal:** append *learned* context bands from a frozen, Sentinel-2-pretrained
**ResNet-18** (via TorchGeo), so the Random Forest can split on land-cover /
texture patterns it could never hand-engineer. GPU optional.

> This does **not** make the model a crop classifier — the backbone was trained
> for coarse EuroSAT-style land cover, which cannot tell wheat from barley. It
> adds *context* features that the RF may find useful.

### Input preparation (`_load_rgb_images`)
- Reads **R, G, B = B04, B03, B02** reflectance per date (reusing
  `load_band_aligned`), builds a **temporal-median RGB composite** (one robust
  cloud-free image; `--per-date` keeps each date instead).
- **Normalizes** each band to zero-mean/unit-std (per-image z-score). This is a
  robust, deterministic input normalization for a frozen feature extractor — the
  exact pretraining statistics are not required because the features are
  PCA-reduced and fed to a tree model, not a softmax head.

### Dense feature extraction (`_build_extractor`, `_tile_features`)
- Loads a frozen `ResNet18_Weights.SENTINEL2_RGB_MOCO` backbone (3-band RGB —
  matches our downloaded bands; no extra download). Set to `.eval()` and run
  under `torch.inference_mode()` on CUDA.
- Runs the conv stem + `layer1..layer2` **fully-convolutionally** (cut at
  `layer2`: 128 channels, stride 8 — a balance of spatial detail vs. semantics),
  then **bilinearly upsamples** the feature map back to pixel resolution.
- Runs in **overlapping tiles** (`--deep-tile-size`, small halo) so GPU memory
  stays bounded and there are no seams — the halo is computed then cropped away.

### Dimensionality reduction (PCA)
128 correlated channels would swamp the RF, so we **PCA-reduce to K components**
(`--deep-components`, default 16). Two passes over the tiles: pass 1 fits PCA on
a random pixel subsample; pass 2 transforms each tile. PCA is fit **once for the
whole scene**, so the transform is identical for training and full-scene
prediction — no train/inference skew.

### How it is stored
The K components are **appended as new bands** (`deep_pca_00..K-1`) to the
existing feature stack: the file is rewritten with `original_count + K` bands
(originals copied through, deep bands written per tile), then the sidecar
`features`/`n_features` are updated. `classify.py` and `export_weightmaps.py`
need no changes — they just see more bands.

---

## 7. Stage 3 — Ground truth & classification (`classify.py`)

**Goal:** learn crop classes from labeled example polygons, then classify every
pixel of the stack.

### Ground truth — where labels are stored
Ground truth is a **vector** file (GeoPackage `.gpkg` or Shapefile) with a
`class_name` attribute (`wheat`, `barley`, `corn`, `vineyard`, `non_agri`, …).
Each feature is a polygon drawn over a field you are sure about. Vector = shapes
+ attributes (not a pixel grid), stored in its own CRS. (`fetch_osm_labels.py`
can pre-seed vineyard/orchard/non_agri from OpenStreetMap; the cereals are drawn
by hand — crop-type labels are not open data.)

### From polygons to training pixels (`extract_training_samples`)
1. **Reproject** the polygons to the stack's CRS if they differ (QGIS often
   digitizes in WGS84).
2. **Rasterize** the polygons onto the stack grid (`rasterio.features.rasterize`):
   each class name → an integer id (1..N; 0 = background), producing a label
   raster the same size as the stack.
3. Where the label raster is non-zero, **pull the full feature vector** (all
   bands, i.e. that pixel's time-series/features) → `X`; the label id → `y`.
4. **Drop any samples with `NaN`** features (e.g. pixels outside a date's
   footprint) — Random Forest cannot handle `NaN`.

### The model (`train_model`)
A **`RandomForestClassifier`** (300 trees, `class_weight="balanced"` to offset
uneven class sizes, `random_state=42`). Data is split 75/25; the held-out 25%
produces a **validation report** (precision/recall/F1 per class) and a confusion
matrix so you can judge label quality before trusting the map. The trained model
is saved with `joblib` (`rf_model.joblib`).

### Classifying the whole scene (`classify_stack_tiled`)
The full raster can be 10000×10000+ px, so prediction runs **tile by tile**
(`--tile-size`, default 1024) to bound memory:
- Read a window of the stack `(n_bands, h, w)`, reshape to `(h*w, n_bands)`.
- Pixels with any `NaN` band → assigned class `0` (background); the rest are
  predicted by the forest.
- Reshape predictions back to `(h, w)` and write into the output.

### How it is stored
Output `classified.tif` is a **single-band `uint8` GeoTIFF** (`predictor=1` —
byte data, no float differencing), one class id per pixel, same grid/CRS as the
stack. A sidecar `classified.json` records the **id → class-name** mapping so the
next stage knows that (say) `3` means `corn`.

---

## 8. Stage 4 — Weightmap export (`export_weightmaps.py`)

**Goal:** turn the class-id raster into one grayscale PNG per class, sized for
Unreal Engine's Landscape system.

### Why resample to a special size
Unreal landscapes must be a specific size: `quads_per_side + 1`, where valid
quad counts are `[126, 252, 504, 1008, 2016, 4032, 8128]` → valid sizes
`127, 253, 505, 1009, 2017, 4033, 8129`. `nearest_valid_unreal_size()` picks the
smallest valid size ≥ your raster's longest side. The class raster is resampled
to a square of that size with **nearest-neighbour** (`resample_to_size`) so no
fractional/invented class ids appear.

### Building each mask
For every class id, `mask = (resized == id)` → a binary image, scaled to full
range: `255` for 8-bit (`uint8`) or `65535` for 16-bit (`uint16`). `255` = "this
pixel is fully this class", `0` = "not this class". One PNG per class:
`weightmap_wheat.png`, `weightmap_corn.png`, … Each is a grayscale image the same
square size, ready to import into an Unreal **Landscape Layer Blend** material.
The script prints the coverage % per class and the exact import size to use.

### How it is stored
Plain **grayscale PNG** files (no georeferencing — Unreal places them by import
size, not CRS). 8-bit by default (`--bit-depth 16` for smoother blends).

---

## 9. File & data-type inventory

| File                          | Produced by            | Bands / content                     | dtype     | Georef? |
|-------------------------------|------------------------|-------------------------------------|-----------|---------|
| `scenes/<date>/B0x.tif`       | `download.py`          | one raw reflectance band            | `float32` | yes     |
| `scenes/<date>/SCL.tif`       | `download.py`          | scene-classification labels         | `float32` | yes     |
| `ndvi_stack.tif`              | `ndvi_stack.py`        | NDVI per date                       | `float32` | yes     |
| `feature_stack.tif`           | `build_features.py`    | spectral+spatial+temporal features  | `float32` | yes     |
| `feature_stack.tif` (+deep)   | `deep_features.py`     | above + `deep_pca_*` bands          | `float32` | yes     |
| `*_stack.json`                | stack builders         | ordered band-name list              | JSON      | —       |
| `ground_truth.gpkg`           | you / `fetch_osm_labels`| labeled polygons (`class_name`)    | vector    | yes     |
| `rf_model.joblib`             | `classify.py`          | trained Random Forest               | binary    | —       |
| `classified.tif`             | `classify.py`          | class id per pixel                  | `uint8`   | yes     |
| `classified.json`            | `classify.py`          | id → class-name map                 | JSON      | —       |
| `weightmaps/weightmap_*.png` | `export_weightmaps.py` | per-class binary mask               | `uint8/16`| no      |

**Values at a glance:** reflectance DN `0–10000` (raw) → `/10000` = `0–1`
reflectance; NDVI `−1…1`; texture/temporal `float32`; class ids small integers;
weightmaps `0` or `255/65535`.

---

## 10. Glossary

- **DN (Digital Number):** the integer stored per pixel; for L2A reflectance,
  `DN = reflectance × 10000`.
- **L2A:** Sentinel-2 Level-2A — atmospherically-corrected *surface* reflectance.
- **STAC:** SpatioTemporal Asset Catalog — the JSON API used to search scenes.
- **CRS:** Coordinate Reference System (we use UTM, metres).
- **Affine transform:** 6 numbers mapping pixel (row,col) ↔ ground (x,y).
- **GeoTIFF:** the `.tif` raster container (band arrays + georeferencing header).
- **Band / stack:** one 2D feature grid / a multi-band GeoTIFF where each band is
  a feature.
- **NDVI:** `(NIR − Red)/(NIR + Red)` greenness index.
- **SCL:** Sentinel-2 Scene Classification Layer (per-pixel cloud/shadow/veg/…).
- **Rasterize:** convert vector polygons into a pixel grid of class ids.
- **Weightmap:** grayscale mask telling Unreal how much of a layer paints each
  pixel.
