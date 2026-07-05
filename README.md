# AgriScape AI — Starter Pipeline

Minimal working implementation of the 4-step workflow: NDVI stack → Random
Forest classification → Unreal-ready weightmaps.

## Setup

```bash
pip install -r requirements.txt
```

## Quick start: run the whole pipeline

`run_pipeline.py` chains every stage in order
(download → NDVI → classify → weightmaps). Because the classifier needs
hand-labeled crop polygons, it runs in two passes around a manual labeling
step:

```bash
# Pass 1 — download, build the NDVI stack, and seed non-crop classes from OSM,
# then stop so you can label wheat/barley/corn in QGIS.
# (AOI defaults to ExampleArea.gpkg; override with --aoi <file> or --bbox W S E N)
python agriscape/run_pipeline.py \
    --window 2024-04-01/2024-04-30 \
    --window 2024-07-01/2024-07-31 \
    --window 2024-09-01/2024-09-30 \
    --bootstrap-osm --max-per-class 500

# ...open ground_truth.gpkg in QGIS, draw wheat/barley/corn fields...

# Pass 2 — classify and export weightmaps.
python agriscape/run_pipeline.py --start-from classify
```

`--start-from {download,ndvi,classify,weightmaps}` and `--stop-after` let you
resume or run any single stage. If you already have a labeled
`ground_truth.gpkg`, drop `--bootstrap-osm` and it runs straight through.

The individual stages can also be run one at a time, as described below.

## 1. Get Sentinel-2 L2A data

**Option A — scripted (recommended):** pull the least-cloudy scene per
phenological window straight from Microsoft Planetary Computer (no account
needed):

```bash
python agriscape/download.py scenes/ \
    --bbox 32.5 39.5 33.5 40.4 \
    --window 2024-04-01/2024-04-30 \
    --window 2024-07-01/2024-07-31 \
    --window 2024-09-01/2024-09-30
```

**Option B — manual:** download scenes for 3-8 dates spanning your crop's
growth cycle from the Copernicus Data Space Ecosystem
(https://dataspace.copernicus.eu) and arrange them like:

```
scenes/
    2024-04-15/B02.jp2 B03.jp2 B04.jp2 B08.jp2 B11.jp2 B12.jp2 SCL.jp2
    2024-06-20/...
    2024-09-10/...
```

`download.py` fetches blue/green/red/NIR/SWIR + SCL. Only B04/B08/SCL are
strictly required (NDVI); the extra bands just enable richer features in
step 2. Both `.jp2` and `.tif` band files are accepted.

## 2. Build the feature stack

The Random Forest classifies each pixel from the bands in this stack, so
richer features mean a smarter model than a bare NDVI calculator. Build the
enriched stack:

```bash
python agriscape/build_features.py scenes/ output/feature_stack.tif
```

It produces one multi-band GeoTIFF combining three feature groups:

- **Spectral** — raw reflectance per date for every band present (blue, green,
  red, NIR, SWIR1, SWIR2) plus the NDVI.
- **Spatial (texture)** — local NDVI standard deviation in a window
  (`--window-size`, default 5×5): homogeneous fields read low, edges/mixed
  pixels high.
- **Temporal** — per-pixel NDVI summary over the season: mean, min, max,
  amplitude, and linear trend slope.

Missing bands are skipped gracefully, and each group can be turned off
(`--no-spectral`, `--no-spatial`, `--no-temporal`). Then point step 3 at
`output/feature_stack.tif`.

For a bare NDVI-only stack instead (original behaviour), use
`python agriscape/ndvi_stack.py scenes/ output/ndvi_stack.tif` — scenes on a
different CRS/grid are reprojected automatically, cloud gaps filled with the
per-pixel temporal median, and `--smooth savgol` (4+ dates) smooths the curves.

## 3. Collect ground truth & classify

The classifier learns from labeled example polygons in a GeoPackage with a
`class_name` field (values like `wheat`, `barley`, `corn`, `vineyard`,
`non_agri`). Crop-type labels are not open data for most regions, so wheat/
barley/corn polygons must be digitized by hand in QGIS. Draw a polygon inside
fields you're sure about (use a satellite basemap or local knowledge), aiming
for ~10-15 per class spread across the area.

### Optional: bootstrap from OpenStreetMap

OSM does reliably tag vineyards, orchards, and non-agricultural land, so you
can auto-generate those classes and only hand-draw the cereals:

```bash
python agriscape/fetch_osm_labels.py output/ndvi_stack.tif ground_truth.gpkg \
    --max-per-class 500
```

This reads the AOI and CRS from the NDVI stack, queries the Overpass API, and
writes `vineyard`, `orchard`, and `non_agri` (built-up/water/roads) polygons.
`--max-per-class` randomly caps each class (OSM returns far more roads than
vineyards; capping keeps training balanced and memory bounded). Open the
result in QGIS, sanity-check the auto polygons against a satellite basemap
(OSM tags can be stale), then add your `wheat`/`barley`/`corn` fields.

### Classify

```bash
python agriscape/classify.py output/feature_stack.tif ground_truth.gpkg \
    output/classified.tif --model-out output/rf_model.joblib
```

(Use `output/ndvi_stack.tif` here instead if you built the NDVI-only stack.)
Ground truth in a different CRS than the stack is reprojected automatically.
This prints a validation report (precision/recall per class) so you can
judge whether you need more/better training polygons before trusting the
output at scale.

## 4. Export Unreal weightmaps

```bash
python agriscape/export_weightmaps.py output/classified.tif \
    output/classified.json output/weightmaps/ --bit-depth 8
```

Produces one PNG per class, resampled to a valid Unreal Landscape size
(e.g. 2017x2017), ready to import into the Landscape Layer Blend material.

## Notes on scaling to 100km x 100km

- At 10m resolution, 100km side = 10,000 pixels — the `classify.py` script
  processes in tiles (default 1024x1024) so memory stays bounded regardless
  of total extent.
- For very large areas, split the AOI into multiple NDVI stacks (e.g. per
  10km x 10km tile) and run the pipeline per-tile, then mosaic weightmaps
  before import, or import as separate Landscape Streaming Proxies in Unreal.
- Consider Dask or a job queue (Prefect/Airflow) once you're running this
  as a repeatable service rather than one-off local jobs.

## Known simplifications to build on

- The Savitzky-Golay smoother (`--smooth savgol`) needs 4+ dates; with the
  minimum 3 dates only the temporal-median gap fill applies. A harmonic
  (Fourier) fit would work at 3 dates if needed.
- `download.py` picks one least-cloudy scene per window; compositing several
  scenes per window (e.g. median composite) would be more robust in
  persistently cloudy regions.
- Reprojection aligns everything to the first date's grid. When mosaicking
  multiple Sentinel-2 tile IDs into one stack, prefer building one stack per
  tile and mosaicking the classified outputs instead.
