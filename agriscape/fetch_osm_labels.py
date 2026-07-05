"""
Helper: bootstrap a ground-truth GeoPackage from OpenStreetMap.

Crop-type reference polygons (wheat/barley/corn) do not exist as open data for
most of the world, so those must be digitized by hand. But OSM *does* reliably
tag vineyards, orchards, and non-agricultural land (built-up areas, water,
major roads) almost everywhere. This script pulls those via the Overpass API
over your area of interest and writes them as labeled polygons in your NDVI
stack's CRS, so in QGIS you only have to add the wheat/barley/corn samples.

The AOI and CRS are taken from the NDVI stack GeoTIFF, so the output lines up
with everything else in the pipeline.

Example:
    python agriscape/fetch_osm_labels.py output/ndvi_stack.tif ground_truth.gpkg

Then open ground_truth.gpkg in QGIS, verify/trim the auto polygons, and draw
your own wheat/barley/corn fields before running classify.py.
"""

import time
from pathlib import Path

import geopandas as gpd
import osm2geojson
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Overpass rejects the default python-requests User-Agent (HTTP 406), and
# asks callers to identify themselves. See the Overpass API usage policy.
HTTP_HEADERS = {
    "User-Agent": "AgriScape-AI/0.2 (ground-truth bootstrap; +https://github.com/)",
    "Accept": "application/json",
}
MAX_RETRIES = 3

# OSM tag selectors -> ground-truth class_name. Roads are lines and get
# buffered into thin polygons; everything else is already an area.
FEATURE_QUERIES = [
    ('way["landuse"="vineyard"]',   "vineyard", False),
    ('relation["landuse"="vineyard"]', "vineyard", False),
    ('way["landuse"="orchard"]',    "orchard",  False),
    ('relation["landuse"="orchard"]', "orchard", False),
    ('way["landuse"="residential"]', "non_agri", False),
    ('relation["landuse"="residential"]', "non_agri", False),
    ('way["natural"="water"]',      "non_agri", False),
    ('relation["natural"="water"]', "non_agri", False),
    ('way["landuse"="reservoir"]',  "non_agri", False),
    ('way["highway"~"motorway|trunk|primary|secondary"]', "non_agri", True),
]


def build_overpass_query(bbox_wgs84: tuple[float, float, float, float]) -> str:
    """bbox is (west, south, east, north) in lon/lat; Overpass wants
    (south, west, north, east)."""
    w, s, e, n = bbox_wgs84
    bbox = f"{s},{w},{n},{e}"
    parts = "\n".join(f"  {sel}({bbox});" for sel, _cls, _road in FEATURE_QUERIES)
    return f"[out:json][timeout:180];\n(\n{parts}\n);\nout geom;"


def run_overpass(query: str) -> dict:
    """POST the query to each Overpass mirror, retrying on rate-limit/server
    errors with exponential backoff before moving to the next mirror."""
    last_err = None
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                print(f"Querying Overpass: {endpoint} (attempt {attempt})")
                resp = requests.post(
                    endpoint, data={"data": query}, headers=HTTP_HEADERS, timeout=300
                )
                if resp.status_code in (429, 502, 503, 504):
                    raise requests.HTTPError(f"{resp.status_code} (server busy)")
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as err:
                last_err = err
                if attempt < MAX_RETRIES:
                    wait = 5 * 2 ** (attempt - 1)  # 5s, 10s
                    print(f"  {err}; retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(f"  {err}; giving up on this mirror.")
    raise RuntimeError(f"All Overpass endpoints failed: {last_err}")


def tags_to_class(tags: dict) -> str | None:
    """Map an OSM feature's tags to a ground-truth class_name, or None."""
    if tags.get("landuse") == "vineyard":
        return "vineyard"
    if tags.get("landuse") == "orchard":
        return "orchard"
    if tags.get("landuse") in {"residential", "reservoir"}:
        return "non_agri"
    if tags.get("natural") == "water":
        return "non_agri"
    if tags.get("highway"):
        return "non_agri"
    return None


def fetch_osm_labels(
    stack_path: str,
    output_path: str,
    bbox_override: tuple[float, float, float, float] | None = None,
    road_buffer_m: float = 8.0,
    max_per_class: int | None = None,
) -> None:
    with rasterio.open(stack_path) as src:
        stack_crs = src.crs
        stack_bounds = src.bounds  # in stack CRS (metric UTM)

    # AOI for Overpass must be in lon/lat.
    if bbox_override is not None:
        bbox_wgs84 = bbox_override
    else:
        bbox_wgs84 = transform_bounds(stack_crs, "EPSG:4326", *stack_bounds)

    print(f"AOI (lon/lat): {tuple(round(v, 4) for v in bbox_wgs84)}")

    data = run_overpass(build_overpass_query(bbox_wgs84))
    fc = osm2geojson.json2geojson(data)

    records = []
    for feat in fc["features"]:
        tags = feat.get("properties", {}).get("tags", {})
        cls = tags_to_class(tags)
        if cls is None:
            continue
        shp = shape(feat["geometry"])
        if shp.is_empty:
            continue
        records.append({"class_name": cls, "geometry": shp, "is_road": bool(tags.get("highway"))})

    if not records:
        raise RuntimeError(
            "No vineyard/orchard/non-agri features found in this AOI. "
            "You'll need to digitize all classes manually in QGIS."
        )

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326").to_crs(stack_crs)

    # Buffer road lines (and any stray line geometries) into thin polygons so
    # they rasterize to an area during training-sample extraction.
    line_types = {"LineString", "MultiLineString"}
    is_line = gdf.geometry.geom_type.isin(line_types)
    gdf.loc[is_line, "geometry"] = gdf.loc[is_line].geometry.buffer(road_buffer_m)

    # Keep only polygonal geometries and clip to the stack extent.
    gdf = gdf[gdf.geometry.geom_type.isin({"Polygon", "MultiPolygon"})]
    aoi = gpd.GeoDataFrame(geometry=[box(*stack_bounds)], crs=stack_crs)
    gdf = gpd.clip(gdf, aoi)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]

    gdf = gdf[["class_name", "geometry"]].reset_index(drop=True)

    # Optionally cap each class to keep training balanced and memory bounded:
    # OSM typically returns far more roads/built-up (non_agri) than vineyards,
    # which skews the training set and bloats extract_training_samples.
    if max_per_class is not None:
        parts = []
        for _cls, grp in gdf.groupby("class_name"):
            if len(grp) > max_per_class:
                grp = grp.sample(n=max_per_class, random_state=42)
            parts.append(grp)
        gdf = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True), crs=gdf.crs
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out, driver="GPKG")

    print(f"\nWrote {len(gdf)} polygons to {out}")
    print("Class counts:")
    for cls, n in gdf["class_name"].value_counts().items():
        print(f"  {cls}: {n}")
    print(
        "\nNext: open this in QGIS, sanity-check the auto polygons, then draw "
        "your wheat / barley / corn fields (class_name = 'wheat' etc.) before "
        "running classify.py."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Bootstrap ground_truth.gpkg from OpenStreetMap (Overpass)"
    )
    parser.add_argument("stack_path", help="NDVI stack GeoTIFF (sets AOI + CRS)")
    parser.add_argument("output", help="Output ground-truth GeoPackage path")
    parser.add_argument(
        "--bbox", nargs=4, type=float, default=None,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Override AOI bbox in lon/lat (default: derived from the stack)",
    )
    parser.add_argument(
        "--road-buffer", type=float, default=8.0,
        help="Half-width in meters to buffer road lines into polygons (default 8)",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Randomly cap each class to this many polygons (e.g. 500) to keep "
             "training balanced and memory bounded. Default: keep all.",
    )
    args = parser.parse_args()

    fetch_osm_labels(
        args.stack_path,
        args.output,
        bbox_override=tuple(args.bbox) if args.bbox else None,
        road_buffer_m=args.road_buffer,
        max_per_class=args.max_per_class,
    )
