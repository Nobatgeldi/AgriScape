"""
Stage 0: Download Sentinel-2 L2A bands (B04, B08, SCL) from the Microsoft
Planetary Computer STAC API, so no manual browsing/downloading is needed.

For each requested acquisition window (e.g. "2024-04-01/2024-04-30") the
least-cloudy matching scene is selected, the three bands are loaded at 10m
over the AOI, and written as GeoTIFFs into the folder layout that
ndvi_stack.py expects:

    scenes/
        2024-04-17/B04.tif  B08.tif  SCL.tif
        2024-06-22/B04.tif  B08.tif  SCL.tif
        ...

The AOI can be given either as a bounding box (--bbox) or as a vector file
(--aoi, e.g. a one-polygon GeoPackage drawn in QGIS). If neither is given,
an ExampleArea.gpkg in the working directory is used automatically.

Example:
    # explicit bbox
    python agriscape/download.py scenes/ \
        --bbox 32.5 39.5 33.5 40.4 \
        --window 2024-04-01/2024-04-30

    # or drive it from a drawn AOI polygon (defaults to ExampleArea.gpkg)
    python agriscape/download.py scenes/ \
        --window 2024-04-01/2024-04-30 \
        --window 2024-07-01/2024-07-31 \
        --window 2024-09-01/2024-09-30
"""

from pathlib import Path

import geopandas as gpd
import planetary_computer
import rioxarray  # noqa: F401 -- registers the .rio accessor on xarray objects
from odc.stac import load as stac_load
from pystac_client import Client

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
BANDS = ["B04", "B08", "SCL"]

# AOI vector used automatically when no --bbox/--aoi is supplied.
DEFAULT_AOI_FILE = "ExampleArea.gpkg"


def bbox_from_vector(path: str | Path) -> list[float]:
    """Read an AOI vector and return its extent as [west, south, east, north]
    in lon/lat (WGS84), reprojecting if needed."""
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI file {path} contains no features")
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    west, south, east, north = gdf.total_bounds
    return [float(west), float(south), float(east), float(north)]


def resolve_bbox(bbox_arg, aoi_path) -> list[float]:
    """Explicit --bbox wins; otherwise fall back to the AOI vector file."""
    if bbox_arg is not None:
        return list(bbox_arg)
    aoi = Path(aoi_path)
    if aoi.exists():
        bbox = bbox_from_vector(aoi)
        print(f"Using AOI from {aoi}: bbox {tuple(round(v, 4) for v in bbox)}")
        return bbox
    raise SystemExit(
        f"No AOI specified. Pass --bbox WEST SOUTH EAST NORTH, or --aoi <file>, "
        f"or place an {DEFAULT_AOI_FILE} in the working directory."
    )


def find_best_item(catalog: Client, bbox: list[float], window: str, max_cloud: float):
    """Return the least-cloudy Sentinel-2 L2A item in the window, or None."""
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=window,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        return None
    return min(items, key=lambda it: it.properties.get("eo:cloud_cover", 100.0))


def download_scene(item, bbox: list[float], out_dir: Path) -> None:
    """Load B04/B08/SCL for one STAC item at 10m and write per-band GeoTIFFs."""
    ds = stac_load(
        [item],
        bands=BANDS,
        bbox=bbox,
        crs="utm",
        resolution=10,
        chunks={"x": 2048, "y": 2048},  # dask-backed: bounded memory on big AOIs
    )
    ds = ds.isel(time=0)

    out_dir.mkdir(parents=True, exist_ok=True)
    for band in BANDS:
        out_path = out_dir / f"{band}.tif"
        ds[band].rio.to_raster(out_path, compress="deflate")
        print(f"  wrote {out_path}")


def download_scenes(
    output_dir: str,
    bbox: list[float],
    windows: list[str],
    max_cloud: float = 20.0,
) -> None:
    output_dir = Path(output_dir)
    catalog = Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    for window in windows:
        print(f"Searching {window} (cloud cover < {max_cloud}%) ...")
        item = find_best_item(catalog, bbox, window, max_cloud)
        if item is None:
            print(f"  WARNING: no scene found for {window}; try widening the "
                  f"window or raising --max-cloud")
            continue

        date = item.datetime.strftime("%Y-%m-%d")
        cloud = item.properties.get("eo:cloud_cover", float("nan"))
        print(f"  selected {item.id} ({date}, {cloud:.1f}% cloud)")
        download_scene(item, bbox, output_dir / date)

    print(f"\nDone. Scenes are ready for: python agriscape/ndvi_stack.py "
          f"{output_dir}/ output/ndvi_stack.tif")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 L2A B04/B08/SCL from Planetary Computer"
    )
    parser.add_argument("output_dir", help="Folder to write per-date scene subfolders")
    parser.add_argument(
        "--bbox", nargs=4, type=float, default=None,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="AOI bounding box in lon/lat (WGS84). Overrides --aoi if both given.",
    )
    parser.add_argument(
        "--aoi", default=DEFAULT_AOI_FILE,
        help=f"AOI vector file (gpkg/shp) whose extent sets the bbox "
             f"(default: {DEFAULT_AOI_FILE} if present)",
    )
    parser.add_argument(
        "--window", action="append", required=True, dest="windows",
        help="Acquisition window as start/end date, e.g. 2024-04-01/2024-04-30. "
             "Repeat for each phenological window (3-8 recommended).",
    )
    parser.add_argument(
        "--max-cloud", type=float, default=20.0,
        help="Maximum scene cloud cover percentage (default 20)",
    )
    args = parser.parse_args()

    bbox = resolve_bbox(args.bbox, args.aoi)
    download_scenes(args.output_dir, bbox, args.windows, args.max_cloud)
