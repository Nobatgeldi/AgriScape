"""
Stage 1-2: Build a multi-temporal NDVI stack from Sentinel-2 L2A scenes.

Expects a folder structure like:
    scenes/
        2024-04-15/B04.jp2  B08.jp2  SCL.jp2
        2024-06-20/B04.tif  B08.tif  SCL.tif
        2024-09-10/B04.jp2  B08.jp2  SCL.jp2

(.jp2 from manual Copernicus downloads and .tif from download.py both work.)

Each date folder = one atmospherically corrected acquisition. Scenes that do
not share the CRS/grid of the first date are reprojected onto it, so mixed
UTM zones or differing extents are handled automatically.

Cloud-masked gaps are filled with the per-pixel temporal median; optionally
the whole time series is then smoothed with a Savitzky-Golay filter
(--smooth savgol) for cleaner phenological curves.

Output: a single multi-band GeoTIFF where band i = NDVI for date i,
plus a JSON sidecar recording which date corresponds to which band index.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject, Resampling as WarpResampling

# SCL classes considered "bad" and masked out (cloud, cloud shadow, snow, etc.)
# See ESA Sentinel-2 SCL legend.
SCL_BAD_VALUES = {0, 1, 3, 8, 9, 10, 11}

BAND_EXTENSIONS = (".jp2", ".tif", ".tiff")


def find_band_file(date_dir: Path, band: str) -> Path:
    """Locate a band file regardless of extension (B04.jp2, B04.tif, ...)."""
    for ext in BAND_EXTENSIONS:
        candidate = date_dir / f"{band}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No {band} file ({'/'.join(BAND_EXTENSIONS)}) in {date_dir}"
    )


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red), safe against divide-by-zero."""
    red = red.astype(np.float32)
    nir = nir.astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom == 0, np.nan, (nir - red) / np.where(denom == 0, 1, denom))
    return ndvi


def load_masked_ndvi(date_dir: Path) -> tuple[np.ndarray, dict]:
    """Load B04/B08/SCL for one date and return cloud-masked NDVI + profile."""
    b04_path = find_band_file(date_dir, "B04")
    b08_path = find_band_file(date_dir, "B08")

    with rasterio.open(b04_path) as src:
        red = src.read(1)
        profile = src.profile

    with rasterio.open(b08_path) as src:
        nir = src.read(1)

    ndvi = compute_ndvi(red, nir)

    try:
        scl_path = find_band_file(date_dir, "SCL")
    except FileNotFoundError:
        scl_path = None

    if scl_path is not None:
        with rasterio.open(scl_path) as src:
            # SCL is usually 20m; resample to 10m to match B04/B08
            scl = src.read(
                1,
                out_shape=(red.shape[0], red.shape[1]),
                resampling=Resampling.nearest,
            )
        bad_mask = np.isin(scl, list(SCL_BAD_VALUES))
        ndvi = np.where(bad_mask, np.nan, ndvi)

    return ndvi, profile


def align_to_reference(
    ndvi: np.ndarray, profile: dict, ref_profile: dict
) -> np.ndarray:
    """Reproject one date's NDVI onto the reference grid if it differs."""
    same_grid = (
        profile["crs"] == ref_profile["crs"]
        and profile["transform"] == ref_profile["transform"]
        and profile["height"] == ref_profile["height"]
        and profile["width"] == ref_profile["width"]
    )
    if same_grid:
        return ndvi

    aligned = np.full(
        (ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32
    )
    reproject(
        source=ndvi.astype(np.float32),
        destination=aligned,
        src_transform=profile["transform"],
        src_crs=profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=WarpResampling.bilinear,
    )
    return aligned


def fill_gaps_median(stack: np.ndarray) -> np.ndarray:
    """Fill NaNs (persistent cloud) with the per-pixel temporal median.
    Pixels that are NaN on every date (outside the scene footprint) become 0."""
    if not np.isnan(stack).any():
        return stack
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        pixel_median = np.nanmedian(stack, axis=0)
    pixel_median = np.nan_to_num(pixel_median, nan=0.0)
    for i in range(stack.shape[0]):
        band = stack[i]
        band[np.isnan(band)] = pixel_median[np.isnan(band)]
        stack[i] = band
    return stack


def smooth_savgol(stack: np.ndarray) -> np.ndarray:
    """Savitzky-Golay smoothing along the time axis for cleaner phenological
    curves. Requires at least 4 dates; otherwise returns the stack unchanged."""
    from scipy.signal import savgol_filter

    n_dates = stack.shape[0]
    if n_dates < 4:
        print(f"Savitzky-Golay skipped: needs >= 4 dates, got {n_dates}")
        return stack

    window = min(5, n_dates if n_dates % 2 == 1 else n_dates - 1)
    polyorder = min(2, window - 1)
    print(f"Applying Savitzky-Golay smoothing (window={window}, order={polyorder})")
    return savgol_filter(stack, window_length=window, polyorder=polyorder, axis=0)


def build_stack(scenes_dir: str, output_path: str, smooth: str = "median") -> None:
    scenes_dir = Path(scenes_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    date_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])

    if not date_dirs:
        raise ValueError(f"No date folders found in {scenes_dir}")

    ndvi_bands = []
    dates = []
    ref_profile = None

    for date_dir in date_dirs:
        print(f"Processing {date_dir.name} ...")
        ndvi, prof = load_masked_ndvi(date_dir)

        if ref_profile is None:
            ref_profile = prof
        else:
            ndvi = align_to_reference(ndvi, prof, ref_profile)

        ndvi_bands.append(ndvi)
        dates.append(date_dir.name)

    stack = np.stack(ndvi_bands, axis=0).astype(np.float32)  # (n_dates, H, W)

    stack = fill_gaps_median(stack)
    if smooth == "savgol":
        stack = smooth_savgol(stack)

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff",
        dtype="float32",
        count=stack.shape[0],
        compress="deflate",
        predictor=3,
    )

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(stack.astype(np.float32))
        for i, date in enumerate(dates, start=1):
            dst.set_band_description(i, date)

    sidecar = {"band_dates": dates, "n_bands": len(dates)}
    with open(Path(output_path).with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"Wrote {stack.shape[0]}-band NDVI stack to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build multi-temporal NDVI stack")
    parser.add_argument("scenes_dir", help="Folder containing per-date scene subfolders")
    parser.add_argument("output", help="Output path for the NDVI stack GeoTIFF")
    parser.add_argument(
        "--smooth", choices=["median", "savgol"], default="median",
        help="Gap handling: 'median' fills cloud gaps only; 'savgol' additionally "
             "smooths the temporal curve (needs >= 4 dates)",
    )
    args = parser.parse_args()

    build_stack(args.scenes_dir, args.output, smooth=args.smooth)
