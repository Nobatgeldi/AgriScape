"""
Stage 1-2 (enriched): build a rich per-pixel feature stack instead of a plain
NDVI stack, so the Random Forest has real context to learn from rather than a
single vegetation index.

Feature groups (each toggleable):

  * spectral  -- raw surface reflectance per date for every band present:
                 blue, green, red, NIR, SWIR1, SWIR2 (whatever was downloaded),
                 plus the NDVI itself.
  * spatial   -- local texture: the standard deviation of NDVI in a small
                 window (default 5x5) per date. Homogeneous fields have low
                 texture; mixed/edge pixels high. (A fast, scalable stand-in
                 for GLCM texture.)
  * temporal  -- season-long summary of the NDVI time series per pixel:
                 mean, min, max, amplitude (max-min), and linear trend slope.

Output: a single multi-band float32 GeoTIFF (band = feature) plus a JSON
sidecar listing feature names. classify.py and export_weightmaps.py consume it
unchanged -- they already treat every band as a feature.

Example:
    python agriscape/build_features.py scenes/ output/feature_stack.tif
    python agriscape/build_features.py scenes/ output/feature_stack.tif \
        --window-size 3 --no-spectral
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter

from ndvi_stack import (
    SCL_BAD_VALUES,
    align_to_reference,
    compute_ndvi,
    fill_gaps_median,
    find_band_file,
)

# (file band id, friendly feature name). Red+NIR are always needed for NDVI;
# the others are optional extra spectral context and skipped if not downloaded.
SPECTRAL_BANDS = [
    ("B02", "blue"),
    ("B03", "green"),
    ("B04", "red"),
    ("B08", "nir"),
    ("B11", "swir1"),
    ("B12", "swir2"),
]


def load_band_aligned(date_dir: Path, band_id: str, ref_profile: dict | None):
    """Load one band, scale to reflectance, and align to the reference grid.
    Returns (array, profile) or (None, None) if the band file is absent."""
    try:
        path = find_band_file(date_dir, band_id)
    except FileNotFoundError:
        return None, None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile
    arr = arr / 10000.0  # Sentinel-2 L2A DN -> surface reflectance
    if ref_profile is not None:
        arr = align_to_reference(arr, profile, ref_profile)
    return arr, profile


def load_scl_bad_mask(date_dir: Path, ref_profile: dict) -> np.ndarray | None:
    """Return a boolean 'bad pixel' mask (cloud/shadow/snow) on the ref grid."""
    try:
        scl_path = find_band_file(date_dir, "SCL")
    except FileNotFoundError:
        return None
    h, w = ref_profile["height"], ref_profile["width"]
    with rasterio.open(scl_path) as src:
        scl = src.read(1, out_shape=(h, w), resampling=Resampling.nearest)
    return np.isin(scl, list(SCL_BAD_VALUES))


def local_std(arr: np.ndarray, size: int) -> np.ndarray:
    """Local standard deviation in a size x size window (texture proxy).
    Input must be NaN-free (call after temporal gap-fill)."""
    arr = arr.astype(np.float32)
    mean = uniform_filter(arr, size=size, mode="reflect")
    sq_mean = uniform_filter(arr * arr, size=size, mode="reflect")
    var = np.clip(sq_mean - mean * mean, 0.0, None)
    return np.sqrt(var, dtype=np.float32)


def temporal_features(ndvi_stack: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Per-pixel summary of the NDVI time series (input already gap-filled)."""
    n = ndvi_stack.shape[0]
    feats = [
        ("ndvi_mean", ndvi_stack.mean(axis=0)),
        ("ndvi_min", ndvi_stack.min(axis=0)),
        ("ndvi_max", ndvi_stack.max(axis=0)),
        ("ndvi_amplitude", ndvi_stack.max(axis=0) - ndvi_stack.min(axis=0)),
    ]
    if n >= 2:
        # OLS slope of NDVI vs. time index, computed per pixel in closed form.
        t = np.arange(n, dtype=np.float32)
        t_bar = t.mean()
        denom = float(((t - t_bar) ** 2).sum())
        y_bar = ndvi_stack.mean(axis=0)
        num = ((t[:, None, None] - t_bar) * (ndvi_stack - y_bar)).sum(axis=0)
        feats.append(("ndvi_slope", (num / denom).astype(np.float32)))
    return [(name, arr.astype(np.float32)) for name, arr in feats]


def build_feature_stack(
    scenes_dir: str,
    output_path: str,
    window_size: int = 5,
    spectral: bool = True,
    spatial: bool = True,
    temporal: bool = True,
) -> None:
    scenes_dir = Path(scenes_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    date_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
    if not date_dirs:
        raise ValueError(f"No date folders found in {scenes_dir}")

    # Reference grid = first date's red band.
    _, ref_profile = load_band_aligned(date_dirs[0], "B04", ref_profile=None)
    if ref_profile is None:
        raise FileNotFoundError(f"No B04 (red) band in {date_dirs[0]}")

    # Which optional spectral bands are actually present in the first date?
    present_bands = []
    for band_id, name in SPECTRAL_BANDS:
        try:
            find_band_file(date_dirs[0], band_id)
            present_bands.append((band_id, name))
        except FileNotFoundError:
            if spectral:
                print(f"Note: band {band_id} ({name}) not found; skipping it.")

    dates = [d.name for d in date_dirs]
    # Collect raw per-date arrays (with NaN where masked) before gap-filling.
    ndvi_by_date = []
    spectral_by_date = {name: [] for _bid, name in present_bands}

    for date_dir in date_dirs:
        print(f"Loading {date_dir.name} ...")
        bad = load_scl_bad_mask(date_dir, ref_profile)

        band_arrays = {}
        for band_id, name in present_bands:
            arr, _ = load_band_aligned(date_dir, band_id, ref_profile)
            if bad is not None:
                arr = np.where(bad, np.nan, arr)
            band_arrays[name] = arr

        red = band_arrays.get("red")
        nir = band_arrays.get("nir")
        if red is None or nir is None:
            # Fall back to loading them just for NDVI even if spectral is off.
            red, _ = load_band_aligned(date_dir, "B04", ref_profile)
            nir, _ = load_band_aligned(date_dir, "B08", ref_profile)
            if bad is not None:
                red = np.where(bad, np.nan, red)
                nir = np.where(bad, np.nan, nir)
        ndvi = compute_ndvi(red, nir)
        if bad is not None:
            ndvi = np.where(bad, np.nan, ndvi)
        ndvi_by_date.append(ndvi)

        if spectral:
            for name in spectral_by_date:
                spectral_by_date[name].append(band_arrays[name])

    # Gap-fill each per-date series with its temporal median.
    ndvi_stack = fill_gaps_median(np.stack(ndvi_by_date, axis=0).astype(np.float32))
    spectral_stacks = {
        name: fill_gaps_median(np.stack(arrs, axis=0).astype(np.float32))
        for name, arrs in spectral_by_date.items()
    }

    # Assemble features in a stable order.
    features: list[tuple[str, np.ndarray]] = []
    for i, date in enumerate(dates):
        if spectral:
            for _bid, name in present_bands:
                features.append((f"{name}_{date}", spectral_stacks[name][i]))
        features.append((f"ndvi_{date}", ndvi_stack[i]))
        if spatial:
            features.append(
                (f"ndvi_std{window_size}_{date}",
                 local_std(ndvi_stack[i], window_size))
            )
    if temporal:
        features.extend(temporal_features(ndvi_stack))

    names = [name for name, _ in features]
    stack = np.stack([arr for _, arr in features], axis=0).astype(np.float32)

    out_profile = ref_profile.copy()
    out_profile.update(
        driver="GTiff", dtype="float32", count=stack.shape[0],
        compress="deflate", predictor=3,
    )
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(stack)
        for i, name in enumerate(names, start=1):
            dst.set_band_description(i, name)

    sidecar = {"features": names, "n_features": len(names), "band_dates": dates}
    with open(output_path.with_suffix(".json"), "w") as f:
        json.dump(sidecar, f, indent=2)

    print(f"\nWrote {len(names)}-feature stack to {output_path}")
    print("Feature groups: "
          f"spectral={spectral} ({len(present_bands)} bands), "
          f"spatial={spatial} (std {window_size}x{window_size}), "
          f"temporal={temporal}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build an enriched feature stack")
    parser.add_argument("scenes_dir", help="Folder of per-date scene subfolders")
    parser.add_argument("output", help="Output path for the feature stack GeoTIFF")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Window (px) for local-texture std (default 5)")
    parser.add_argument("--no-spectral", action="store_true",
                        help="Drop raw reflectance bands (keep NDVI)")
    parser.add_argument("--no-spatial", action="store_true",
                        help="Drop local-texture features")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Drop time-series summary features")
    args = parser.parse_args()

    build_feature_stack(
        args.scenes_dir,
        args.output,
        window_size=args.window_size,
        spectral=not args.no_spectral,
        spatial=not args.no_spatial,
        temporal=not args.no_temporal,
    )
