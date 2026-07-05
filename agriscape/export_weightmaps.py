"""
Stage 4: Turn a classified raster into per-class grayscale weightmaps sized
for Unreal Engine's Landscape system.

Unreal landscape resolution must satisfy:
    size = (quads_per_component * components_per_side) + 1
Recommended valid sizes: 127, 253, 505, 1009, 2017, 4033, 8129.
This script resamples/pads the classified raster to the nearest valid size
and writes one PNG per class (255 = fully that class, 0 = not that class).
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# Total landscape quads per side (63-quad components x 2/4/8/16/32/64/128
# components) before the +1, per Epic's recommended landscape sizes.
VALID_QUADS_PER_SIDE = [126, 252, 504, 1008, 2016, 4032, 8128]


def nearest_valid_unreal_size(pixels: int) -> int:
    """Find the smallest Unreal-valid size >= the requested pixel count."""
    for quads in VALID_QUADS_PER_SIDE:
        size = quads + 1
        if size >= pixels:
            return size
    # fall back to the largest supported size
    return VALID_QUADS_PER_SIDE[-1] + 1


def resample_to_size(array: np.ndarray, target_size: int) -> np.ndarray:
    """Nearest-neighbor resample a 2D class-id array to a square target size.
    Uses nearest neighbor to avoid inventing fractional class ids."""
    img = Image.fromarray(array)
    resized = img.resize((target_size, target_size), resample=Image.NEAREST)
    return np.array(resized)


def export_weightmaps(
    classified_path: str,
    class_map_path: str,
    output_dir: str,
    bit_depth: int = 8,
    target_size: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(class_map_path) as f:
        id_to_class = {int(k): v for k, v in json.load(f).items()}

    with rasterio.open(classified_path) as src:
        classified = src.read(1)

    h, w = classified.shape
    longest_side = max(h, w)
    if target_size is None:
        target_size = nearest_valid_unreal_size(longest_side)

    print(f"Resampling {w}x{h} raster to {target_size}x{target_size} (Unreal-valid size)")
    classified_resized = resample_to_size(classified, target_size)

    max_val = 65535 if bit_depth == 16 else 255
    dtype = np.uint16 if bit_depth == 16 else np.uint8

    for class_id, class_name in id_to_class.items():
        mask = (classified_resized == class_id).astype(dtype) * max_val
        out_path = output_dir / f"weightmap_{class_name}.png"
        Image.fromarray(mask).save(out_path)
        pct = 100 * np.count_nonzero(mask) / mask.size
        print(f"  {class_name}: {out_path.name}  ({pct:.1f}% coverage)")

    print(f"\nAll weightmaps exported to {output_dir}")
    print(f"Import size for Unreal Landscape: {target_size} x {target_size}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export Unreal-ready weightmaps")
    parser.add_argument("classified_raster", help="Path to classified GeoTIFF")
    parser.add_argument("class_map", help="Path to class-id JSON sidecar")
    parser.add_argument("output_dir", help="Directory to write PNG weightmaps")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8)
    parser.add_argument("--target-size", type=int, default=None,
                         help="Force a specific square output size")
    args = parser.parse_args()

    export_weightmaps(
        args.classified_raster,
        args.class_map,
        args.output_dir,
        bit_depth=args.bit_depth,
        target_size=args.target_size,
    )
