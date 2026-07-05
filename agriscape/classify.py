"""
Stage 3: Train a Random Forest on ground-truth polygons and classify the
full NDVI stack, tile-by-tile so it scales to 100km x 100km areas without
blowing up memory.

Ground truth input: a vector file (GeoPackage/Shapefile) with a "class_name"
field, e.g. polygons labeled "wheat", "barley", "corn", "vineyard", "non_agri".
"""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def extract_training_samples(
    stack_path: str, ground_truth_path: str, class_field: str = "class_name"
):
    """Rasterize ground-truth polygons and pull the NDVI time-series at
    every labeled pixel."""
    gdf = gpd.read_file(ground_truth_path)

    with rasterio.open(stack_path) as src:
        # Ground truth may be digitized in a different CRS (e.g. WGS84 in QGIS)
        if gdf.crs is not None and src.crs is not None and gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        transform = src.transform
        out_shape = (src.height, src.width)
        stack = src.read()  # (n_bands, H, W) -- fine for training extraction

        classes = sorted(gdf[class_field].unique())
        class_to_id = {name: i + 1 for i, name in enumerate(classes)}  # 0 = background

        shapes = [
            (geom, class_to_id[cls])
            for geom, cls in zip(gdf.geometry, gdf[class_field])
        ]
        label_raster = rasterize(
            shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8"
        )

    mask = label_raster > 0
    X = stack[:, mask].T  # (n_samples, n_bands)
    y = label_raster[mask]

    # Drop samples with NaN features (e.g. pixels outside the reprojected
    # footprint of some dates) -- RandomForest cannot handle NaN.
    valid = ~np.isnan(X).any(axis=1)
    if not valid.all():
        print(f"Dropping {np.count_nonzero(~valid)} samples with NaN features")
        X, y = X[valid], y[valid]

    return X, y, class_to_id


def train_model(X, y, model_out: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Validation report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"Model saved to {model_out}")
    return clf


def classify_stack_tiled(
    stack_path: str,
    model_path: str,
    class_to_id: dict,
    output_path: str,
    tile_size: int = 1024,
):
    """Classify the full NDVI stack in tiles to keep memory bounded on
    very large areas (100km x 100km at 10m = 10000x10000 px)."""
    clf = joblib.load(model_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(stack_path) as src:
        profile = src.profile
        out_profile = profile.copy()
        out_profile.update(dtype="uint8", count=1, compress="deflate", predictor=1)

        with rasterio.open(output_path, "w", **out_profile) as dst:
            for row_off in range(0, src.height, tile_size):
                for col_off in range(0, src.width, tile_size):
                    h = min(tile_size, src.height - row_off)
                    w = min(tile_size, src.width - col_off)
                    window = Window(col_off, row_off, w, h)

                    tile = src.read(window=window)  # (n_bands, h, w)
                    n_bands, th, tw = tile.shape
                    flat = tile.reshape(n_bands, -1).T  # (h*w, n_bands)

                    # Pixels with NaN (outside footprint) get class 0 (background)
                    nan_rows = np.isnan(flat).any(axis=1)
                    preds = np.zeros(flat.shape[0], dtype=np.uint8)
                    if (~nan_rows).any():
                        preds[~nan_rows] = clf.predict(flat[~nan_rows]).astype(np.uint8)
                    preds = preds.reshape(th, tw)

                    dst.write(preds, 1, window=window)
                    print(f"Classified tile at row {row_off}, col {col_off}")

    id_to_class = {v: k for k, v in class_to_id.items()}
    with open(Path(output_path).with_suffix(".json"), "w") as f:
        json.dump(id_to_class, f, indent=2)

    print(f"Classification complete: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RF and classify NDVI stack")
    parser.add_argument("stack_path", help="Path to multi-temporal NDVI stack GeoTIFF")
    parser.add_argument("ground_truth", help="Path to labeled polygons (gpkg/shp)")
    parser.add_argument("output", help="Path for output classified raster")
    parser.add_argument("--class-field", default="class_name")
    parser.add_argument("--model-out", default="rf_model.joblib")
    parser.add_argument("--tile-size", type=int, default=1024)
    args = parser.parse_args()

    X, y, class_to_id = extract_training_samples(
        args.stack_path, args.ground_truth, args.class_field
    )
    print(f"Extracted {X.shape[0]} training samples across {len(class_to_id)} classes")
    print("Class mapping:", class_to_id)

    clf = train_model(X, y, args.model_out)

    classify_stack_tiled(
        args.stack_path,
        args.model_out,
        class_to_id,
        args.output,
        tile_size=args.tile_size,
    )
