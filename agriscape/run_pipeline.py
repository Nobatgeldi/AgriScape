"""
End-to-end pipeline runner: executes every AgriScape stage in order.

    download  ->  ndvi  ->  [ground truth]  ->  classify  ->  weightmaps

The one step that cannot be fully automated is the ground truth: the Random
Forest needs hand-labeled wheat/barley/corn polygons (crop-type labels are not
open data). So the runner behaves like this at the ground-truth gate:

  * if the ground-truth GeoPackage already exists  -> continue to classify;
  * elif --bootstrap-osm is set  -> build vineyard/orchard/non_agri polygons
      from OpenStreetMap, then STOP so you can add cereal fields in QGIS;
  * else  -> STOP with instructions.

After you've labeled polygons, resume with:  --start-from classify

Examples:
    # first pass: download + NDVI + OSM bootstrap, then stop for labeling
    python agriscape/run_pipeline.py \
        --window 2024-04-01/2024-04-30 \
        --window 2024-07-01/2024-07-31 \
        --window 2024-09-01/2024-09-30 \
        --bootstrap-osm --max-per-class 500

    # ...draw wheat/barley/corn polygons in QGIS into ground_truth.gpkg...

    # second pass: classify + export weightmaps
    python agriscape/run_pipeline.py --start-from classify
"""

import sys
from pathlib import Path

# Allow both `python agriscape/run_pipeline.py` and `python -m agriscape.run_pipeline`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import download          # noqa: E402
import ndvi_stack        # noqa: E402
import build_features    # noqa: E402
import deep_features     # noqa: E402
import classify          # noqa: E402
import export_weightmaps  # noqa: E402
import fetch_osm_labels  # noqa: E402

STEPS = ["download", "ndvi", "classify", "weightmaps"]


def banner(msg: str) -> None:
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def run(cfg) -> None:
    out = Path(cfg.output_dir)
    scenes_dir = Path(cfg.scenes_dir)
    stack_path = out / "ndvi_stack.tif"
    classified_path = out / "classified.tif"
    class_map_path = out / "classified.json"
    model_path = out / "rf_model.joblib"
    weightmaps_dir = out / "weightmaps"
    ground_truth = Path(cfg.ground_truth)

    start_idx = STEPS.index(cfg.start_from)
    stop_idx = STEPS.index(cfg.stop_after)
    if stop_idx < start_idx:
        raise SystemExit(f"--stop-after ({cfg.stop_after}) is before "
                         f"--start-from ({cfg.start_from})")

    def active(step: str) -> bool:
        return start_idx <= STEPS.index(step) <= stop_idx

    # --- Stage 0: download --------------------------------------------------
    if active("download"):
        banner("STAGE 0/4  Download Sentinel-2 scenes")
        if not cfg.windows:
            raise SystemExit("--window is required to run the download stage "
                             "(or use --start-from ndvi if scenes exist)")
        bbox = download.resolve_bbox(cfg.bbox, cfg.aoi)
        download.download_scenes(str(scenes_dir), bbox, cfg.windows, cfg.max_cloud)
    else:
        print(f"Skipping download (start-from={cfg.start_from})")

    # --- Stage 1-2: feature stack ------------------------------------------
    if active("ndvi"):
        if cfg.features == "full":
            banner("STAGE 1-2/4  Build enriched feature stack "
                   "(spectral + spatial + temporal)")
            build_features.build_feature_stack(
                str(scenes_dir), str(stack_path), window_size=cfg.window_size,
            )
        else:
            banner("STAGE 1-2/4  Build multi-temporal NDVI stack")
            ndvi_stack.build_stack(str(scenes_dir), str(stack_path), smooth=cfg.smooth)

        if cfg.deep_features:
            banner("STAGE 1-2/4  Append deep ResNet-18 features (TorchGeo)")
            deep_features.add_deep_features(
                str(scenes_dir), str(stack_path),
                weights=cfg.deep_weights, checkpoint=cfg.deep_checkpoint,
                n_components=cfg.deep_components, tile_size=cfg.deep_tile_size,
                device=cfg.deep_device,
            )
    else:
        print(f"Skipping feature stack (start-from={cfg.start_from})")

    # --- Ground-truth gate --------------------------------------------------
    if active("classify"):
        if not ground_truth.exists():
            if cfg.bootstrap_osm:
                banner("GROUND TRUTH  Bootstrapping vineyard/orchard/non_agri from OSM")
                fetch_osm_labels.fetch_osm_labels(
                    str(stack_path), str(ground_truth),
                    max_per_class=cfg.max_per_class,
                )
                print("\n" + "-" * 70)
                print(f"Ground truth seeded at {ground_truth}, but it has NO "
                      f"wheat/barley/corn samples yet.")
                print("Open it in QGIS, draw those crop fields (class_name = "
                      "'wheat'/'barley'/'corn'), then resume with:")
                print(f"    python agriscape/run_pipeline.py --start-from classify")
                print("-" * 70)
                return
            raise SystemExit(
                f"Ground truth {ground_truth} not found. Either:\n"
                f"  * digitize labeled polygons in QGIS (field 'class_name'), or\n"
                f"  * re-run with --bootstrap-osm to seed non-crop classes from OSM.\n"
                f"Then resume with --start-from classify."
            )

        # --- Stage 3: classify ---------------------------------------------
        banner("STAGE 3/4  Train Random Forest & classify")
        X, y, class_to_id = classify.extract_training_samples(
            str(stack_path), str(ground_truth), cfg.class_field
        )
        print(f"Extracted {X.shape[0]} samples across {len(class_to_id)} classes")
        print("Class mapping:", class_to_id)
        classify.train_model(X, y, str(model_path))
        classify.classify_stack_tiled(
            str(stack_path), str(model_path), class_to_id,
            str(classified_path), tile_size=cfg.tile_size,
        )
    else:
        print(f"Skipping classify (stop-after={cfg.stop_after})")

    # --- Stage 4: weightmaps -----------------------------------------------
    if active("weightmaps"):
        banner("STAGE 4/4  Export Unreal weightmaps")
        export_weightmaps.export_weightmaps(
            str(classified_path), str(class_map_path), str(weightmaps_dir),
            bit_depth=cfg.bit_depth, target_size=cfg.target_size,
        )

    banner("PIPELINE COMPLETE")
    print(f"NDVI stack : {stack_path}")
    print(f"Classified : {classified_path}")
    print(f"Weightmaps : {weightmaps_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the full AgriScape pipeline in order",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # AOI / acquisition
    parser.add_argument("--aoi", default=download.DEFAULT_AOI_FILE,
                        help="AOI vector file whose extent sets the download bbox")
    parser.add_argument("--bbox", nargs=4, type=float, default=None,
                        metavar=("W", "S", "E", "N"),
                        help="AOI bbox in lon/lat (overrides --aoi)")
    parser.add_argument("--window", action="append", dest="windows", default=[],
                        help="Acquisition window start/end, e.g. 2024-04-01/2024-04-30 "
                             "(repeat 3-8 times)")
    parser.add_argument("--max-cloud", type=float, default=20.0)
    # paths
    parser.add_argument("--scenes-dir", default="scenes")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--ground-truth", default="ground_truth.gpkg")
    # stage params
    parser.add_argument("--features", choices=["full", "ndvi"], default="full",
                        help="'full' = enriched spectral+spatial+temporal feature "
                             "stack; 'ndvi' = plain NDVI-only stack")
    parser.add_argument("--window-size", type=int, default=5,
                        help="Texture window (px) for the enriched feature stack")
    # deep features (optional, needs requirements-torch.txt + GPU)
    parser.add_argument("--deep-features", action="store_true",
                        help="Append frozen ResNet-18 (TorchGeo) deep bands to "
                             "the feature stack (needs requirements-torch.txt)")
    parser.add_argument("--deep-weights", default="sentinel2_rgb_moco",
                        help="TorchGeo ResNet18_Weights name for deep features")
    parser.add_argument("--deep-checkpoint", default=None,
                        help="Optional EuroSAT-fine-tuned checkpoint for deep features")
    parser.add_argument("--deep-components", type=int, default=16,
                        help="Number of PCA deep-feature bands (default 16)")
    parser.add_argument("--deep-tile-size", type=int, default=512)
    parser.add_argument("--deep-device", default="cuda", help="cuda or cpu")
    parser.add_argument("--smooth", choices=["median", "savgol"], default="median",
                        help="NDVI-only gap handling (used when --features ndvi)")
    parser.add_argument("--class-field", default="class_name")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8)
    parser.add_argument("--target-size", type=int, default=None)
    # ground-truth bootstrap
    parser.add_argument("--bootstrap-osm", action="store_true",
                        help="If ground truth is missing, seed non-crop classes "
                             "from OpenStreetMap, then stop for manual labeling")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap OSM polygons per class (used with --bootstrap-osm)")
    # flow control
    parser.add_argument("--start-from", choices=STEPS, default="download",
                        help="Resume the pipeline from this stage")
    parser.add_argument("--stop-after", choices=STEPS, default="weightmaps",
                        help="Stop the pipeline after this stage")
    cfg = parser.parse_args()

    run(cfg)
