"""
Optional stage: augment the feature stack with deep "context" bands from a
frozen, Sentinel-2-pretrained ResNet-18 (TorchGeo).

The Random Forest crop classifier only sees hand-engineered indices. This adds
a learned representation of what fields / roads / built-up / orchards look like:
we run the frozen backbone densely over the RGB imagery, take an intermediate
convolutional feature map (spatial context), reduce it with PCA, and append the
components as extra bands (`deep_pca_00..K-1`) to the existing feature stack.
classify.py / export_weightmaps.py consume them unchanged.

This does NOT turn the model into a crop classifier -- EuroSAT-style scene
classes can't separate wheat/barley/corn. It just gives the RF richer texture /
land-cover context to split on.

Example:
    python agriscape/deep_features.py scenes/ output/feature_stack.tif \
        --components 16 --device cuda
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from sklearn.decomposition import PCA

from build_features import load_band_aligned
from ndvi_stack import fill_gaps_median, find_band_file

# torch / torchgeo are heavy optional deps (see requirements-torch.txt); import
# lazily inside functions so the rest of the package works without them.

# RGB feed order for TorchGeo Sentinel-2 RGB weights: red, green, blue.
RGB_BANDS = ["B04", "B03", "B02"]
CUT_STRIDES = {"layer1": 4, "layer2": 8, "layer3": 16, "layer4": 32}


def _load_rgb_images(scenes_dir: Path, per_date: bool):
    """Return (images, ref_profile) where images is (n_img, 3, H, W) float32,
    per-band z-scored. n_img == 1 (temporal-median composite) unless per_date."""
    date_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir()])
    if not date_dirs:
        raise ValueError(f"No date folders in {scenes_dir}")

    _, ref_profile = load_band_aligned(date_dirs[0], "B04", ref_profile=None)
    if ref_profile is None:
        raise FileNotFoundError(f"No B04 (red) band in {date_dirs[0]}")

    # Confirm RGB is available (deep features need blue/green/red).
    for b in RGB_BANDS:
        find_band_file(date_dirs[0], b)  # raises if missing

    # Per date, stack the 3 aligned reflectance bands -> (n_dates, 3, H, W).
    per_date_rgb = []
    for date_dir in date_dirs:
        chans = []
        for b in RGB_BANDS:
            arr, _ = load_band_aligned(date_dir, b, ref_profile)
            chans.append(arr)
        per_date_rgb.append(np.stack(chans, axis=0))  # (3, H, W)
    cube = np.stack(per_date_rgb, axis=0).astype(np.float32)  # (n_dates, 3, H, W)

    if per_date:
        # Gap-fill each channel's time series, keep every date as its own image.
        for c in range(3):
            cube[:, c] = fill_gaps_median(cube[:, c])
        images = cube
    else:
        # Temporal-median composite -> a single robust cloud-free RGB image.
        images = np.nanmedian(cube, axis=0)[None, ...]  # (1, 3, H, W)
        images = np.nan_to_num(images, nan=0.0)

    # Per-image, per-band z-score: robust, deterministic normalization for a
    # frozen feature extractor (exact pretraining stats not required here).
    for i in range(images.shape[0]):
        for c in range(3):
            band = images[i, c]
            mean, std = float(np.nanmean(band)), float(np.nanstd(band))
            images[i, c] = (band - mean) / (std + 1e-6)
    images = np.nan_to_num(images, nan=0.0).astype(np.float32)
    return images, ref_profile


def _build_extractor(weights_name, checkpoint, cut_layer, device):
    """Return (forward_fn, n_channels): forward_fn(tensor[B,3,h,w]) -> feature
    map [B,C,h',w'] taken at cut_layer."""
    import torch
    from torchgeo.models import ResNet18_Weights, resnet18

    if checkpoint:
        model = resnet18(weights=None)
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        # Strip Lightning/task prefixes and drop the classifier head.
        cleaned = {}
        for k, v in state.items():
            nk = k.replace("model.", "").replace("backbone.", "")
            if nk.startswith("fc.") or nk.startswith("head."):
                continue
            cleaned[nk] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"Loaded checkpoint {checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
    else:
        weights = getattr(ResNet18_Weights, weights_name.upper())
        model = resnet18(weights=weights)
        print(f"Loaded TorchGeo weights ResNet18_Weights.{weights_name.upper()}")

    model = model.to(device).eval()

    def stem(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.act1(x) if hasattr(model, "act1") else model.relu(x)
        x = model.maxpool(x)
        return x

    order = ["layer1", "layer2", "layer3", "layer4"]
    cut_idx = order.index(cut_layer)

    def forward_fn(x):
        x = stem(x)
        for name in order[: cut_idx + 1]:
            x = getattr(model, name)(x)
        return x

    # Probe channel count with a tiny dummy pass.
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        n_channels = forward_fn(dummy).shape[1]
    return forward_fn, n_channels


def _iter_tiles(height, width, tile_size, halo):
    """Yield (core_window, read_window) pairs; read_window includes the halo."""
    from rasterio.windows import Window
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            th = min(tile_size, height - row)
            tw = min(tile_size, width - col)
            r0, c0 = max(0, row - halo), max(0, col - halo)
            r1 = min(height, row + th + halo)
            c1 = min(width, col + tw + halo)
            core = Window(col, row, tw, th)
            read = Window(c0, r0, c1 - c0, r1 - r0)
            # offset of the core inside the read window
            off = (row - r0, col - c0)
            yield core, read, off, (th, tw)


def _tile_features(forward_fn, images, read_win, off, core_hw, device):
    """Dense feature map for the core region of one tile, concatenated across
    input images. Returns (C_total, th, tw) float32 on CPU."""
    import torch
    import torch.nn.functional as F

    r0, c0 = int(read_win.row_off), int(read_win.col_off)
    h, w = int(read_win.height), int(read_win.width)
    oy, ox = off
    th, tw = core_hw

    feats = []
    for i in range(images.shape[0]):
        patch = images[i, :, r0:r0 + h, c0:c0 + w]  # (3, h, w)
        x = torch.from_numpy(patch).unsqueeze(0).to(device)
        with torch.inference_mode():
            fmap = forward_fn(x)  # (1, C, h', w')
            fmap = F.interpolate(fmap, size=(h, w), mode="bilinear",
                                 align_corners=False)
        fmap = fmap[0].cpu().numpy()  # (C, h, w)
        feats.append(fmap[:, oy:oy + th, ox:ox + tw])  # crop halo -> core
    return np.concatenate(feats, axis=0).astype(np.float32)


def add_deep_features(
    scenes_dir: str,
    stack_path: str,
    weights: str = "sentinel2_rgb_moco",
    checkpoint: str | None = None,
    n_components: int = 16,
    tile_size: int = 512,
    cut_layer: str = "layer2",
    per_date: bool = False,
    device: str = "cuda",
    pca_sample: int = 50000,
    seed: int = 42,
) -> None:
    import torch

    scenes_dir = Path(scenes_dir)
    stack_path = Path(stack_path)
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; falling back to CPU (slow).")
        device = "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Loading RGB imagery ...")
    images, ref_profile = _load_rgb_images(scenes_dir, per_date)
    H, W = images.shape[2], images.shape[3]

    with rasterio.open(stack_path) as src:
        if (src.height, src.width) != (H, W):
            raise ValueError(
                f"Stack grid {(src.height, src.width)} != scene grid {(H, W)}; "
                f"deep_features must run on the stack built from these scenes."
            )
        orig_count = src.count
        stack_profile = src.profile

    forward_fn, n_ch = _build_extractor(weights, checkpoint, cut_layer, device)
    n_img = images.shape[0]
    print(f"Feature extractor: {cut_layer} -> {n_ch} ch/image x {n_img} image(s) "
          f"= {n_ch * n_img} raw dims -> PCA {n_components}")

    halo = 4 * CUT_STRIDES[cut_layer]
    tiles = list(_iter_tiles(H, W, tile_size, halo))

    # --- Pass 1: fit PCA on a random pixel subsample ------------------------
    rng = np.random.default_rng(seed)
    per_tile = max(1, pca_sample // len(tiles))
    samples = []
    for core, read, off, core_hw in tiles:
        feat = _tile_features(forward_fn, images, read, off, core_hw, device)
        c, th, tw = feat.shape
        flat = feat.reshape(c, -1).T  # (th*tw, C)
        take = min(per_tile, flat.shape[0])
        idx = rng.choice(flat.shape[0], size=take, replace=False)
        samples.append(flat[idx])
    samples = np.concatenate(samples, axis=0)
    n_components = min(n_components, samples.shape[1])
    pca = PCA(n_components=n_components, random_state=seed).fit(samples)
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"PCA fit on {samples.shape[0]} px; {n_components} comps capture "
          f"{evr:.1%} variance")

    # --- Write augmented stack: original bands + K PCA bands ----------------
    deep_names = [f"deep_pca_{i:02d}" for i in range(n_components)]
    out_profile = stack_profile.copy()
    out_profile.update(count=orig_count + n_components)

    fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=str(stack_path.parent))
    os.close(fd)
    with rasterio.open(stack_path) as src, \
            rasterio.open(tmp_path, "w", **out_profile) as dst:
        for b in range(1, orig_count + 1):  # copy original bands
            dst.write(src.read(b), b)
            if src.descriptions[b - 1]:
                dst.set_band_description(b, src.descriptions[b - 1])
        # Pass 2: transform each tile and write the K deep bands.
        for core, read, off, core_hw in tiles:
            feat = _tile_features(forward_fn, images, read, off, core_hw, device)
            c, th, tw = feat.shape
            comps = pca.transform(feat.reshape(c, -1).T)  # (th*tw, K)
            comps = comps.T.reshape(n_components, th, tw).astype(np.float32)
            for k in range(n_components):
                dst.write(comps[k], orig_count + 1 + k, window=core)
        for k, name in enumerate(deep_names):
            dst.set_band_description(orig_count + 1 + k, name)

    os.replace(tmp_path, stack_path)

    # --- Update sidecar -----------------------------------------------------
    sidecar_path = stack_path.with_suffix(".json")
    sidecar = {}
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text())
    features = sidecar.get("features", [])
    features = list(features) + deep_names
    sidecar["features"] = features
    sidecar["n_features"] = len(features)
    sidecar_path.write_text(json.dumps(sidecar, indent=2))

    print(f"Appended {n_components} deep bands to {stack_path} "
          f"(now {orig_count + n_components} bands)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Append frozen-ResNet-18 deep features to a feature stack"
    )
    parser.add_argument("scenes_dir", help="Folder of per-date scene subfolders")
    parser.add_argument("stack_path", help="Feature stack GeoTIFF to augment in place")
    parser.add_argument("--weights", default="sentinel2_rgb_moco",
                        help="TorchGeo ResNet18_Weights name (RGB variants use "
                             "our downloaded bands)")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional EuroSAT-fine-tuned Lightning checkpoint "
                             "(overrides --weights; head is dropped)")
    parser.add_argument("--components", type=int, default=16,
                        help="Number of PCA components / deep bands (default 16)")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--cut-layer", choices=list(CUT_STRIDES), default="layer2")
    parser.add_argument("--per-date", action="store_true",
                        help="Embed each date separately (default: median composite)")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    args = parser.parse_args()

    add_deep_features(
        args.scenes_dir, args.stack_path,
        weights=args.weights, checkpoint=args.checkpoint,
        n_components=args.components, tile_size=args.tile_size,
        cut_layer=args.cut_layer, per_date=args.per_date, device=args.device,
    )
