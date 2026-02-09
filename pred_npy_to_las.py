import os
import numpy as np
import laspy
from pathlib import Path


def save_pred_to_las(tile_dir, pred_path, out_path):
    # Load data from tile directory
    coords = np.load(tile_dir / "coord.npy")   # (N,3)
    labels_gt = np.load(tile_dir / "segment.npy")  # (N,)
    intensity = np.load(tile_dir / "strength.npy")  # (N,)

    # Load predictions
    preds = np.load(pred_path)  # (N,)
    assert preds.shape[0] == coords.shape[0], \
        f"Mismatch: {preds.shape[0]} preds vs {coords.shape[0]} points"

    # Create LAS header
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    # Coordinates
    las.x = coords[:, 0]
    las.y = coords[:, 1]
    las.z = coords[:, 2]

    # Intensity
    las.intensity = intensity.astype(np.uint16).reshape(-1)  # LAS intensity is uint16

    # Predicted labels as classification
    las.classification = preds.astype(np.uint8).reshape(-1)

    # Ground truth labels in user_data
    las.user_data = labels_gt.astype(np.uint8).reshape(-1)

    # Write LAS
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    print(f"[✓] Saved LAS: {out_path}")


if __name__ == "__main__":
    # Paths
    tile_root = Path("data/sncf_mls/test")
    result_dir = Path("exp/sncf_mls/semseg-pt-v3m1-0-base/result")
    output_dir = Path("exp/sncf_mls/semseg-pt-v3m1-0-base/result_las")

    # Loop over tiles
    for tile_dir in tile_root.glob("sncf_*_tile_*"):
        stem = tile_dir.name
        pred_file = result_dir / f"{stem}_pred.npy"
        if pred_file.exists():
            out_file = output_dir / f"{stem}_pred.las"
            save_pred_to_las(tile_dir, pred_file, out_file)
