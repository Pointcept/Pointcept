import os
import numpy as np
import laspy
from pathlib import Path

def save_pred_to_las(npz_path, pred_path, out_path):
    # Load original npz (points + GT labels)
    npz = np.load(npz_path)
    points = npz["points"]  # (N, 3)
    labels_gt = npz["labels"]  # (N,)
    
    # Load predictions
    preds = np.load(pred_path)  # (N,)
    assert preds.shape[0] == points.shape[0], \
        f"Mismatch: {preds.shape[0]} preds vs {points.shape[0]} points"

    # Create LAS header
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)

    # Coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Store predicted labels as "classification"
    las.classification = preds.astype(np.uint8)

    # Optionally: store GT labels in "user_data"
    las.user_data = labels_gt.astype(np.uint8).reshape(-1)

    # Write file
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(out_path))
    print(f"[✓] Saved LAS: {out_path}")


if __name__ == "__main__":
    # Example usage
    tile_dir = "data/sncf_mls/pointcept_dataset_tiles/test/sncf_08"
    result_dir = "exp/sncf_mls/semseg-spunet-v1m1-0-base/result"
    output_dir = "exp/sncf_mls/semseg-spunet-v1m1-0-base/result_las"

    for npz_file in Path(tile_dir).glob("*.npz"):
        stem = npz_file.stem
        pred_file = Path(result_dir) / f"{stem}.npz_pred.npy"
        if pred_file.exists():
            out_file = Path(output_dir) / f"{stem}_pred.las"
            save_pred_to_las(npz_file, pred_file, out_file)
