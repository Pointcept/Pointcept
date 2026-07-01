"""Strip `"Mesh"` points from an already-built data_root, keeping only the
6 per-tooth landmark classes -- the "landmark-only variant" mentioned in
BITS2BITES.md's Input section. `labels.csv` and the fold split are unchanged
(same configs apply, just far fewer points per sample -> much faster to train).
"""

import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a landmark-only data_root from a mesh+landmarks one")
    parser.add_argument("--input-dir", default="data/dental_landmarks_mesh")
    parser.add_argument("--output-dir", default="data/dental_landmarks_only")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(input_dir / "labels.csv", output_dir / "labels.csv")

    for fold_dir in sorted(input_dir.glob("fold_*")):
        out_fold_dir = output_dir / fold_dir.name
        out_fold_dir.mkdir(exist_ok=True)
        for json_path in sorted(fold_dir.glob("dental_*.json")):
            with open(json_path) as f:
                data = json.load(f)
            data["objects"] = [o for o in data["objects"] if o["class"] != "Mesh"]
            data["description"] = "landmarks"
            with open(out_fold_dir / json_path.name, "w") as f:
                json.dump(data, f)
        print(f"{fold_dir.name}: {len(list(out_fold_dir.glob('dental_*.json')))} samples")

    print(f"Done. Landmark-only data ready at {output_dir}")


if __name__ == "__main__":
    main()
