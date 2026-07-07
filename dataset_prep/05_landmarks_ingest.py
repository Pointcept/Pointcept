#!/usr/bin/env python3
"""Step 3c — ingest BracketPrediction's 3DTeethLand output (predictions.csv) into
the Bits2Bites landmark format, writing one landmarks/dental_<id:04d>.json per
patient (both jaws merged into one file).

predictions.csv columns: key, coord_x, coord_y, coord_z, class, score
    key    = "<id>_<arch>"
    class  in {Planar, Bracket, Incisal, Cusp, OuterPoint, Mesial, Distal,
               InnerPoint, FacialPoint}

Only the six classes DentalDataset knows (POINT_CLASSES = Mesial, Distal, Cusp,
FacialPoint, OuterPoint, InnerPoint) are kept; Planar/Bracket/Incisal are dropped.
Coordinates are already in the oriented frame (inference ran on oriented meshes).

Output schema per file (matches datasets/Bits2Bites/landmarks/dental_XXXX.json):
    {"version": "1.1", "description": "landmarks", "key": "dental_0201",
     "objects": [{"key": "...", "class": "Mesial", "coord": [x, y, z]}, ...]}

Usage:
    python 05_landmarks_ingest.py \
        --predictions <lm_out>/<exp>/predictions.csv \
        --landmarks-dir /work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented/landmarks
    # or point --output-folder at the run root to auto-pick the newest predictions.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

KEEP_CLASSES = {"Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--predictions", type=Path, help="Path to a predictions.csv")
    g.add_argument("--output-folder", type=Path, help="BracketPrediction --output-folder; newest predictions.csv is used")
    p.add_argument("--landmarks-dir", type=Path, required=True, help="Oriented dataset's landmarks/ dir to write into")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing dental_<id>.json files")
    return p.parse_args()


def find_predictions(output_folder: Path) -> Path:
    candidates = sorted(output_folder.rglob("predictions.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        print(f"error: no predictions.csv under {output_folder}", file=sys.stderr)
        sys.exit(1)
    return candidates[-1]


def main():
    args = parse_args()
    pred = args.predictions or find_predictions(args.output_folder)
    print(f"Reading {pred}")

    per_patient = {}  # id -> list of objects
    kept = dropped = 0
    with open(pred, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            if cls not in KEEP_CLASSES:
                dropped += 1
                continue
            pid = int(row["key"].split("_")[0])
            objs = per_patient.setdefault(pid, [])
            objs.append({
                "key": f"uuid_{len(objs)}",
                "class": cls,
                "coord": [float(row["coord_x"]), float(row["coord_y"]), float(row["coord_z"])],
            })
            kept += 1

    args.landmarks_dir.mkdir(parents=True, exist_ok=True)
    written = skipped = 0
    for pid, objects in sorted(per_patient.items()):
        out = args.landmarks_dir / f"dental_{pid:04d}.json"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        out.write_text(json.dumps({
            "version": "1.1",
            "description": "landmarks",
            "key": f"dental_{pid:04d}",
            "objects": objects,
        }))
        written += 1

    print(f"Kept {kept} landmark points ({dropped} non-tooth points dropped) "
          f"across {len(per_patient)} patients.")
    print(f"Wrote {written} dental_<id>.json ({skipped} skipped, already present) -> {args.landmarks_dir}")


if __name__ == "__main__":
    main()
