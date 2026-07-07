#!/usr/bin/env python3
"""Step 3a — stage the oriented scans into the input layout BracketPrediction's
3DTeethLand landmark predictor expects, for every patient that has no landmarks
yet (i.e. the newly-added B patients; A already has landmarks carried through
orientation in step 2).

BracketPrediction input layout (`main.py` / `_iter_patient_files`):
    <lm_in>/lower/<id>/<id>_lower.stl
    <lm_in>/upper/<id>/<id>_upper.stl
    <lm_in>/testing_lower.txt   # one "<id>_lower" per line
    <lm_in>/testing_upper.txt   # one "<id>_upper" per line

Meshes are symlinked by default (--copy to copy). Because inference runs on the
ORIENTED meshes, the predicted landmarks come back already in the oriented frame,
so they line up with the meshes we train on — no back-transform needed.

Usage:
    python 04_landmarks_prep.py \
        --dataset /work/grana_maxillo/lborghi/datasets/Bits2Bites_oriented \
        --lm-in   /work/grana_maxillo/lborghi/datasets/Bits2Bites_lm_in
"""
import argparse
import csv
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, required=True, help="Oriented dataset (03 output)")
    p.add_argument("--lm-in", type=Path, required=True, help="Directory to build the BracketPrediction input in")
    p.add_argument("--copy", action="store_true", help="Copy STLs instead of symlinking")
    p.add_argument("--all", action="store_true", help="Stage ALL patients, even those that already have landmarks")
    p.add_argument("--force", action="store_true", help="Restage a patient even if already staged")
    return p.parse_args()


def patient_ids(dataset: Path):
    with open(dataset / "Annotations.csv", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        return sorted(int(r[0]) for r in reader if r)


def main():
    args = parse_args()
    (args.lm_in / "lower").mkdir(parents=True, exist_ok=True)
    (args.lm_in / "upper").mkdir(parents=True, exist_ok=True)

    lower_list, upper_list, staged = [], [], 0
    for pid in patient_ids(args.dataset):
        if not args.all and (args.dataset / "landmarks" / f"dental_{pid:04d}.json").is_file():
            continue
        for arch in ("lower", "upper"):
            src = (args.dataset / str(pid) / f"{arch}.stl").resolve()
            if not src.is_file():
                print(f"warning: missing {src}, skipping {pid}_{arch}", file=sys.stderr)
                continue
            pdir = args.lm_in / arch / str(pid)
            pdir.mkdir(parents=True, exist_ok=True)
            dst = pdir / f"{pid}_{arch}.stl"
            if dst.exists() or dst.is_symlink():
                if not args.force:
                    (lower_list if arch == "lower" else upper_list).append(f"{pid}_{arch}")
                    continue
                dst.unlink()
            if args.copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
            (lower_list if arch == "lower" else upper_list).append(f"{pid}_{arch}")
        staged += 1

    (args.lm_in / "testing_lower.txt").write_text("\n".join(lower_list) + "\n")
    (args.lm_in / "testing_upper.txt").write_text("\n".join(upper_list) + "\n")

    print(f"Staged {staged} patients ({len(lower_list)} lower, {len(upper_list)} upper) -> {args.lm_in}")
    print(f"Sample lists: testing_lower.txt, testing_upper.txt")


if __name__ == "__main__":
    main()
