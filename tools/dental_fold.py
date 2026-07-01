"""
Assemble the train/ and val/ splits for a chosen cross-validation fold.

The Bits2Bites dataset ships as ``data_root/fold_1 ... fold_5`` (each holding
``dental_*.json``) plus ``data_root/labels.csv``. This script copies the chosen
fold into ``data_root/val/`` (the held-out fold, used for both validation and
final testing) and the remaining folds into ``data_root/train/`` — the layout
``DentalDataset`` expects. Run it once per fold before training/testing; stale
per-split caches (``dental_*.pth``) are removed so the dataset re-parses.
"""

import argparse
import shutil
from pathlib import Path


def prepare_folds(fold_val: int, data_root, num_folds: int = 5) -> None:
    assert 1 <= fold_val <= num_folds, f"fold_val must be in 1..{num_folds}"
    root = Path(data_root)
    train_dir, val_dir = root / "train", root / "val"

    for d in (train_dir, val_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    for i in range(1, num_folds + 1):
        fold_dir = root / f"fold_{i}"
        assert fold_dir.is_dir(), f"{fold_dir} not found"
        target = val_dir if i == fold_val else train_dir
        for f in fold_dir.glob("*.json"):
            shutil.copy(f, target / f.name)

    # drop stale caches so DentalDataset re-parses the freshly assembled split
    for cache in root.glob("dental_*.pth"):
        cache.unlink()

    print(f"Fold {fold_val} -> val, remaining -> train (data_root={data_root}).")


def main():
    parser = argparse.ArgumentParser(description="Prepare a CV fold for DentalDataset.")
    parser.add_argument("--fold-val", type=int, required=True, help="held-out fold (1..5)")
    parser.add_argument("--data-root", type=str, required=True, help="dataset root dir")
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()
    prepare_folds(args.fold_val, args.data_root, args.num_folds)


if __name__ == "__main__":
    main()
