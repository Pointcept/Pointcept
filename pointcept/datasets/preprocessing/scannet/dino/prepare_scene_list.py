import os
import argparse
from pathlib import Path

import numpy as np

if __name__ == "__main__":
    num_train_list = 12
    num_val_list = 3
    meta_root = Path(os.path.dirname(__file__)).parent / "meta_data"

    # Load train/val splits
    train_scenes = np.loadtxt(meta_root / "scannetv2_train.txt", dtype=str)
    val_scenes = np.loadtxt(meta_root / "scannetv2_val.txt", dtype=str)

    for i in range(num_train_list):
        np.savetxt(
            meta_root / f"scannetv2_train_{i}.txt",
            train_scenes[i::num_train_list],
            fmt="%s",
        )
    for i in range(num_val_list):
        np.savetxt(
            meta_root / f"scannetv2_val_{i}.txt",
            val_scenes[i::num_val_list],
            fmt="%s",
        )
