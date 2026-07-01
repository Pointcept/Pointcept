"""
Dental (Bits2Bites) dataset — multi-task occlusal classification.

Author: Lorenzo Borghi (lorenzobrg@pm.me)
"""

import os
import copy
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from pointcept.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose


# Per-tooth landmark types → the 6-D one-hot channels of point_label_onehot.
POINT_CLASSES = ["Mesial", "Distal", "Cusp", "FacialPoint", "OuterPoint", "InnerPoint"]
POINT_CLASS_TO_IDX = {c: i for i, c in enumerate(POINT_CLASSES)}

# Raw CSV label vocabularies per task (kept in the original Italian to match the
# annotation files). Task order matches num_classes_list = [3, 3, 4, 3, 2].
RIGHT_LEFT_CLASSES = ["prima classe", "seconda classe", "terza classe"]
ANTERIOR_CLASSES = ["normale", "profondo", "aperto", "inverso"]
TRANSVERSE_CLASSES = ["normale", "scissor", "cross"]
MIDLINE_CLASSES = ["centrata", "deviata"]

STRING2IDX = {
    "right": {c: i for i, c in enumerate(RIGHT_LEFT_CLASSES)},
    "left": {c: i for i, c in enumerate(RIGHT_LEFT_CLASSES)},
    "anterior": {c: i for i, c in enumerate(ANTERIOR_CLASSES)},
    "transverse": {c: i for i, c in enumerate(TRANSVERSE_CLASSES)},
    "midline": {c: i for i, c in enumerate(MIDLINE_CLASSES)},
}


def _map_label(value: str, category: str) -> int:
    """Map a raw CSV label string to a class index; unknown/empty -> -1 (ignored)."""
    try:
        return STRING2IDX[category][value]
    except KeyError:
        return -1


@DATASETS.register_module()
class DentalDataset(Dataset):
    r"""Paired intra-oral scans with five occlusal labels.

    Folder layout::

        data_root/
            train/   dental_<id>.json ...
            val/     dental_<id>.json ...
            labels.csv               # id, right, left, anterior, transverse, midline

    Each JSON holds ``objects``; every object has a ``coord`` (xyz) and a
    ``class`` (either ``"Mesh"`` or one of POINT_CLASSES). Mesh points get an
    all-zero landmark one-hot; landmark points set their matching bit. Parsed
    samples are cached to ``data_root/dental_<split>.pth``.
    """

    def __init__(
        self,
        split: str = "train",
        data_root: str | os.PathLike = "data/dental",
        transform: Sequence[dict] | None = None,
        save_record: bool = True,
        test_mode: bool = False,  # accepted for config compatibility; single code path
        test_cfg=None,
        loop: int = 1,
        label_csv: str | os.PathLike | None = None,
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.json_dir = self.data_root / self.split
        if label_csv is None:
            label_csv = self.data_root / "labels.csv"
        self.labels_df = pd.read_csv(label_csv, dtype=str)
        self.test_mode = test_mode
        self.transform = transform if callable(transform) else Compose(transform)
        self.loop = loop if not test_mode else 1

        self.data_list = self._build_data_list()
        logger = get_root_logger()
        logger.info(
            f"Totally {len(self.data_list)} x {self.loop} samples in {self.split} set."
        )

        record_path = self.data_root / f"dental_{self.split}.pth"
        if record_path.is_file():
            logger.info(f"Loading record: {record_path.name} ...")
            self.data_cache = torch.load(record_path, weights_only=False)
        else:
            logger.info(f"Preparing record: {record_path.name} ...")
            self.data_cache = {}
            for idx, data_name in enumerate(self.data_list):
                logger.info(f"Parsing [{idx + 1}/{len(self.data_list)}] {data_name}")
                self.data_cache[data_name] = self._load_sample(idx)
            if save_record:
                torch.save(self.data_cache, record_path)

    def _build_data_list(self) -> list:
        assert self.json_dir.is_dir(), f"{self.json_dir} not found"
        return sorted(p.stem for p in self.json_dir.glob("dental_*.json"))

    def _load_sample(self, idx: int) -> dict:
        data_name = self.data_list[idx % len(self.data_list)]  # e.g. dental_0123
        patient_id = str(int(data_name.split("_")[1]))

        with open(self.json_dir / f"{data_name}.json", "r") as fp:
            obj = json.load(fp)

        coords = np.asarray([o["coord"] for o in obj["objects"]], dtype=np.float32)
        one_hot = np.zeros((len(obj["objects"]), len(POINT_CLASSES)), dtype=np.float32)
        for i, o in enumerate(obj["objects"]):
            if o["class"] != "Mesh":
                one_hot[i, POINT_CLASS_TO_IDX[o["class"]]] = 1.0

        row = self.labels_df[self.labels_df.iloc[:, 0] == patient_id]
        if row.empty:
            raise KeyError(f"Patient {patient_id} not found in labels.csv")
        row = row.iloc[0]

        return {
            "name": data_name,
            "coord": coords,  # (N, 3) float32
            "point_label_onehot": one_hot,  # (N, 6) float32
            # per-point keys subsampled together by GridSample/RandomDropout
            "index_valid_keys": ["coord", "point_label_onehot"],
            # five clinical targets, each (1,) int64; -1 marks an ignored label
            "label_0": np.array([_map_label(row[1], "right")], dtype=np.int64),
            "label_1": np.array([_map_label(row[2], "left")], dtype=np.int64),
            "label_2": np.array([_map_label(row[3], "anterior")], dtype=np.int64),
            "label_3": np.array([_map_label(row[4], "transverse")], dtype=np.int64),
            "label_4": np.array([_map_label(row[5], "midline")], dtype=np.int64),
        }

    def __len__(self) -> int:
        return len(self.data_list) * self.loop

    def get_data_name(self, idx: int) -> str:
        return self.data_list[idx % len(self.data_list)]

    def __getitem__(self, idx: int) -> dict:
        item = copy.deepcopy(self.data_cache[self.get_data_name(idx)])
        return self.transform(item)
