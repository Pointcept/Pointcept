from pathlib import Path
import os
import numpy as np
from collections.abc import Sequence
from plyfile import PlyData
from .builder import DATASETS
from .defaults import DefaultDataset


@DATASETS.register_module()
class SNCFMLSDataset(DefaultDataset):
    SPLITS = {
        "train": [
            "sncf_01.ply", "sncf_02.ply", "sncf_03.ply", "sncf_05.ply",
            "sncf_06.ply", "sncf_10.ply", "sncf_12.ply", "sncf_14.ply",
            "sncf_15.ply", "sncf_16.ply",
        ],
        "val": ["sncf_09.ply", "sncf_11.ply", "sncf_13.ply"],
        "test": ["sncf_04.ply", "sncf_07.ply", "sncf_08.ply"],
    }

    def __init__(
        self,
        data_root: str = "data/sncf_mls",
        split="train",
        ignore_index: int = -1,
        **kwargs,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)

        # Build data_list
        self.data_list = []
        splits = split if isinstance(split, (list, tuple)) else [split]
        for s in splits:
            split_dir = self.data_root / s
            for ply_file in self.SPLITS[s]:
                path = split_dir / ply_file
                self.data_list.append({
                    "ply_path": str(path),
                    "scene_id": os.path.splitext(ply_file)[0],
                })

        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        return self.data_list

    def get_data(self, idx):
        entry = self.data_list[idx % len(self.data_list)]
        ply_path = entry["ply_path"]
        plydata = PlyData.read(ply_path)
        v = plydata['vertex'].data
        coord = np.vstack((v['x'], v['y'], v['z'])).T.astype(np.float64)
        centroid = coord.mean(axis=0, keepdims=True)
        coord -= centroid
        coord = coord.astype(np.float32)
        labels = v['scalar_Classification'].astype(np.int32)
        if self.learning_map:
            labels = np.vectorize(self.learning_map.__getitem__)(
                labels).astype(np.int64)
        else:
            labels = np.full((labels.shape[0],),
                             self.ignore_index, dtype=np.int64)

        return dict(coord=coord, segment=labels, name=entry["scene_id"])

    @staticmethod
    def get_learning_map(ignore_index):
        # Default pass-through; override when needed
        return {}
