"""
Waymo dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
import glob
import random

from .builder import DATASETS
from .defaults import DefaultDataset, DefaultImagePointDataset


@DATASETS.register_module()
class WaymoDataset(DefaultDataset):
    def __init__(
        self,
        timestamp=(0,),
        reference_label=True,
        timing_embedding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert timestamp[0] == 0
        self.timestamp = timestamp
        self.reference_label = reference_label
        self.timing_embedding = timing_embedding
        self.data_list = sorted(self.data_list)
        _, self.sequence_offset, self.sequence_index = np.unique(
            [os.path.dirname(data) for data in self.data_list],
            return_index=True,
            return_inverse=True,
        )
        self.sequence_offset = np.append(self.sequence_offset, len(self.data_list))

    def get_data_list(self):
        if isinstance(self.split, str):
            self.split = [self.split]
        data_list = []
        for split in self.split:
            data_list += glob.glob(os.path.join(self.data_root, split, "*", "*"))
        return data_list

    @staticmethod
    def align_pose(coord, pose, target_pose):
        coord = np.hstack((coord, np.ones_like(coord[:, :1])))
        pose_align = np.matmul(np.linalg.inv(target_pose), pose)
        coord = (pose_align @ coord.T).T[:, :3]
        return coord

    def get_single_frame(self, idx):
        return super().get_data(idx)

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            return self.get_single_frame(idx)

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        major_frame = self.get_single_frame(idx)
        name = major_frame.pop("name")
        target_pose = major_frame.pop("pose")
        for key in major_frame.keys():
            major_frame[key] = [major_frame[key]]

        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
            refer_frame = self.get_single_frame(refer_idx)
            refer_frame.pop("name")
            pose = refer_frame.pop("pose")
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
            )
            if not self.reference_label:
                refer_frame["segment"] = (
                    np.ones_like(refer_frame["segment"]) * self.ignore_index
                )

            if self.timing_embedding:
                refer_frame["strength"] = np.hstack(
                    (
                        refer_frame["strength"],
                        np.ones_like(refer_frame["strength"]) * timestamp,
                    )
                )

            for key in major_frame.keys():
                major_frame[key].append(refer_frame[key])
        for key in major_frame.keys():
            major_frame[key] = np.concatenate(major_frame[key], axis=0)
        major_frame["name"] = name
        return major_frame

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        sequence_path, frame_name = os.path.split(file_path)
        sequence_name = os.path.basename(sequence_path)
        data_name = f"{sequence_name}_{frame_name}"
        return data_name


@DATASETS.register_module()
class WaymoColorNormalDataset(WaymoDataset):
    @staticmethod
    def estimate_normals(points, center=np.array([0, 0, 0])):
        normals = points - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals

    def get_single_frame(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        split = self.get_split_name(idx)
        data_dict = {}
        assets = os.listdir(data_path)
        for asset in assets:
            if not asset.endswith(".npy"):
                continue
            if asset[:-4] not in self.VALID_ASSETS:
                continue
            data_dict[asset[:-4]] = np.load(os.path.join(data_path, asset))
        data_dict["name"] = name
        data_dict["split"] = split
        data_dict["color"] = np.zeros_like(data_dict["coord"])  # placeholder for color
        data_dict["normal"] = np.zeros_like(data_dict["coord"])

        if "coord" in data_dict.keys():
            data_dict["coord"] = data_dict["coord"].astype(np.float32)

        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"].astype(np.float32)

        if "normal" in data_dict.keys():
            data_dict["normal"] = data_dict["normal"].astype(np.float32)

        if "segment" in data_dict.keys():
            data_dict["segment"] = data_dict["segment"].reshape([-1]).astype(np.int32)
        else:
            data_dict["segment"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )

        if "instance" in data_dict.keys():
            data_dict["instance"] = data_dict["instance"].reshape([-1]).astype(np.int32)
        else:
            data_dict["instance"] = (
                np.ones(data_dict["coord"].shape[0], dtype=np.int32) * -1
            )
        return data_dict

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            return self.get_single_frame(idx)

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        major_frame = self.get_single_frame(idx)
        name = major_frame.pop("name")
        target_pose = major_frame.pop("pose")
        for key in major_frame.keys():
            major_frame[key] = [major_frame[key]]

        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
            refer_frame = self.get_single_frame(refer_idx)
            refer_frame.pop("name")
            pose = refer_frame.pop("pose")
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
            )
            if not self.reference_label:
                refer_frame["segment"] = (
                    np.ones_like(refer_frame["segment"]) * self.ignore_index
                )

            if self.timing_embedding:
                refer_frame["strength"] = np.hstack(
                    (
                        refer_frame["strength"],
                        np.ones_like(refer_frame["strength"]) * timestamp,
                    )
                )

            for key in major_frame.keys():
                major_frame[key].append(refer_frame[key])
        for key in major_frame.keys():
            major_frame[key] = np.concatenate(major_frame[key], axis=0)
        major_frame["name"] = name
        return major_frame


@DATASETS.register_module()
class WaymoImagePointDataset(DefaultImagePointDataset):
    def __init__(
        self,
        if_img=False,
        if_sweep=False,
        reference_label=True,
        timing_embedding=False,
        sweeps=3,
        sweep_gap=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.if_img = if_img
        self.if_sweep = if_sweep
        if if_sweep:
            self.timestamp = tuple(range(0, sweeps * sweep_gap, sweep_gap))
        else:
            self.timestamp = (0,)
        self.reference_label = reference_label
        self.timing_embedding = timing_embedding
        self.sweeps = sweeps
        _, self.sequence_offset, self.sequence_index = np.unique(
            [data.split("_with_camera_labels_")[0] for data in self.data_list.keys()],
            return_index=True,
            return_inverse=True,
        )
        self.sequence_offset = np.append(self.sequence_offset, len(self.data_list))

    @staticmethod
    def align_pose(coord, pose, target_pose):
        coord = np.hstack((coord, np.ones_like(coord[:, :1])))
        pose_align = np.matmul(np.linalg.inv(target_pose), pose)
        coord = (pose_align @ coord.T).T[:, :3]
        return coord

    def get_single_frame(self, idx):
        return super().get_data(idx)

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            data_dict = self.get_single_frame(idx)
            if not self.if_img:
                data_dict.pop("img_num")
                data_dict.pop("images")
                data_dict.pop("correspondence")
            return data_dict

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        self.timestamp = [0] + [
            timestamp
            for timestamp in self.timestamp[1:]
            if timestamp + idx >= lower and upper > timestamp + idx
        ]
        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
        imgs_idx = random.sample(self.timestamp, 1)[0]
        major_frame = self.get_single_frame(idx)
        name = major_frame.pop("name")
        target_pose = major_frame.pop("pose")
        if not self.if_img:
            major_frame.pop("img_num")
            major_frame.pop("images")
            major_frame.pop("correspondence")
        elif imgs_idx == 0:
            major_frame["correspondence"] = [major_frame["correspondence"]]
        else:
            major_frame.pop("img_num")
            major_frame.pop("images")
            major_frame["correspondence"] = [
                -np.ones_like(major_frame["correspondence"])
            ]
        for key in major_frame.keys():
            if key in self.PC_VALID_ASSETS:
                major_frame[key] = [major_frame[key]]

        for timestamp in self.timestamp[1:]:
            refer_idx = timestamp + idx
            if refer_idx < lower or upper <= refer_idx:
                continue
            refer_frame = self.get_single_frame(refer_idx)
            refer_frame.pop("name")
            pose = refer_frame.pop("pose")
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
            )
            if not self.reference_label:
                refer_frame["segment"] = (
                    np.ones_like(refer_frame["segment"]) * self.ignore_index
                )

            if self.timing_embedding:
                refer_frame["strength"] = np.hstack(
                    (
                        refer_frame["strength"],
                        np.ones_like(refer_frame["strength"]) * timestamp,
                    )
                )

            for key in major_frame.keys():
                if key in self.PC_VALID_ASSETS:
                    major_frame[key].append(refer_frame[key])

            if self.if_img:
                if imgs_idx == timestamp:
                    major_frame["img_num"] = refer_frame["img_num"]
                    major_frame["images"] = refer_frame["images"]
                    major_frame["correspondence"].append(refer_frame["correspondence"])
                else:
                    major_frame["correspondence"].append(
                        -np.ones_like(refer_frame["correspondence"])
                    )
        frame_pcd_offset = [
            frame_coord.shape[0] for frame_coord in major_frame["coord"]
        ]
        frame_pcd_offset = np.cumsum(frame_pcd_offset)

        for key in major_frame.keys():
            if key in self.PC_VALID_ASSETS + ["correspondence"]:
                major_frame[key] = np.concatenate(major_frame[key], axis=0)
        major_frame["name"] = name
        major_frame["frame_pcd_offset"] = frame_pcd_offset
        return major_frame
