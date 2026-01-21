"""
Semantic KITTI dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from PIL import Image
import torch
import open3d as o3d
import random

from .builder import DATASETS
from .defaults import DefaultDataset, DefaultImagePointDataset

os.environ["OMP_NUM_THREADS"] = "1"


@DATASETS.register_module()
class SemanticKITTIDataset(DefaultDataset):
    def __init__(self, ignore_index=-1, **kwargs):
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError

        data_list = []
        for seq in seq_list:
            seq = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv


@DATASETS.register_module()
class SemanticKITTIColorNormalDataset(SemanticKITTIDataset):
    @staticmethod
    def estimate_normals(points, center=np.array([0, 0, 0])):
        normals = points - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        normal = self.estimate_normals(coord)
        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)
        data_dict = dict(
            coord=coord,
            color=np.zeros_like(coord),  # placeholder for color
            normal=np.zeros_like(coord),  # placeholder for normal
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict


@DATASETS.register_module()
class SemanticKITTIImagePointDataset(DefaultImagePointDataset):
    CAMERA_TYPES = [2, 3]

    def __init__(
        self,
        if_img=False,
        if_sweep=False,
        sweeps=3,
        sweep_gap=5,
        ignore_index=-1,
        **kwargs,
    ):
        self.if_img = if_img
        self.ignore_index = ignore_index
        self.learning_map = self.get_learning_map(ignore_index)
        self.learning_map_inv = self.get_learning_map_inv(ignore_index)
        if if_sweep:
            self.timestamp = tuple(range(0, sweeps * sweep_gap, sweep_gap))
        else:
            self.timestamp = (0,)
        super().__init__(ignore_index=ignore_index, **kwargs)

    @staticmethod
    def project_lidar_to_image_with_color(
        lidar_points,  # shape: (N, 3) or (N, 4)
        image,  # shape: (H, W, 3), uint8 RGB
        cam_intrinsic,  # shape: (3, 3)
        lidar_to_cam,  # shape: (4, 4)
        lidar_colors,
    ):
        """
        Projects LiDAR points to the image, fetches pixel color and pixel coordinates.
        Returns:
            filtered_points: (M, 3) - 3D points in camera frame that project onto the image.
            colors:          (M, 3) - RGB colors at projected 2D locations.
            uv_coords:       (M, 2) - Integer pixel coordinates (u, v) on the image.
            mask:            (N,)   - (optional) Boolean mask indicating which lidar points are used.
        """
        lidar_uv_coords = np.full(
            (lidar_points.shape[0], 2), -1, dtype=int
        )  # Default to (-1, -1)
        lidar_points_coord = lidar_points[:, :3]
        ones = np.ones((lidar_points_coord.shape[0], 1))
        lidar_hom = np.concatenate([lidar_points_coord, ones], axis=1)  # (N, 4)

        points_cam = (lidar_to_cam @ lidar_hom.T).T  # (N, 4)

        valid = points_cam[:, 2] > 0
        points_cam = points_cam[valid]
        points_cam_homogeneous = np.hstack(
            (points_cam, np.ones((points_cam.shape[0], 1)))
        )

        pts_2d = (cam_intrinsic @ points_cam_homogeneous.T).T  # (N, 4)
        pts_2d = pts_2d[:, :2] / pts_2d[:, 2:3]  # (N, 2) - pixel (u, v)

        H, W = image.shape[:2]
        u, v = pts_2d[:, 0], pts_2d[:, 1]
        inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u = u[inside].astype(int)
        v = v[inside].astype(int)

        mask = np.zeros(lidar_points.shape[0], dtype=bool)
        mask[np.where(valid)[0][inside]] = True

        lidar_colors[mask] = image[v, u, :]
        lidar_uv_coords[mask] = np.stack([u, v], axis=1)  # (M, 2)

        return lidar_colors, lidar_uv_coords, mask

    @staticmethod
    def parse_calib_file(filepath):
        calib_data = {}
        with open(filepath, "r") as f:
            file_content = f.read()

        for line in file_content.strip().split("\n"):
            if ":" not in line:
                continue
            key, value_str = line.split(":", 1)
            key = key.strip()
            values = np.fromstring(value_str, dtype=np.float64, sep=" ")
            if values.size == 12:
                matrix = values.reshape(3, 4)
            elif values.size == 9:
                matrix = values.reshape(3, 3)
            else:
                matrix = values
            calib_data[key] = matrix

        return calib_data

    @staticmethod
    def align_pose(coord, pose, target_pose):
        coord = np.hstack((coord, np.ones_like(coord[:, :1])))
        try:
            pose_align = np.matmul(np.linalg.inv(target_pose), pose)
        except:
            print(target_pose)
            exit()
        coord = (pose_align @ coord.T).T[:, :3]
        return coord

    @staticmethod
    def get_pose(poses_file_path, frame_index, Tr):
        Tr = np.vstack((Tr, np.array([0, 0, 0, 1])))
        Tr_inv = np.linalg.inv(Tr)
        with open(poses_file_path, "r") as f:
            lines = f.readlines()

        if not frame_index < len(lines):
            print(frame_index, len(lines))
        assert frame_index < len(lines)
        line = lines[frame_index].strip().split()
        values = [float(v) for v in line]
        pose_matrix_3x4 = np.array(values).reshape(3, 4)
        pose_matrix_4x4 = np.vstack((pose_matrix_3x4, np.array([0, 0, 0, 1])))
        pose_matrix_4x4 = np.matmul(Tr_inv, np.matmul(pose_matrix_4x4, Tr))
        return pose_matrix_4x4

    def get_data_list(self):
        split2seq = dict(
            train=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
            val=[8],
            test=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        )
        if isinstance(self.split, str):
            seq_list = split2seq[self.split]
        elif isinstance(self.split, list):
            seq_list = []
            for split in self.split:
                seq_list += split2seq[split]
        else:
            raise NotImplementedError
        seq_list = sorted(seq_list)

        data_list = []
        split_list = {"train": [], "val": [], "test": []}
        for seq in seq_list:
            seq2 = str(seq).zfill(2)
            seq_folder = os.path.join(self.data_root, "dataset", "sequences", seq2)
            seq_files = sorted(os.listdir(os.path.join(seq_folder, "velodyne")))
            data_list += [
                os.path.join(seq_folder, "velodyne", file) for file in seq_files
            ]
            if seq in split2seq["train"]:
                split_list["train"].extend(
                    [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
                )
            elif seq in split2seq["val"]:
                split_list["val"].extend(
                    [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
                )
            elif seq in split2seq["test"]:
                split_list["test"].extend(
                    [os.path.join(seq_folder, "velodyne", file) for file in seq_files]
                )
        _, self.sequence_offset, self.sequence_index = np.unique(
            [os.path.dirname(os.path.dirname(data)) for data in data_list],
            return_index=True,
            return_inverse=True,
        )
        self.sequence_offset = np.append(self.sequence_offset, len(data_list))
        return data_list, split_list

    def get_single_frame(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        with open(data_path, "rb") as b:
            scan = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        coord = scan[:, :3]
        strength = scan[:, -1].reshape([-1, 1])

        label_file = data_path.replace("velodyne", "labels").replace(".bin", ".label")
        if os.path.exists(label_file):
            with open(label_file, "rb") as a:
                segment = np.fromfile(a, dtype=np.int32).reshape(-1)
                segment = np.vectorize(self.learning_map.__getitem__)(
                    segment & 0xFFFF
                ).astype(np.int32)
        else:
            segment = np.zeros(scan.shape[0]).astype(np.int32)

        imgs = []
        cam_correspondences = []
        correspondence_start = 0
        color = np.zeros((coord.shape[0], 3), dtype=int)  # Default to black

        calibration_path = os.path.join(
            os.path.dirname(os.path.dirname(data_path)), "calib.txt"
        )
        calib_info = self.parse_calib_file(calibration_path)
        Tr = calib_info["Tr"]
        for id, cam_name in enumerate(self.CAMERA_TYPES):
            image_path = data_path.replace("velodyne", f"image_{cam_name}").replace(
                ".bin", ".png"
            )
            P = calib_info[f"P{cam_name}"]
            cam_image = Image.open(image_path)
            cam_image_np = np.array(cam_image)
            color, correspondence_info, _ = self.project_lidar_to_image_with_color(
                coord, cam_image_np, P, Tr, color
            )
            correspondence_point_id = (
                np.array(range(correspondence_info.shape[0])).reshape((-1, 1))
                + correspondence_start
            )
            correspondence_info = np.hstack(
                [correspondence_info, correspondence_point_id]
            )
            # if np.random.rand()<self.img_ratio:
            cam_correspondences.append(correspondence_info)
            imgs.append(cam_image)
        correspondence_start += coord.shape[0]
        cam_center = np.array([0, 0, 0])
        normal = self.get_normals(cam_center, coord)

        img_DLC = dict()
        if self.if_img:
            img_width, img_height = imgs[0].size
            div_w = img_width // self.patch_w
            div_h = img_height // self.patch_h
            div_min = max(min(div_w, div_h), 1)
            crop_img_width = div_min * self.patch_w
            crop_img_height = div_min * self.patch_h
            left = int((img_width - crop_img_width) / 2)
            top = int((img_height - crop_img_height) / 2)
            right = int((img_width + crop_img_width) / 2)
            bottom = int((img_height + crop_img_height) / 2)
            imgs = [img.crop((left, top, right, bottom)) for img in imgs]
            imgs = [self.transform_img(img) for img in imgs]
            if len(imgs) > 0:
                imgs_list = torch.stack(imgs)
                img_DLC["images"] = imgs_list.float()
            else:
                img_DLC["images"] = torch.empty(
                    (
                        0,
                        3,
                        self.patch_h * self.patch_size,
                        self.patch_w * self.patch_size,
                    )
                )
            img_DLC["img_num"] = np.array([img_DLC["images"].shape[0]], dtype=np.int32)

            correspondence_infos = np.ones(
                (coord.shape[0], len(cam_correspondences), 2), dtype=np.int32
            ) * (-1)
            for id, correspondence_info in enumerate(cam_correspondences):
                correspondence_info = self.resize_correspondence_info(
                    correspondence_info,
                    (self.patch_h * self.patch_size, self.patch_w * self.patch_size),
                    (img_height, img_width),
                    (left, top, right, bottom),
                    self.patch_size,
                )
                correspondence_infos[correspondence_info[:, -1], id, :] = (
                    correspondence_info[:, :-1]
                )
            img_DLC["correspondence"] = correspondence_infos

        color = color.astype(np.float32)
        data_dict = dict(
            coord=coord + np.array([0, 0, 1.73]),
            color=color,
            normal=normal,
            strength=strength,
            segment=segment,
            name=self.get_data_name(idx),
        )
        data_dict.update(img_DLC)
        data_dict["Tr"] = Tr
        return data_dict

    def get_data(self, idx):
        idx = idx % len(self.data_list)
        if self.timestamp == (0,):
            data_dict = self.get_single_frame(idx)
            return data_dict

        sequence_index = self.sequence_index[idx]
        lower, upper = self.sequence_offset[[sequence_index, sequence_index + 1]]
        assert lower <= idx and idx < upper
        imgs_idx = random.sample(self.timestamp, 1)[0]
        major_frame = self.get_single_frame(idx)
        poses_file_path = os.path.dirname(os.path.dirname(self.data_list[idx]))
        poses_file_path = os.path.join(poses_file_path, "poses.txt")
        Tr = major_frame.pop("Tr")
        target_pose = self.get_pose(poses_file_path, idx - lower, Tr)
        name = major_frame.pop("name")
        if self.if_img:
            if imgs_idx == 0:
                major_frame["correspondence"] = [major_frame["correspondence"]]
            else:
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
            Tr = refer_frame.pop("Tr")
            pose = self.get_pose(poses_file_path, idx - lower + timestamp, Tr)
            refer_frame["coord"] = self.align_pose(
                refer_frame["coord"], pose, target_pose
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

    @staticmethod
    def get_normals(cam_center, coords):
        Cs = np.repeat(cam_center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        # normals = estimate_normals(coords,center = cam_center)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product > 0
        normals[flip_mask] = -normals[flip_mask]

        # Normalize normals a nd m
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals

    def get_data_name(self, idx):
        file_path = self.data_list[idx % len(self.data_list)]
        dir_path, file_name = os.path.split(file_path)
        sequence_name = os.path.basename(os.path.dirname(dir_path))
        frame_name = os.path.splitext(file_name)[0]
        data_name = f"{sequence_name}_{frame_name}"
        return data_name

    @staticmethod
    def get_learning_map(ignore_index):
        learning_map = {
            0: ignore_index,  # "unlabeled"
            1: ignore_index,  # "outlier" mapped to "unlabeled" --------------------------mapped
            10: 0,  # "car"
            11: 1,  # "bicycle"
            13: 4,  # "bus" mapped to "other-vehicle" --------------------------mapped
            15: 2,  # "motorcycle"
            16: 4,  # "on-rails" mapped to "other-vehicle" ---------------------mapped
            18: 3,  # "truck"
            20: 4,  # "other-vehicle"
            30: 5,  # "person"
            31: 6,  # "bicyclist"
            32: 7,  # "motorcyclist"
            40: 8,  # "road"
            44: 9,  # "parking"
            48: 10,  # "sidewalk"
            49: 11,  # "other-ground"
            50: 12,  # "building"
            51: 13,  # "fence"
            52: ignore_index,  # "other-structure" mapped to "unlabeled" ------------------mapped
            60: 8,  # "lane-marking" to "road" ---------------------------------mapped
            70: 14,  # "vegetation"
            71: 15,  # "trunk"
            72: 16,  # "terrain"
            80: 17,  # "pole"
            81: 18,  # "traffic-sign"
            99: ignore_index,  # "other-object" to "unlabeled" ----------------------------mapped
            252: 0,  # "moving-car" to "car" ------------------------------------mapped
            253: 6,  # "moving-bicyclist" to "bicyclist" ------------------------mapped
            254: 5,  # "moving-person" to "person" ------------------------------mapped
            255: 7,  # "moving-motorcyclist" to "motorcyclist" ------------------mapped
            256: 4,  # "moving-on-rails" mapped to "other-vehicle" --------------mapped
            257: 4,  # "moving-bus" mapped to "other-vehicle" -------------------mapped
            258: 3,  # "moving-truck" to "truck" --------------------------------mapped
            259: 4,  # "moving-other"-vehicle to "other-vehicle" ----------------mapped
        }
        return learning_map

    @staticmethod
    def get_learning_map_inv(ignore_index):
        learning_map_inv = {
            ignore_index: ignore_index,  # "unlabeled"
            0: 10,  # "car"
            1: 11,  # "bicycle"
            2: 15,  # "motorcycle"
            3: 18,  # "truck"
            4: 20,  # "other-vehicle"
            5: 30,  # "person"
            6: 31,  # "bicyclist"
            7: 32,  # "motorcyclist"
            8: 40,  # "road"
            9: 44,  # "parking"
            10: 48,  # "sidewalk"
            11: 49,  # "other-ground"
            12: 50,  # "building"
            13: 51,  # "fence"
            14: 70,  # "vegetation"
            15: 71,  # "trunk"
            16: 72,  # "terrain"
            17: 80,  # "pole"
            18: 81,  # "traffic-sign"
        }
        return learning_map_inv
