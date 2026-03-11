"""
nuScenes Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Zheng Zhang
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections.abc import Sequence
from typing import List
import pickle
from PIL import Image
import open3d as o3d
import torch
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2

from .builder import DATASETS
from .defaults import DefaultDataset, DefaultImagePointDataset
from nuscenes.utils.data_classes import RadarPointCloud
os.environ["OMP_NUM_THREADS"] = "1"

name_to_label = {
    "car":1,
    "bicycle":2,
    "pedestrian":3,
}

def label_points_from_boxes(points: np.ndarray, boxes: List, box_rotations: List, box_labels: List, default_label: int = 0):
    """
    Efficiently assigns labels to points using a spatial pre-filter.
    
    :param points: <np.float: N, 3>. Point cloud coordinates.
    :param boxes: List of Box objects.
    :param default_label: Label for points outside any box.
    :return: <np.int: N>. Label array.
    """
    num_points = points.shape[0]
    point_labels = np.full((num_points,), default_label, dtype=np.int64)
    
    # Pre-calculate squared distances to avoid expensive sqrt() calls
    # for points that are clearly far away.
    for box, box_rotation, box_label in zip(boxes, box_rotations, box_labels):
        # 1. Quick Spatial Culling (Bounding Sphere check)
        # The circumradius is the distance from center to any corner: 
        # R = sqrt((w/2)^2 + (l/2)^2 + (h/2)^2)
        l,w, h = box[3:6]
        max_radius_sq = (w/2)**2 + (l/2)**2 + (h/2)**2
        
        # Translate points relative to center
        translated_points = points - box[0:3]
        
        # Squared Euclidean distance from center for all points
        dist_sq = np.sum(translated_points**2, axis=1)
        
        # Create a mask for points that are "near enough" to potentially be inside
        near_mask = dist_sq <= max_radius_sq
        
        if not np.any(near_mask):
            continue
            
        # 2. Precise Orientation Check
        # Only perform rotation on the subset of points within the sphere
        points_to_check = translated_points[near_mask]
        
        # Rotate points into box local frame
        # We use box.rotation_matrix.T because we are rotating the coordinate system
        local_points = np.dot(points_to_check, box_rotation)
        
        # Local Box dimensions check:
        # Based on the Box class: wlh = [width, length, height]
        # Conventionally: x=length, y=width, z=height
        in_box_mask = (
            (np.abs(local_points[:, 0]) <= l / 2) & 
            (np.abs(local_points[:, 1]) <= w / 2) & 
            (np.abs(local_points[:, 2]) <= h / 2)
        )
        
        # 3. Apply Labels
        # We need to map the 'in_box_mask' back to the original point indices
        # We use np.where to find the indices of the 'near_mask' and then slice them
        near_indices = np.where(near_mask)[0]
        final_indices = near_indices[in_box_mask]
        
        point_labels[final_indices] = name_to_label[box_label]
        
    return point_labels

@DATASETS.register_module()
class PercivDataset(DefaultDataset):
    def __init__(self, radar_mapping, norm_params, sweeps=5, max_sweeps=10, ignore_index=-1, **kwargs):
        self.sweeps = sweeps
        self.max_sweeps = max_sweeps
        self.radar_mapping = radar_mapping  
        self.norm_params = norm_params
        self.doppler_mean = self.norm_params["doppler"][0]
        self.doppler_std = self.norm_params["doppler"][1]
        self.rcs_mean = self.norm_params["rcs"][0]
        self.rcs_std = self.norm_params["rcs"][1]
        super().__init__(ignore_index=ignore_index, **kwargs)

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.max_sweeps}sweeps_train.pkl"
            )
        elif split == "val":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.max_sweeps}sweeps_val.pkl"
            )
        elif split == "test":
            return os.path.join(
                self.data_root, "info", f"nuscenes_infos_{self.max_sweeps}sweeps_test.pkl"
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
        elif isinstance(self.split, Sequence):
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path in info_paths:
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
        return data_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, data["radar_path"])
        points = RadarPointCloud.from_file(str(lidar_path)).points.T
        
        coord = points[:, :3]
        
        # 2. Extract raw features
        raw_doppler = points[:, self.radar_mapping["doppler"]].reshape([-1, 1])
        raw_rcs = points[:, self.radar_mapping["rcs"]].reshape([-1, 1])
        
        # 3. Apply Z-score normalization
        # A small epsilon (1e-6) is added to the denominator to prevent division by zero
        doppler = (raw_doppler - self.doppler_mean) / (self.doppler_std + 1e-6)
        rcs = (raw_rcs - self.rcs_mean) / (self.rcs_std + 1e-6)
        
        image_path = os.path.join(self.data_root, data.get("cam_front_path", ""))
        
        data_dict = dict(
            coord=coord,
            doppler=doppler,
            rcs=rcs,
            image_path=image_path,
            name=self.get_data_name(idx),
        )
        return data_dict

    def get_data_name(self, idx):
        # return data name for lidar seg, optimize the code when need to support detection
        return self.data_list[idx % len(self.data_list)]["radar_token"]


@DATASETS.register_module()
class PercivMultisweepDataset(PercivDataset):
    def __init__(self, radar_mapping, norm_params, sweeps=5, max_sweeps=10, ignore_index=-1, add_time_diff_dim='index', cam_chan='OAK_CAM_FRONT', debug=False, **kwargs):
        super().__init__(radar_mapping=radar_mapping, norm_params=norm_params, ignore_index=ignore_index, **kwargs)        
        self.add_time_diff_dim = add_time_diff_dim
        self.sweeps = sweeps
        self.max_sweeps = max_sweeps
        self.cam_chan = cam_chan
        self.debug = debug

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        sweeps = data['sweeps']
        radar_path = os.path.join(self.data_root, data["radar_path"])
        points = RadarPointCloud.from_file(str(radar_path)).points.T
        if self.add_time_diff_dim:
            time_diff = np.ones((points.shape[0], 1)) * 0
            points = np.concatenate([points, time_diff], axis=1)
        points_sweep_list = [points]
        for idx, sweep in enumerate(sweeps[:(self.sweeps-1)]): #iterate over sweeps
            sweep_path = os.path.join(self.data_root, sweep['radar_path'])
            points_sweep = RadarPointCloud.from_file(str(sweep_path)).points.T
            trans_matrix = sweep['transform_matrix'] if sweep['transform_matrix'] is not None else np.eye(4)
            points_homogeneous = np.hstack((points_sweep[:,:3], np.ones((points_sweep.shape[0], 1))))
            transformed_points_homogeneous = trans_matrix.dot(points_homogeneous.T).T

            points_sweep[:,:3] = transformed_points_homogeneous[:, :3]
        
            if self.add_time_diff_dim:
                # timestamp = sweep['timestamp'] * 1e-6
                # time_diff = (ts - timestamp) if self.add_time_diff_dim=='with_seconds' else idx
                time_diff = idx+1
                time_diff = np.ones((points_sweep.shape[0], 1)) * time_diff

                points_sweep_ = np.concatenate(
                    [points_sweep, time_diff], axis=1)

            points_sweep_list.append(points_sweep_)
        
        points = np.concatenate(points_sweep_list, axis=0)
                
        # 2. Extract raw features
        coord = points[:, :3]
        raw_doppler = points[:, self.radar_mapping["doppler"]].reshape([-1, 1])
        raw_rcs = points[:, self.radar_mapping["rcs"]].reshape([-1, 1])
        time = points[:, -1].reshape([-1, 1])

        # 3. Apply Z-score normalization
        # A small epsilon (1e-6) is added to the denominator to prevent division by zero
        doppler = (raw_doppler - self.doppler_mean) / (self.doppler_std + 1e-6)
        rcs = (raw_rcs - self.rcs_mean) / (self.rcs_std + 1e-6)
        
        segment = label_points_from_boxes(coord, data["gt_boxes"], data["gt_boxes_rotation_matrices"], data["gt_names"], default_label=0)

        # some camera infos for debug and visualization
        cam_info = {}
        image_path = os.path.join(self.data_root, data.get("cam_front_path", ""))
        lidar_to_ref = data["lidar_to_ref"] if 'lidar_to_ref' in data.keys() else data["ref_to_lidar"] #legacy wrong naming
        ref_to_lidar = np.linalg.inv(lidar_to_ref)
        cam_info = data["cams"][self.cam_chan]
        cam_intrinsic = cam_info["camera_intrinsics"]
        cam2lidar = np.eye(4)
        cam2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
        cam2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
        lidar2cam = np.linalg.inv(cam2lidar)
        ref2cam = lidar2cam @ ref_to_lidar
        cam_info = dict(            
            image_path=image_path,
            ref2cam=ref2cam,
            cam_intrinsic=cam_intrinsic,)
        data_dict = dict(
            coord=coord,
            color=np.zeros_like(coord),  # placeholder for color
            normal=np.zeros_like(coord),  # placeholder for normal
            doppler=doppler,
            rcs=rcs,
            time =time,
            segment=segment,
            cam_info=cam_info,
            name=self.get_data_name(idx),
        )

        if self.debug:
            self.plot_debug(data_dict, data["gt_boxes"])
        return data_dict
    
    def plot_debug(self, data_dict, gt_boxes,save_dir="./debug_vis"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Extract Data
        coord_radar = data_dict["coord"]  # [N, 3]
        segment = data_dict["segment"]    # [N] Class labels
        
        # Extract box information from the data_dict
        # (Assuming these were passed in the data_dict during the return in your previous code)
        # Load image
        img_path = data_dict["cam_info"]["image_path"]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor='white')
        
        # --- 1. Plot Camera Image ---
        axes[0].imshow(img)
        axes[0].set_title(f"Camera View: {data_dict['name']}", fontsize=15)
        axes[0].axis('off')
        
        # --- 2. Plot Radar BEV ---
        # Define our class mapping
        # 0: Background, 1: Car (Blue), 2: Cyclist (Red), 3: Pedestrian (Green)
        class_config = {
            1: {'color': 'blue',  'label': 'Car',        'size': 25},
            2: {'color': 'red',   'label': 'Cyclist',    'size': 25},
            3: {'color': 'green', 'label': 'Pedestrian', 'size': 25},
        }

        # 1. Plot Unlabelled / Background points (Class 0)
        bg_mask = (segment == 0)
        axes[1].scatter(
            coord_radar[bg_mask, 0], 
            coord_radar[bg_mask, 1], 
            c='gray', 
            s=5,           # Smaller size
            alpha=0.4,     # Slightly transparent
            label='Background',
            edgecolors='none',
            zorder=2
        )

        # 2. Plot specific classes with distinct colors
        for class_id, config in class_config.items():
            class_mask = (segment == class_id)
            if np.any(class_mask):
                axes[1].scatter(
                    coord_radar[class_mask, 0], 
                    coord_radar[class_mask, 1], 
                    c=config['color'], 
                    s=config['size'],
                    label=config['label'],
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=3      # Plot on top of background
                )

        # Add a legend instead of a colorbar for better clarity
        axes[1].legend(loc='upper right', scatterpoints=1, fontsize='small', framealpha=0.8)

        # --- 3. Plot Bounding Boxes ---
        # If gt_boxes is [x, y, z, l, w, h, yaw]
        for box in gt_boxes:
            cx, cy, cz, l, w, h, yaw = box[:7]
            
            # Calculate the four corners of the box in BEV (X-Y plane)
            # 1. Box corners in local frame (centered at 0,0)
            # x points forward (length), y points left (width)
            local_corners = np.array([
                [ l/2,  w/2],  # Front Left
                [ l/2, -w/2],  # Front Right
                [-l/2, -w/2],  # Rear Right
                [-l/2,  w/2]   # Rear Left
            ])
            
            # 2. Rotate corners by yaw
            rot_mat = np.array([
                [np.cos(yaw), -np.sin(yaw)],
                [np.sin(yaw),  np.cos(yaw)]
            ])
            rotated_corners = (rot_mat @ local_corners.T).T
            
            # 3. Translate to world/lidar center
            final_corners = rotated_corners + np.array([cx, cy])
            
            # Create Polygon patch
            rect = patches.Polygon(
                final_corners, 
                closed=True, 
                linewidth=2, 
                edgecolor='red', 
                facecolor='none', 
                alpha=0.8,
                zorder=4
            )
            axes[1].add_patch(rect)
            
            # Optional: Add an arrow pointing in the 'forward' direction (heading)
            axes[1].arrow(cx, cy, np.cos(yaw) * (l/2), np.sin(yaw) * (l/2), 
                        color='red', width=0.1, head_width=0.5)

        # --- Formatting ---
        axes[1].set_aspect('equal', adjustable='box')
        limit = 60 # Slightly larger to see boxes at the edge
        axes[1].set_xlim(-limit, limit)
        axes[1].set_ylim(-limit, limit)
        axes[1].grid(True, linestyle='--', alpha=0.5)
        axes[1].set_title("Radar BEV (Colored by Class)", fontsize=15)
        axes[1].set_xlabel("X (Forward) [m]")
        axes[1].set_ylabel("Y (Left) [m]")
        
        save_path = os.path.join(save_dir, f"{data_dict['name']}_radar_vis.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()


@DATASETS.register_module()
class PercivColorNormalDataset(PercivDataset):
    @staticmethod
    def estimate_normals(points, center=np.array([0, 0, 0])):
        normals = points - center[None, :]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / norms
        return normals

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, "raw", data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )
        coord = points[:, :3]
        normal = self.estimate_normals(coord)
        if "gt_segment_path" in data.keys():
            gt_segment_path = os.path.join(
                self.data_root, "raw", data["gt_segment_path"]
            )
            segment = np.fromfile(
                str(gt_segment_path), dtype=np.uint8, count=-1
            ).reshape([-1])
            segment = np.vectorize(self.learning_map.__getitem__)(segment).astype(
                np.int64
            )
        else:
            segment = np.ones((points.shape[0],), dtype=np.int64) * self.ignore_index
        data_dict = dict(
            coord=coord,
            color=np.zeros_like(coord),  # placeholder for color
            normal=np.zeros_like(coord),  # placeholder for normal
            segment=segment,
            name=self.get_data_name(idx),
        )
        return data_dict


@DATASETS.register_module()
class PercivImagePointDataset(DefaultImagePointDataset):
    def __init__(
        self,
        radar_mapping,
        norm_params,
        if_img=False,
        if_sweep=False,
        sweeps_max=10,
        sweeps=10,
        sweep_gap=1,
        ignore_index=-1,
        img_num=4,
        camera_types=[
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_BACK_RIGHT",
    ],
        debug = False,
        **kwargs,
    ):
        self.sweeps = sweeps
        self.sweep_gap = sweep_gap
        self.sweeps_max = sweeps_max
        self.if_sweep = if_sweep
        self.img_ratio = img_num / (6 * sweeps)
        self.radar_mapping = radar_mapping  
        self.norm_params = norm_params
        self.doppler_mean = self.norm_params["doppler"][0]
        self.doppler_std = self.norm_params["doppler"][1]
        self.rcs_mean = self.norm_params["rcs"][0]
        self.rcs_std = self.norm_params["rcs"][1]
        self.camera_types = camera_types
        self.debug = debug
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

        pts_2d = (cam_intrinsic @ points_cam[:, :3].T).T  # (N, 3)
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

    def get_info_path(self, split):
        assert split in ["train", "val", "test"]
        if split == "train":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_train.pkl",
            )
        elif split == "val":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_val.pkl",
            )
        elif split == "test":
            return os.path.join(
                self.data_root,
                "info",
                f"nuscenes_infos_{self.sweeps_max}sweeps_test.pkl",
            )
        else:
            raise NotImplementedError

    def get_data_list(self):
        split_list = {}
        if isinstance(self.split, str):
            info_paths = [self.get_info_path(self.split)]
            split = [self.split]
        elif isinstance(self.split, Sequence):
            split = self.split
            info_paths = [self.get_info_path(s) for s in self.split]
        else:
            raise NotImplementedError
        data_list = []
        for info_path, split_i in zip(info_paths, split):
            with open(info_path, "rb") as f:
                info = pickle.load(f)
                data_list.extend(info)
                split_list[split_i] = list([i["token"] for i in info])
        return data_list, split_list

    def get_data(self, idx):
        data = self.data_list[idx % len(self.data_list)]
        lidar_path = os.path.join(self.data_root, data["lidar_path"])
        points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape(
            [-1, 5]
        )

        imgs = []
        cam_coords = []
        cam_normals = []
        cam_colors = []
        cam_strengths = []
        cam_correspondences = []
        correspondence_start = 0
        frame_pcd_offset = []
        lidar_colors = np.zeros((points.shape[0], 3), dtype=int)  # Default to black
        for id, cam_name in enumerate(self.camera_types):
            cam_info = data["cams"][cam_name]
            cam_intrinsic = cam_info["camera_intrinsics"]
            cam_image = Image.open(
                os.path.join(self.data_root, data["cams"][cam_name]["data_path"])
            )
            cam_image_np = np.array(cam_image)
            sensor2lidar = np.eye(4)
            sensor2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
            sensor2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
            lidar2sensor = np.linalg.inv(sensor2lidar)
            lidar_colors, correspondence_info, _ = (
                self.project_lidar_to_image_with_color(
                    points, cam_image_np, cam_intrinsic, lidar2sensor, lidar_colors
                )
            )
            correspondence_point_id = (
                np.array(range(correspondence_info.shape[0])).reshape((-1, 1))
                + correspondence_start
            )
            correspondence_info = np.hstack(
                [correspondence_info, correspondence_point_id]
            )
            if np.random.rand() < self.img_ratio:
                cam_correspondences.append(correspondence_info)
                imgs.append(cam_image)
        correspondence_start += points.shape[0]
        cam_coord = points[:, :3]
        cam_center = np.array([0, 0, 0])
        cam_normal = self.get_normals(cam_center, cam_coord)
        cam_normals.append(cam_normal)
        cam_strength = points[:, 3].reshape([-1, 1]) / 255
        cam_coords.append(cam_coord)
        cam_colors.append(lidar_colors)
        cam_strengths.append(cam_strength)

        if self.if_sweep:
            frame_pcd_offset.append(points.shape[0])
            for id, sweep in enumerate(
                data["sweeps"][: (self.sweep_gap * self.sweeps) : self.sweep_gap]
            ):
                lidar_path = os.path.join(self.data_root, sweep["lidar_path"])
                points = np.fromfile(
                    str(lidar_path), dtype=np.float32, count=-1
                ).reshape([-1, 5])
                lidar_colors = np.zeros(
                    (points.shape[0], 3), dtype=int
                )  # Default to black
                cam_lidar_tm = (
                    sweep["transform_matrix"]
                    if sweep["transform_matrix"] is not None
                    else np.eye(4)
                )
                for id, cam_name in enumerate(self.camera_types):
                    cam_info = sweep["cams"][cam_name]
                    # cam_image_np = np.array(imgs[id])
                    cam_intrinsic = cam_info["camera_intrinsics"]
                    cam_image = Image.open(
                        os.path.join(self.data_root, cam_info["data_path"])
                    )
                    cam_image_np = np.array(cam_image)
                    sensor2lidar = np.eye(4)
                    sensor2lidar[:3, :3] = cam_info["sensor2lidar_rotation"]
                    sensor2lidar[:3, 3] = cam_info["sensor2lidar_translation"]
                    # sensor2lidar = cam_lidar_tm @ sensor2lidar
                    lidar2sensor = np.linalg.inv(sensor2lidar)
                    lidar_colors, correspondence_info, _ = (
                        self.project_lidar_to_image_with_color(
                            points,
                            cam_image_np,
                            cam_intrinsic,
                            lidar2sensor,
                            lidar_colors,
                        )
                    )
                    correspondence_point_id = (
                        np.array(range(correspondence_info.shape[0])).reshape((-1, 1))
                        + correspondence_start
                    )
                    correspondence_info = np.hstack(
                        [correspondence_info, correspondence_point_id]
                    )
                    if np.random.rand() < self.img_ratio:
                        cam_correspondences.append(correspondence_info)
                        imgs.append(cam_image)
                correspondence_start += correspondence_info.shape[0]
                frame_pcd_offset.append(correspondence_start)
                cam_coord = points[:, :3]
                cam_center = np.array([0, 0, 0])
                cam_normal = self.get_normals(cam_center, cam_coord)
                cam_normals.append(cam_normal)
                ones = np.ones((points.shape[0], 1))
                cam_coord_hom = np.concatenate([cam_coord, ones], axis=1)  # (N, 4)
                cam_coord = cam_coord_hom @ cam_lidar_tm.T
                cam_coord = cam_coord[:, :3]
                cam_strength = points[:, 3].reshape([-1, 1]) / 255
                cam_coords.append(cam_coord)
                cam_colors.append(lidar_colors)
                cam_strengths.append(cam_strength)

        coord = np.vstack(cam_coords)
        color = np.vstack(cam_colors)
        normal = np.vstack(cam_normals)
        strength = np.vstack(cam_strengths)

        # 1. Create a mask where any color channel (R, G, or B) is NOT zero
        # np.any(..., axis=1) checks if any value in the row is True
        mask = np.any(color != 0, axis=1)

        # 2. Apply the mask to all synchronized arrays
        coord = coord[mask]
        color = color[mask]
        normal = normal[mask]
        strength = strength[mask]
        frame_pcd_offset = np.array(frame_pcd_offset)

        # default assumption is that ref is lidar. So points are now in lidar frame, not reference frame.
        car_from_ref = np.linalg.inv(data["ref_from_car"])
        lidar_to_ref = data["lidar_to_ref"] if 'lidar_to_ref' in data.keys() else data["ref_to_lidar"] #legacy wrong naming
        coord_homo = np.hstack((coord, np.ones((coord.shape[0], 1))))
        # coord_homo = coord_homo @ car_from_ref.T #original
        coord_homo = coord_homo @ lidar_to_ref.T
        lidar_coord = coord_homo[:, :3]

        radar_path = os.path.join(self.data_root, data["radar_path"])
        points = RadarPointCloud.from_file(str(radar_path)).points.T
        coord = points[:, :3]

        # 2. Extract raw features
        raw_doppler = points[:, self.radar_mapping["doppler"]].reshape([-1, 1])
        raw_rcs = points[:, self.radar_mapping["rcs"]].reshape([-1, 1])
        
        # 3. Apply Z-score normalization
        # A small epsilon (1e-6) is added to the denominator to prevent division by zero
        doppler = (raw_doppler - self.doppler_mean) / (self.doppler_std + 1e-6)
        rcs = (raw_rcs - self.rcs_mean) / (self.rcs_std + 1e-6)

        image_path = os.path.join(self.data_root, data.get("cam_front_path", ""))
        color = color.astype(np.float32)
        if self.if_sweep:
            data_dict = dict(
                coord=coord,
                doppler=doppler,
                rcs=rcs,
                lidar_coord=lidar_coord,
                color=color,
                normal=normal,
                strength=strength, 
                image_path=image_path,
                frame_pcd_offset=frame_pcd_offset,
                name=self.get_data_name(idx),
            )
        else:
            data_dict = dict(
                coord=coord,
                doppler=doppler,
                rcs=rcs,
                lidar_coord=lidar_coord,
                color=color,
                normal=normal,
                strength=strength,            
                image_path=image_path,
                name=self.get_data_name(idx),
            )

        if self.debug:
            self.plot_debug(data_dict)
        return data_dict

    def plot_debug(self, data_dict, save_dir="./debug_vis"):
        os.makedirs(save_dir, exist_ok=True)
        lidar_coord = data_dict["lidar_coord"]
        color = data_dict["color"].astype(np.uint8)
        coord_radar = data_dict["coord"]
        raw_doppler = data_dict["doppler"]
        
        # Load image
        img = cv2.cvtColor(cv2.imread(data_dict["image_path"]), cv2.COLOR_BGR2RGB) / 255    
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10),)
        
        # 1. Plot Image
        axes[0].imshow(img)
        axes[0].axis('off')
        
        # 2. Plot Point Cloud Scatter
        axes[1].scatter(lidar_coord[:, 0], lidar_coord[:, 1], c=color / 255, s=0.1)
        axes[1].scatter(coord_radar[:, 0], coord_radar[:, 1], c=raw_doppler, cmap='viridis', s=1)
        # Ensure 1 unit on x equals 1 unit on y
        axes[1].set_aspect('equal', adjustable='box')
        
        # Set limits to 50m in all directions (assuming 0,0 is center or start)
        # If your data is centered at 0, use (-50, 50). 
        # If it starts at 0, use (0, 50). Below assumes a -50 to 50 spread.
        limit = 50
        axes[1].set_xlim(-limit, limit)
        axes[1].set_ylim(-limit, limit)
        
        axes[1].axis('off') 
        
        save_path = os.path.join(save_dir, f"{data_dict['name']}.png")
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

    def get_data_name(self, idx):
        try:

            return self.data_list[idx % len(self.data_list)]["radar_token"]
        except:
            return self.data_list[idx % len(self.data_list)]["token"]
    @staticmethod
    def get_normals(cam_center, coords):
        Cs = np.repeat(cam_center.reshape((1, -1)), coords.shape[0], axis=0)
        view_dirs = coords - Cs
        view_dirs = view_dirs / np.linalg.norm(view_dirs, axis=-1, keepdims=True)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        pcd.estimate_normals()
        normals = np.asarray(pcd.normals)
        dot_product = np.sum(normals * view_dirs, axis=-1)
        flip_mask = dot_product > 0
        normals[flip_mask] = -normals[flip_mask]
        normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals