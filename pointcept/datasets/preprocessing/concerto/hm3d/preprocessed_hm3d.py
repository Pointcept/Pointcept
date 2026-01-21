"""
Preprocessing Script for hm3d using habitat-sim
https://github.com/facebookresearch/habitat-sim/blob/main/examples/tutorials/notebooks/ECCV_2020_Navigation.ipynb

Author: Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import itertools
import multiprocessing as mp
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple
import habitat_sim
import imageio
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree
import trimesh
import sys

sys.path.append("../../hm3d/")
from hm3d_constants import CLASS_LABELS_40


from common.utils import (
    convert_heading_to_quaternion,
    get_topdown_map,
)

MPCAT2INDEX = dict([(CLASS_LABELS_40[i], i) for i in range(40)])
MPCAT2INDEX["unlabeled"] = -1
CAT2INDEX = dict()
with open(Path("../../hm3d") / "hm3dsem_category_mappings.tsv") as f:
    f.readline()  # raw_category	category	mpcat40
    lines = f.readlines()
for line in lines:
    line = line.strip().split("\t")
    if len(line) == 2:
        # L2164: "\tunknown\tunlabeled"
        line.insert(0, "")
    CAT2INDEX[line[1]] = MPCAT2INDEX[line[2]]


def correspondenceGet(depth, K, T, img_size):
    height, width = img_size
    if np.isnan(T).any() or np.isinf(T).any():
        return None
    pixel = np.transpose(np.indices((width, height)), (2, 1, 0))
    pixel = pixel.reshape((-1, 2))
    pixel = np.hstack((pixel, np.ones((pixel.shape[0], 1))))
    depth = depth.reshape((-1, 1))
    valid = ~np.isinf(depth).squeeze(-1)
    coord = np.zeros_like(pixel, dtype=np.float32)
    coord[valid] = depth[valid] * (np.linalg.inv(K) @ pixel[valid].T).T  # coord_camera
    coord[valid] = coord[valid] @ T[:3, :3].T + T[:3, 3]  # column then row
    pixel = pixel[valid]
    coord = coord[valid]
    if coord.shape[0] == 0:
        return None
    pixel = pixel[:, :2]
    coord_dict = {"pixel": pixel, "coord": coord}
    return coord_dict


def uv_to_texture_color(uv, texture):
    width, height = texture.size
    u = int(uv[0] * width)
    v = int((1 - uv[1]) * height)  # Flip y-axis for image coordinates
    if 0 <= u < width and 0 <= v < height:
        return texture.getpixel((u, v))
    else:
        return 0, 0, 0


def load_hex_mapping(mapping_path):
    hex2label = {}
    with open(mapping_path) as f:
        f.readline()  # remove 'HM3D Semantic Annotations\n'
        lines = f.readlines()  # get the left
    for line in lines:
        line = line.strip().split(",")
        cat = line[2].strip('"')
        if cat == "trashcan":
            cat = "trash can"
        elif cat == "fridge":
            cat = "refrigerator"
        hex2label[line[1]] = dict(instance=line[0], segment=CAT2INDEX[cat])
    return hex2label


def make_habitat_configuration(
    scene_path: str,
    hfov: int = 90,
    resolution: Tuple[int] = (300, 300),
    stage_json_path: Optional[str] = None,
):
    # simulator configuration
    backend_cfg = habitat_sim.SimulatorConfiguration()
    if stage_json_path is not None:
        backend_cfg.scene_dataset_config_file = stage_json_path
        backend_cfg.scene_id = "habitat/" + scene_path.split("/")[-1]
    else:
        backend_cfg.scene_id = scene_path

    # agent configuration
    rgb_sensor_cfg = habitat_sim.CameraSensorSpec()
    rgb_sensor_cfg.uuid = "rgba"
    rgb_sensor_cfg.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_cfg.resolution = resolution
    rgb_sensor_cfg.hfov = hfov
    rgb_sensor_cfg.position = [0.0, 0.0, 0.0]
    depth_sensor_cfg = habitat_sim.CameraSensorSpec()
    depth_sensor_cfg.uuid = "depth"
    depth_sensor_cfg.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_cfg.resolution = resolution
    depth_sensor_cfg.hfov = hfov
    depth_sensor_cfg.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor_cfg, depth_sensor_cfg]

    return habitat_sim.Configuration(backend_cfg, [agent_cfg])


def robust_load_sim(scene_path: str, **kwargs: Any) -> habitat_sim.Simulator:
    sim_cfg = make_habitat_configuration(scene_path, **kwargs)
    hsim = habitat_sim.Simulator(sim_cfg)
    if not hsim.pathfinder.is_loaded:
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        hsim.recompute_navmesh(hsim.pathfinder, navmesh_settings)
    return hsim


def get_floor_heights(
    sim: habitat_sim.Simulator, max_points_to_sample: int = 20000
) -> List[Dict[str, float]]:
    """Get heights of different floors in a scene. This is done in two steps.
    (1) Randomly samples navigable points in the scene.
    (2) Cluster the points based on discretized y coordinates to get floors.

    Args:
        sim: habitat simulator instance
        max_points_to_sample: number of navigable points to randomly sample
    """
    nav_points = []
    for _ in range(max_points_to_sample):
        nav_points.append(sim.pathfinder.get_random_navigable_point())
    nav_points = np.stack(nav_points, axis=0)
    y_coors = np.around(nav_points[:, 1], decimals=1)
    # cluster Y coordinates
    clustering = DBSCAN(eps=0.2, min_samples=2000).fit(y_coors[:, np.newaxis])
    c_labels = clustering.labels_
    n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
    # get floor extents in Y
    # each cluster corresponds to points from 1 floor
    floor_extents = []
    core_sample_y = y_coors[clustering.core_sample_indices_]
    core_sample_labels = c_labels[clustering.core_sample_indices_]
    for i in range(n_clusters):
        floor_min = core_sample_y[core_sample_labels == i].min().item()
        floor_max = core_sample_y[core_sample_labels == i].max().item()
        floor_mean = core_sample_y[core_sample_labels == i].mean().item()
        floor_extents.append({"min": floor_min, "max": floor_max, "mean": floor_mean})
    floor_extents = sorted(floor_extents, key=lambda x: x["mean"])

    # reject floors that have too few points
    max_points = 0
    for fext in floor_extents:
        top_down_map = get_topdown_map(sim.pathfinder, fext["mean"])
        max_points = max(np.count_nonzero(top_down_map), max_points)
    clean_floor_extents = []
    for fext in floor_extents:
        top_down_map = get_topdown_map(sim.pathfinder, fext["mean"])
        num_points = np.count_nonzero(top_down_map)
        if num_points < 0.2 * max_points:
            continue
        clean_floor_extents.append(fext)

    return clean_floor_extents


def get_navmesh_extents_at_y(
    sim: habitat_sim.Simulator, y_bounds: Optional[Tuple[float]] = None
) -> Tuple[float]:
    if y_bounds is None:
        lower_bound, upper_bound = sim.pathfinder.get_bounds()
    else:
        assert len(y_bounds) == 2
        assert y_bounds[0] < y_bounds[1]
        navmesh_vertices = np.array(sim.pathfinder.build_navmesh_vertices())
        navmesh_vertices = navmesh_vertices[
            (y_bounds[0] <= navmesh_vertices[:, 1])
            & (navmesh_vertices[:, 1] <= y_bounds[1])
        ]
        lower_bound = navmesh_vertices.min(axis=0)
        upper_bound = navmesh_vertices.max(axis=0)
    return (lower_bound, upper_bound)


def get_dense_navmesh_vertices(
    sim: habitat_sim.Simulator, sampling_resolution: float = 0.5
) -> np.ndarray:

    navmesh_vertices = []
    floor_extents = get_floor_heights(sim)
    for fext in floor_extents:
        l_bound, u_bound = get_navmesh_extents_at_y(
            sim, y_bounds=(fext["min"] - 0.5, fext["max"] + 0.5)
        )
        x_range = np.arange(l_bound[0].item(), u_bound[0].item(), sampling_resolution)
        y = fext["mean"]
        z_range = np.arange(l_bound[2].item(), u_bound[2].item(), sampling_resolution)
        for x, z in itertools.product(x_range, z_range):
            if sim.pathfinder.is_navigable(np.array([x, y, z])):
                navmesh_vertices.append((np.array([x, y, z])))
    if len(navmesh_vertices) > 0:
        navmesh_vertices = np.stack(navmesh_vertices, axis=0)
    else:
        navmesh_vertices = np.zeros((0, 3))
    return navmesh_vertices


def get_dense_mesh_vertices(
    sim: habitat_sim.Simulator, room_bounds: dict, sampling_resolution: float = 0.5
) -> np.ndarray:
    navmesh_vertices = []
    for room_id, room_bound in room_bounds.items():
        x_range = np.arange(
            room_bound["min_x"] + 0.5, room_bound["max_x"] - 0.5, sampling_resolution
        )
        y_range = np.arange(
            room_bound["min_y"] + 0.5, room_bound["max_y"] - 0.5, sampling_resolution
        )
        z_range = np.arange(
            room_bound["min_z"] + 0.5, room_bound["max_z"] - 0.5, sampling_resolution
        )
        for x, y, z in itertools.product(x_range, y_range, z_range):
            navmesh_vertices.append((np.array([x, y, z])))
    if len(navmesh_vertices) > 0:
        navmesh_vertices = np.stack(navmesh_vertices, axis=0)
    else:
        navmesh_vertices = np.zeros((0, 3))
    return navmesh_vertices


def get_scene_name(scene_path, dataset):
    if dataset == "replica":
        scene_name = scene_path.split("/")[-2].split(".")[0]
    else:
        scene_name = scene_path.split("/")[-1].split(".")[0]
    return scene_name


def plySave(coords, colors, save_path):
    ply_path = save_path / "point_cloud.ply"
    trimesh.points.PointCloud(coords, colors=colors).export(ply_path)


def _aux_fn(args):
    (
        density,
        scene_path,
        save_prefix,
        hfov,
        resolution,
        sampling_resolution,
        num_rotations,
        sim_kwargs,
        parse_pointclouds,
        parse_depths,
    ) = args
    if sim_kwargs is None:
        sim_kwargs = {}
    sim = robust_load_sim(scene_path, hfov=hfov, resolution=resolution, **sim_kwargs)
    save_prefix = Path(save_prefix)
    rgb_save_prefix = save_prefix / "images"
    pc_save_prefix = save_prefix
    intrinsic_save_path = Path(rgb_save_prefix) / "intrinsic"
    intrinsic_save_path.mkdir(parents=True, exist_ok=True)
    np.save(intrinsic_save_path / "K.npy", K)
    scene_path = Path(scene_path)
    scene_label_path = scene_path.with_suffix(".semantic.glb")
    scene_mapping_path = scene_path.with_suffix(".semantic.txt")
    scene_name = scene_path.parent.name.replace("-", "_")
    scene_id = scene_name.split("_")[0]
    labeled = True if scene_label_path.is_file() else False

    # test split (900-1000) is reserved by official
    if 0 <= int(scene_id) < 800:
        split = "train"
    else:
        split = "val"
    print(f"Parsing scene {scene_name} in {split} split...")
    scene = trimesh.load(scene_path)
    if labeled:
        labeled_scene_ = trimesh.load(scene_label_path)
        labeled_scene = trimesh.Scene()
        for name, mesh in labeled_scene_.geometry.items():
            # some case, name in labeled scene and scene is not matched, so only use chunk id
            labeled_scene.add_geometry(mesh, geom_name=name.split("_")[0])
        del labeled_scene_
        hex2label = load_hex_mapping(scene_mapping_path)

    room_dict = {}
    room_bound = {}
    for name, mesh in scene.geometry.items():
        room_id = "_".join(name.split("_")[1:3]).replace("group", "").replace("sub", "")
        if room_id not in room_dict.keys():
            room_dict[room_id] = trimesh.Scene()
            room_bound[room_id] = {
                "min_x": np.inf,
                "max_x": -np.inf,
                "min_y": np.inf,
                "max_y": -np.inf,
                "min_z": np.inf,
                "max_z": -np.inf,
            }
        room_dict[room_id].add_geometry(mesh, geom_name=name.split("_")[0])
        bounding_box = mesh.bounds
        room_bound[room_id]["min_x"] = min(
            room_bound[room_id]["min_x"], bounding_box[0][0]
        )
        room_bound[room_id]["max_x"] = max(
            room_bound[room_id]["max_x"], bounding_box[1][0]
        )
        room_bound[room_id]["min_y"] = min(
            room_bound[room_id]["min_y"], bounding_box[0][1]
        )
        room_bound[room_id]["max_y"] = max(
            room_bound[room_id]["max_y"], bounding_box[1][1]
        )
        room_bound[room_id]["min_z"] = min(
            room_bound[room_id]["min_z"], bounding_box[0][2]
        )
        room_bound[room_id]["max_z"] = max(
            room_bound[room_id]["max_z"], bounding_box[1][2]
        )

    # Get dense navmesh vertices
    navmesh_vertices = get_dense_mesh_vertices(
        sim, room_bound, sampling_resolution=sampling_resolution
    )
    room_vertices = dict()
    for navmesh_vertex in navmesh_vertices:
        for room_id, bound in room_bound.items():
            if (
                bound["min_x"] < navmesh_vertex[0] < bound["max_x"]
                and bound["min_z"] < navmesh_vertex[2] < bound["max_z"]
                and bound["min_y"] < navmesh_vertex[1] < bound["max_y"]
            ):
                if room_id not in room_vertices.keys():
                    room_vertices[room_id] = []
                room_vertices[room_id].append(navmesh_vertex)
                break

    for room_id, scene in room_dict.items():
        if room_id not in room_vertices.keys():
            continue
        np.random.seed(int(scene_id + room_id.replace("_", "")))
        room_coord = []
        room_color = []
        room_normal = []
        if labeled:
            room_label_color = []

        for name in scene.geometry.keys():
            mesh = scene.geometry[name]
            num_points = int(np.sum(mesh.area_faces) / density**2)
            if num_points == 0:
                continue
            coords, face_indices = mesh.sample(num_points, return_index=True)
            faces = mesh.faces[face_indices]
            triangles = mesh.vertices[faces]
            bary_coords = trimesh.triangles.points_to_barycentric(triangles, coords)
            uv_coords = mesh.visual.uv[faces]
            sampled_uvs = np.einsum("ijk,ij->ik", uv_coords, bary_coords)

            pbr_material = mesh.visual.material
            texture_image = pbr_material.baseColorTexture
            if texture_image is None:
                continue
            colors = np.array(
                [uv_to_texture_color(uv, texture_image) for uv in sampled_uvs]
            )
            normals = mesh.vertex_normals[faces]
            normals = np.einsum("ijk,ij->ik", normals, bary_coords)
            room_coord.append(coords)
            room_color.append(colors)
            room_normal.append(normals)

            if labeled:
                labeled_mash = labeled_scene.geometry[name]
                label_texture_image = labeled_mash.visual.material.baseColorTexture
                label_color = np.array(
                    [uv_to_texture_color(uv, label_texture_image) for uv in sampled_uvs]
                )
                room_label_color.append(label_color)
        if len(room_coord) == 0:
            continue

        room_coord = np.concatenate(room_coord, axis=0).astype("float32")
        if parse_pointclouds:
            room_color = np.concatenate(room_color, axis=0).astype("uint8")
            room_normal = np.concatenate(room_normal, axis=0).astype("float32")
            data_dict = dict(coord=room_coord, color=room_color, normal=room_normal)

            if labeled:
                room_label_color = np.concatenate(room_label_color, axis=0)
                instance_label_color = np.unique(room_label_color, axis=0)
                room_instance = -np.ones(len(room_label_color), dtype="int16")
                room_segment = -np.ones(len(room_label_color), dtype="int16")
                for i in range(len(instance_label_color)):
                    label_color = instance_label_color[i]
                    label_hex = "{c[0]:02x}{c[1]:02x}{c[2]:02x}".format(c=label_color)
                    mask = np.all(room_label_color == label_color, axis=-1)
                    room_instance[mask] = i
                    if label_hex.upper() in hex2label.keys():
                        room_segment[mask] = hex2label[label_hex.upper()]["segment"]
                data_dict["instance"] = room_instance
                data_dict["segment"] = room_segment
            pc_save_path_single = (
                Path(pc_save_prefix) / split / "_".join([scene_name, room_id])
            )
            os.makedirs(pc_save_path_single, exist_ok=True)
            print(f"Saving data to {pc_save_path_single}...")
            for key, value in data_dict.items():
                np.save(pc_save_path_single / f"{key}.npy", value)

        agent = sim.get_agent(0)
        rgb_paths = []
        depth_paths = []
        if not sim.pathfinder.is_loaded:
            sim.close()
            return rgb_paths, depth_paths

        vertex = sum(room_vertices[room_id]) / len(room_vertices[room_id])

        T0 = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
        Ts = []
        loc_ = vertex + np.random.uniform(-0.25, 0.25, size=vertex.shape)
        count = 0
        loc = loc_ @ T0[:3, :3].T
        init_angle = random.uniform(0, 90)
        pose_save_path = (
            Path(rgb_save_prefix) / split / "_".join([scene_name, room_id]) / "pose"
        )
        pose_save_path.mkdir(parents=True, exist_ok=True)
        rgb_path = (
            Path(rgb_save_prefix) / split / "_".join([scene_name, room_id]) / "color"
        )
        os.makedirs(rgb_path, exist_ok=True)
        if parse_depths:
            depth_path = (
                Path(rgb_save_prefix)
                / split
                / "_".join([scene_name, room_id])
                / "depth"
            )
            os.makedirs(depth_path, exist_ok=True)
        for heading in np.linspace(init_angle, init_angle + 360, num_rotations + 1)[
            :-1
        ]:
            rot = convert_heading_to_quaternion(heading)
            angle = np.radians(-heading)  # Replace 45 with the desired angle in degrees
            R_y = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )
            T = np.eye(4)
            T[:3, :3] = T0[:3, :3] @ R_y
            T[:3, 3] = loc_
            Ts.append(T)
            np.save(pose_save_path / f"{count}.npy", T)
            agent_state = agent.get_state()
            agent_state.position = loc
            agent_state.rotation = rot
            agent.set_state(agent_state, reset_sensors=True)
            obs = sim.get_sensor_observations()
            rgb = obs["rgba"][..., :3]
            depth = obs["depth"]
            rgb_path_ = rgb_path / f"{count}.png"
            imageio.imwrite(rgb_path_, rgb)
            if parse_depths:
                depth_path_ = depth_path / f"{count}.png"
                depth_img = (depth * 1000).astype(np.uint16)  # Convert to mm
                imageio.imwrite(depth_path_, depth_img)
            coord_dict = correspondenceGet(depth, K, T, RESOLUTION)
            if coord_dict is None:
                correspondences = np.ones((1, 3))
            else:
                pixels_ = coord_dict["pixel"]
                coords_ = coord_dict["coord"]
                tree = cKDTree(room_coord)
                dis, idx = tree.query(coords_, k=1)
                idx_valid = idx[dis < 0.01]
                pixels_valid = pixels_[dis < 0.01]
                co_save_path = (
                    Path(rgb_save_prefix)
                    / split
                    / "_".join([scene_name, room_id])
                    / "correspondence"
                )
                co_save_path.mkdir(parents=True, exist_ok=True)
                correspondences = np.hstack((pixels_valid, idx_valid.reshape(-1, 1)))
            np.save(co_save_path / "{}.npy".format(count), correspondences)
            count += 1
    sim.close()


HFOV = 90
HFOV_rad = np.deg2rad(HFOV)
RESOLUTION = [720, 720]
NUM_ROTATIONS = 4
F = RESOLUTION[1] / (2 * np.tan(HFOV_rad / 2))
CX, CY = RESOLUTION[1] / 2, RESOLUTION[0] / 2
K = np.array([[F, 0, CX], [0, F, CY], [0, 0, 1]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--parse_pointclouds",
        action="store_true",
        help="Parse point clouds from the scenes",
    )
    parser.add_argument(
        "--parse_depths",
        action="store_true",
        help="Parse depths from the scenes",
    )
    parser.add_argument(
        "--density",
        default=0.02,
        type=float,
        help="Sampling density on mesh surface (m)",
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--worker_id",
        default=0,
        type=int,
        help="Num workers for preprocessing.",
    )
    args = parser.parse_args()

    scene_list = glob.glob(os.path.join(args.dataset_root, "*", "*.glb"))
    scene_list = [scene for scene in scene_list if not scene.endswith("semantic.glb")]
    scene_list = [scene for scene in scene_list if not scene.endswith("basis.glb")]
    scene_list = sorted(scene_list, key=lambda x: int(x.split("/")[-2].split("-")[0]))
    assert len(scene_list) == 900
    print(
        f"Found {len(scene_list)} scenes in {args.dataset_root},using {args.num_workers} workers."
    )
    scene_list = [scene_list[i :: args.num_workers] for i in range(args.num_workers)][
        args.worker_id
    ]

    pc_train_dir = os.path.join(args.output_root, "pointclouds", "train")
    pc_val_dir = os.path.join(args.output_root, "pointclouds", "val")
    existing_scene_ids = set()

    for pc_dir in [pc_train_dir, pc_val_dir]:
        if os.path.exists(pc_dir):
            for folder in os.listdir(pc_dir):
                if "_" in folder:
                    scene_id = folder.split("_")[0]
                    existing_scene_ids.add(scene_id)

    scene_list = [
        scene
        for scene in scene_list
        if scene.split("/")[-2].split("-")[0] not in existing_scene_ids
    ]
    print(f"Processing {len(scene_list)} scenes...")

    for scene_path in scene_list:
        inputs = (
            args.density,
            scene_path,
            args.output_root,
            HFOV,
            RESOLUTION,
            0.5,
            NUM_ROTATIONS,
            None,
            args.parse_pointclouds,
            args.parse_depths,
        )
        _aux_fn(inputs)
