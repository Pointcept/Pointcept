"""
Preprocessing Script for ARKitScenes

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pathlib import Path
import argparse
import os
import pandas as pd
import camtools as ct
import numpy as np
from scipy.spatial import cKDTree
import shutil
import open3d as o3d
import cv2
import glob
import numpy as np
import os
import plyfile
import multiprocessing as mp
from rotation import convert_angle_axis_to_matrix3


def correspondenceGet(mesh, K, T, img_size, coords_gt):
    height, width = img_size
    if np.isnan(T).any() or np.isinf(T).any():
        return None
    depth = ct.raycast.mesh_to_im_depth(
        mesh=mesh, K=K, T=np.linalg.inv(T), height=height, width=width
    )
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


def correspondenceSave(mesh, scene_dir, coords_gt, output_dir, img_size, Ks, Ts):
    os.makedirs(output_dir, exist_ok=True)
    scene_dir = Path(scene_dir)
    index_gt = [
        img_name[:-4].split("_")[1]
        for img_name in os.listdir(str(scene_dir / "color"))
        if img_name.endswith(".png")
    ]
    index_gt = sorted(index_gt, key=lambda x: float(x))

    coords_gt_ = coords_gt
    pixels_ = []
    coords_ = []

    for i, (K, T) in enumerate(zip(Ks, Ts)):
        coord_dict = correspondenceGet(mesh, K, T, img_size, coords_gt)
        if coord_dict is None:
            correspondences = -np.ones((1, 3))
        else:
            pixels_ = coord_dict["pixel"]
            coords_ = coord_dict["coord"]
            tree = cKDTree(coords_gt_)
            dis, idx = tree.query(coords_, k=1)
            idx_valid = idx[dis < 0.01]
            pixels_valid = pixels_[dis < 0.01]
            correspondences = np.hstack((pixels_valid, idx_valid.reshape(-1, 1)))
        np.save(Path(output_dir) / "{}.npy".format(index_gt[i]), correspondences)


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, "rb") as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata["vertex"].data).values
        faces = np.stack(plydata["face"].data["vertex_indices"], axis=0)
        return vertices, faces


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec**2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv**2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def TrajStringToMatrix(traj_str):
    """convert traj_str into translation and rotation matrices
    Args:
        traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
        The file has seven columns:
        * Column 1: timestamp
        * Columns 2-4: rotation (axis-angle representation in radians)
        * Columns 5-7: translation (usually in meters)

    Returns:
        ts: translation matrix
        Rt: rotation matrix
    """
    # line=[float(x) for x in traj_str.split()]
    # ts = line[0];
    # R = cv2.Rodrigues(np.array(line[1:4]))[0];
    # t = np.array(line[4:7]);
    # Rt = np.concatenate((np.concatenate((R, t[:,np.newaxis]), axis=1), [[0.0,0.0,0.0,1.0]]), axis=0)
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


def handle_process(
    scene_path,
    output_path,
    pointclouds_root,
    frame_gap,
    parse_pointclouds,
    parse_depths,
):
    print("loading from ", scene_path)
    split = os.path.basename(os.path.dirname(scene_path))
    scene_id = os.path.basename(scene_path)
    img_folder = os.path.join(scene_path, f"{scene_id}_frames", "lowres_wide")
    traj_file = os.path.join(scene_path, f"{scene_id}_frames", "lowres_wide.traj")
    pc_output_path = os.path.join(output_path, split, f"{scene_id}")
    pc_input_path = os.path.join(pointclouds_root, split, f"{scene_id}")
    im_output_path = os.path.join(output_path, "images", split, f"{scene_id}")
    os.makedirs(im_output_path, exist_ok=True)
    if not os.path.exists(img_folder):
        frame_ids = []
    else:
        images = sorted(
            glob.glob(os.path.join(img_folder, "*.png")),
            key=lambda x: float(x.split("/")[-1].split("_")[1][:-4]),
        )
        frame_ids = [os.path.basename(x) for x in images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        video_id = img_folder.split("/")[-2].split("_frames")[0]
        frame_ids.sort()
        frame_ids = frame_ids[::frame_gap]
    if len(frame_ids) > 0:
        height, width = cv2.imread(images[0]).shape[:2]
        if os.path.exists(traj_file):
            with open(traj_file) as f:
                traj = f.readlines()
            # convert traj to json dict
            poses_from_traj = {}
            for line in traj:
                traj_timestamp = line.split(" ")[0]
                poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = (
                    TrajStringToMatrix(line)[1].tolist()
                )

            if os.path.exists(traj_file):
                poses = poses_from_traj
            else:
                poses = {}

            # get img dlc
            poses_keys = list(poses.keys())
            intrinsics = []
            poses_list = []
            Ks_path = os.path.join(im_output_path, "intrinsic")
            Ts_path = os.path.join(im_output_path, "pose")
            img_path = os.path.join(im_output_path, "color")

            os.makedirs(Ks_path, exist_ok=True)
            os.makedirs(Ts_path, exist_ok=True)
            os.makedirs(img_path, exist_ok=True)
            if parse_depths:
                depth_path = os.path.join(im_output_path, "depth")
                os.makedirs(depth_path, exist_ok=True)
            for frame_id in frame_ids:
                shutil.copy(
                    os.path.join(
                        scene_path,
                        f"{scene_id}_frames",
                        "lowres_wide",
                        f"{video_id}_{frame_id}.png",
                    ),
                    img_path,
                )
                if parse_depths:
                    shutil.copy(
                        os.path.join(
                            scene_path,
                            f"{scene_id}_frames",
                            "lowres_depth",
                            f"{video_id}_{frame_id}.png",
                        ),
                        depth_path,
                    )
                intrinsic_fn = os.path.join(
                    scene_path,
                    f"{scene_id}_frames",
                    "lowres_wide_intrinsics",
                    f"{video_id}_{frame_id}.pincam",
                )
                intrinsic = st2_camera_intrinsics(intrinsic_fn)
                intrinsics.append(intrinsic)
                np.save(os.path.join(Ks_path, f"{video_id}_{frame_id}.npy"), intrinsic)
                frame_id = f"{round(float(frame_id), 3):.3f}"
                closest_key = min(
                    poses_keys, key=lambda k: abs(float(k) - float(frame_id))
                )
                np.save(
                    os.path.join(Ts_path, f"{video_id}_{frame_id}.npy"),
                    poses[closest_key],
                )
                poses_list.append(poses[closest_key])
            # match extrinsics
            intrinsics = np.stack(intrinsics)
            poses = np.stack(poses_list)

            ply_path = os.path.join(scene_path, f"{scene_id}_3dod_mesh.ply")
            if parse_pointclouds:
                vertices, faces = read_plymesh(ply_path)
                coords = vertices[:, :3]
                colors = vertices[:, 3:6]
                normals = vertex_normal(coords, faces)
                data_dict = dict(coord=coords, color=colors, normal=normals)
                os.makedirs(pc_output_path, exist_ok=True)
                for key in data_dict.keys():
                    np.save(os.path.join(pc_output_path, f"{key}.npy"), data_dict[key])
            else:
                coords = np.load(os.path.join(pc_input_path, "coord.npy"))

            # Save img data
            mesh = o3d.io.read_triangle_mesh(ply_path)
            correspondenceSave(
                mesh,
                im_output_path,
                coords,
                os.path.join(im_output_path, "correspondence"),
                (height, width),
                intrinsics,
                poses,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ArkitScenes dataset containing 3dod folder",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
    )
    parser.add_argument(
        "--pointclouds_root",
        default="data/arkitscenes",
        type=str,
        help="Input path where previous pointclouds folder located",
    )
    parser.add_argument(
        "--frame_gap", default=50, type=int, help="Frame gap for processing"
    )
    parser.add_argument(
        "--num_workers",
        default=mp.cpu_count(),
        type=int,
        help="Num workers for preprocessing.",
    )
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="thread_id",
    )
    parser.add_argument(
        "--parse_pointclouds", action="store_true", help="Whether parse point clouds"
    )
    parser.add_argument(
        "--parse_depths", action="store_true", help="Whether parse depths"
    )
    opt = parser.parse_args()
    # Load scene paths
    metadata_path = f"{opt.dataset_root}/3dod/metadata.csv"
    metadata = pd.read_csv(metadata_path)
    # Initialize the splits dictionary
    splits = {}
    scene_paths = []
    # Populate the splits dictionary
    for index, row in metadata.iterrows():
        scene_id = str(row["video_id"])
        split = row["fold"]
        splits[scene_id] = split
        scene_paths.append(os.path.join(opt.dataset_root, "3dod", split, scene_id))
    # Preprocess data.
    scene_paths_list = np.array_split(scene_paths, opt.num_workers)
    scene_paths_ = scene_paths_list[opt.thread_id]
    print("Processing scenes...")
    for scene_path in scene_paths_:
        handle_process(
            scene_path,
            opt.output_root,
            opt.pointclouds_root,
            opt.frame_gap,
            opt.parse_pointclouds,
            opt.parse_depths,
        )
