"""
Preprocessing Script for ScanNet 20/200

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import json
import torch
import shutil
import os
import torch
import camtools as ct
import open3d as o3d
from scipy.spatial import cKDTree
import cv2
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import OrderedDict
from scipy.spatial.transform import Rotation
from pathlib import Path
import re

try:
    import pointseg
except:
    # Pointseg is located in libs/pointseg
    warnings.warn("Pointseg is not installed, superpoint segmentation will be skipped.")
    pointseg = None

REGEXPR_DSLR = re.compile(r".*DSC(?P<frameid>\d+).JPG$")
REGEXPR_IPHONE = re.compile(r"frame_(?P<frameid>\d+).jpg$")


def pose_from_qwxyz_txyz(elems):
    qw, qx, qy, qz, tx, ty, tz = map(float, elems)
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat((qx, qy, qz, qw)).as_matrix()
    pose[:3, 3] = (tx, ty, tz)
    c2w = np.linalg.inv(pose)  # returns cam2world
    return torch.tensor(c2w, dtype=torch.float32)


def get_frame_number(name, cam_type="dslr"):
    if cam_type == "dslr":
        regex_expr = REGEXPR_DSLR
    elif cam_type == "iphone":
        regex_expr = REGEXPR_IPHONE
    else:
        raise NotImplementedError(f"wrong {cam_type=} for get_frame_number")
    matches = re.match(regex_expr, name)
    return matches["frameid"]


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def load_sfm(sfm_dir, cam_type="dslr"):
    # load cameras
    with open(os.path.join(sfm_dir, "cameras.txt"), "r") as f:
        raw = f.read().splitlines()[3:]  # skip header

    intrinsics = {}
    for camera in raw:
        camera = camera.split(" ")
        intrinsics[int(camera[0])] = [camera[1]] + [float(cam) for cam in camera[2:]]
        ins = intrinsics[int(camera[0])]
    w, h, ins = undistort_ins(ins)
    # load images
    with open(os.path.join(sfm_dir, "images.txt"), "r") as f:
        raw = f.read().splitlines()
        raw = [line for line in raw if not line.startswith("#")]  # skip header

    img_idx = {}
    img_infos = {}
    for image, points in zip(raw[0::2], raw[1::2]):
        image = image.split(" ")
        points = points.split(" ")

        idx = image[0]
        img_name = image[-1]
        assert img_name not in img_idx, "duplicate db image: " + img_name
        img_idx[img_name] = idx  # register image name

        # current_points2D = {int(i): (float(x), float(y))
        #                     for i, x, y in zip(points[2::3], points[0::3], points[1::3]) if i != '-1'}
        img_infos[idx] = dict(
            intrinsics=ins,
            path=img_name,
            frame_id=get_frame_number(img_name, cam_type),
            cam_to_world=pose_from_qwxyz_txyz(image[1:-2]),
            width=w,
            height=h,
        )

    return img_idx, img_infos


def undistort_ins(intrinsics):
    camera_type = intrinsics[0]

    width = int(intrinsics[1])
    height = int(intrinsics[2])
    fx = intrinsics[3]
    fy = intrinsics[4]
    cx = intrinsics[5]
    cy = intrinsics[6]
    distortion = np.array(intrinsics[7:])

    K = np.zeros([3, 3])
    K[0, 0] = fx
    K[0, 2] = cx
    K[1, 1] = fy
    K[1, 2] = cy
    K[2, 2] = 1

    K = colmap_to_opencv_intrinsics(K)
    if camera_type == "OPENCV_FISHEYE":
        assert len(distortion) == 4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K,
            distortion,
            (width, height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = width / 2.0
        new_K[1, 2] = height / 2.0

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )
    else:
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            K, distortion, (width, height), 1, (width, height), True
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            K, distortion, np.eye(3), new_K, (width, height), cv2.CV_32FC1
        )

    # undistorted_image = cv2.remap(rgb, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # undistorted_mask = cv2.remap(mask, map1, map2, interpolation=cv2.INTER_LINEAR,
    #                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    new_K = opencv_to_colmap_intrinsics(new_K)
    new_K = torch.tensor(new_K, dtype=torch.float32)
    return width, height, new_K


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
    return coord_dict, depth


def correspondenceSave(
    mesh, scene_dir, coords_gt, Ks, Ts, output_dir, img_size, parse_depths
):
    os.makedirs(output_dir / "correspondence", exist_ok=True)
    scene_dir = Path(scene_dir)
    index_gt = [
        img_name.split(".")[0]
        for img_name in os.listdir(str(scene_dir / "color"))
        if img_name.endswith(".JPG")
    ]
    index_gt = sorted(index_gt, key=lambda x: int(x.split("DSC")[1]))

    Ks = Ks[:, :3, :3]
    coords_gt_ = coords_gt
    pixels_ = []
    coords_ = []

    for i, (K, T) in enumerate(zip(Ks, Ts)):
        try:
            coord_dict, depth = correspondenceGet(mesh, K, T, img_size, coords_gt)
        except:
            print(output_dir)
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
        np.save(
            output_dir / "correspondence" / "{}.npy".format(index_gt[i]),
            correspondences,
        )
        if parse_depths:
            os.makedirs(output_dir / "depth", exist_ok=True)
            depth = depth * 1000
            depth = np.where(np.isinf(depth), 65535, depth).astype(np.uint16)
            depth = depth.reshape(img_size)
            cv2.imwrite(output_dir / "depth" / "{}.png".format(index_gt[i]), depth)


def handle_process(
    name,
    split,
    dataset_path,
    output_path,
    pointclouds_root,
    label_mapping,
    class2idx,
    ignore_index=-1,
    frame_gap=20,
    parse_pointclouds=False,
    parse_depths=False,
):
    pc_output_path = output_path / split
    im_output_path = output_path / "images" / split
    scene_path = dataset_path / "data" / name
    mesh_path = scene_path / "scans" / "mesh_aligned_0.05.ply"
    segs_path = scene_path / "scans" / "segments.json"
    anno_path = scene_path / "scans" / "segments_anno.json"
    # load mesh vertices and colors
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if parse_pointclouds:
        # extract mesh information
        mesh.compute_vertex_normals(normalized=True)
        coord = np.array(mesh.vertices).astype(np.float32)
        color = (np.array(mesh.vertex_colors) * 255).astype(np.uint8)
        normal = np.array(mesh.vertex_normals).astype(np.float32)
        # extract superpoint information
        if pointseg is not None:
            vertices_sp = torch.from_numpy(np.array(mesh.vertices).astype(np.float32))
            faces_sp = torch.from_numpy(np.array(mesh.triangles).astype(np.int64))
            superpoint = pointseg.segment_mesh(vertices_sp, faces_sp).numpy()
        else:
            superpoint = None

        # Save pointcloud data
        os.makedirs(pc_output_path / name, exist_ok=True)
        np.save(pc_output_path / name / "coord.npy", coord)
        np.save(pc_output_path / name / "color.npy", color)
        np.save(pc_output_path / name / "normal.npy", normal)
        if superpoint is not None:
            np.save(pc_output_path / "superpoint.npy", superpoint)

        if split != "test":
            # get label on vertices
            # load segments = vertices per segment ID
            with open(segs_path) as f:
                segments = json.load(f)
            # load anno = (instance, groups of segments)
            with open(anno_path) as f:
                anno = json.load(f)
            seg_indices = np.array(segments["segIndices"], dtype=np.uint32)
            num_vertices = len(seg_indices)
            assert num_vertices == len(coord)
            semantic_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index
            instance_gt = np.ones((num_vertices, 3), dtype=np.int16) * ignore_index

            # number of labels are used per vertex. initially 0
            # increment each time a new label is added
            instance_size = np.ones((num_vertices, 3), dtype=np.int16) * np.inf

            # keep track of the size of the instance (#vertices) assigned to each vertex
            # later, keep the label of the smallest instance for major label of vertices
            # store inf initially so that we can pick the smallest instance
            labels_used = np.zeros(num_vertices, dtype=np.int16)

            for idx, instance in enumerate(anno["segGroups"]):
                label = instance["label"]
                instance["label_orig"] = label
                # remap label
                instance["label"] = label_mapping.get(label, None)
                instance["label_index"] = class2idx.get(instance["label"], ignore_index)

                if instance["label_index"] == ignore_index:
                    continue
                # get all the vertices with segment index in this instance
                # and max number of labels not yet applied
                mask = np.isin(seg_indices, instance["segments"]) & (labels_used < 3)
                size = mask.sum()
                if size == 0:
                    continue

                # get the position to add the label - 0, 1, 2
                label_position = labels_used[mask]
                semantic_gt[mask, label_position] = instance["label_index"]
                # store all valid instance (include ignored instance)
                instance_gt[mask, label_position] = instance["objectId"]
                instance_size[mask, label_position] = size
                labels_used[mask] += 1

            # major label is the label of smallest instance for each vertex
            # use major label for single class segmentation
            # shift major label to the first column
            mask = labels_used > 1
            if mask.sum() > 0:
                major_label_position = np.argmin(instance_size[mask], axis=1)

                major_semantic_label = semantic_gt[mask, major_label_position]
                semantic_gt[mask, major_label_position] = semantic_gt[:, 0][mask]
                semantic_gt[:, 0][mask] = major_semantic_label

                major_instance_label = instance_gt[mask, major_label_position]
                instance_gt[mask, major_label_position] = instance_gt[:, 0][mask]
                instance_gt[:, 0][mask] = major_instance_label

            np.save(pc_output_path / name / "segment.npy", semantic_gt)
            np.save(pc_output_path / name / "instance.npy", instance_gt)
    else:
        coord = np.load(pointclouds_root / split / name / "coord.npy")
    rgb_path = scene_path / "dslr" / "undistorted_images"
    colmap_path = scene_path / "dslr" / "colmap"
    img_idxs, img_infos = load_sfm(str(colmap_path), cam_type="dslr")
    rgb_path = sorted(
        rgb_path.glob("*.JPG"), key=lambda x: int(str(x)[:-4].split("DSC")[1])
    )[::frame_gap]
    rgb_path = [str(path) for path in rgb_path]
    color_output_path = im_output_path / name / "color"
    os.makedirs(color_output_path, exist_ok=True)
    for rgb_path_i in rgb_path:
        shutil.copy(rgb_path_i, color_output_path)
    imgnames = [path.split("/")[-1] for path in rgb_path]
    img_idxs = [img_idxs[imgname] for imgname in imgnames]
    img_infos = [img_infos[img_idx] for img_idx in img_idxs]
    intrinsic_save_path = im_output_path / name / "intrinsic"
    pose_save_path = im_output_path / name / "pose"
    os.makedirs(intrinsic_save_path, exist_ok=True)
    os.makedirs(pose_save_path, exist_ok=True)
    Ks = []
    Ts = []
    for id, img_info in enumerate(img_infos):
        np.save(intrinsic_save_path / f"{id}.npy", img_info["intrinsics"])
        np.save(pose_save_path / f"{id}.npy", img_info["cam_to_world"])
        Ks.append(img_info["intrinsics"])
        Ts.append(img_info["cam_to_world"])
    Ks = np.stack(Ks)
    Ts = np.stack(Ts)

    if len(rgb_path) == 0 or len(img_infos) == 0:
        return
    h = img_infos[0]["height"]
    w = img_infos[0]["width"]
    correspondenceSave(
        mesh,
        im_output_path / name,
        coord,
        Ks,
        Ts,
        im_output_path / name,
        (h, w),
        parse_depths,
    )


def filter_map_classes(mapping, mapping_type):
    if mapping_type == "semantic":
        map_key = "semantic_map_to"
    elif mapping_type == "instance":
        map_key = "instance_map_to"
    else:
        raise NotImplementedError
    # create a dict with classes to be mapped
    # classes that don't have mapping are entered as x->x
    # otherwise x->y
    map_dict = OrderedDict()

    for i in range(mapping.shape[0]):
        row = mapping.iloc[i]
        class_name = row["class"]
        map_target = row[map_key]

        # map to None or some other label -> don't add this class to the label list
        try:
            if len(map_target) > 0:
                # map to None -> don't use this class
                if map_target == "None":
                    pass
                else:
                    # map to something else -> use this class
                    map_dict[class_name] = map_target
        except TypeError:
            # nan values -> no mapping, keep label as is
            if class_name not in map_dict:
                map_dict[class_name] = class_name

    return map_dict


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
        "--pointclouds_root",
        default="data/scannetpp",
        type=str,
        help="Input path where previous pointclouds folder located",
    )
    parser.add_argument(
        "--ignore_index",
        default=-1,
        type=int,
        help="Default ignore index.",
    )
    parser.add_argument(
        "--frame_gap", default=20, type=int, help="Frame gap for processing"
    )
    parser.add_argument(
        "--parse_pointclouds", action="store_true", help="Whether parse point clouds"
    )
    parser.add_argument(
        "--parse_depths", action="store_true", help="Whether parse depths"
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
        help="Thread id for parallel processing",
    )
    config = parser.parse_args()

    print("Loading meta data...")
    config.dataset_root = Path(config.dataset_root)
    config.output_root = Path(config.output_root)
    config.pointclouds_root = Path(config.pointclouds_root)

    train_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_train.txt",
        dtype=str,
    )
    print("Num samples in training split:", len(train_list))

    val_list = np.loadtxt(
        config.dataset_root / "splits" / "nvs_sem_val.txt",
        dtype=str,
    )
    print("Num samples in validation split:", len(val_list))

    test_list = np.loadtxt(
        config.dataset_root / "splits" / "sem_test.txt",
        dtype=str,
    )
    print("Num samples in testing split:", len(test_list))

    data_list = np.concatenate([train_list, val_list, test_list])
    split_list = np.concatenate(
        [
            np.full_like(train_list, "train"),
            np.full_like(val_list, "val"),
            np.full_like(test_list, "test"),
        ]
    )

    # Parsing label information and mapping
    segment_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in segment class list:", len(segment_class_names))

    instance_class_names = np.loadtxt(
        config.dataset_root / "metadata" / "semantic_benchmark" / "top100_instance.txt",
        dtype=str,
        delimiter=".",  # dummy delimiter to replace " "
    )
    print("Num classes in instance class list:", len(instance_class_names))

    label_mapping = pd.read_csv(
        config.dataset_root / "metadata" / "semantic_benchmark" / "map_benchmark.csv"
    )
    label_mapping = filter_map_classes(label_mapping, mapping_type="semantic")
    class2idx = {
        class_name: idx for (idx, class_name) in enumerate(segment_class_names)
    }
    # Load scene paths

    data_list_list = np.array_split(data_list, config.num_workers)
    data_list_ = data_list_list[config.thread_id]
    split_list_list = np.array_split(split_list, config.num_workers)
    split_list_ = split_list_list[config.thread_id]
    # Preprocess data.
    print("Processing scenes...")
    for data_list_i, split_list_i in zip(data_list_, split_list_):
        handle_process(
            data_list_i,
            split_list_i,
            config.dataset_root,
            config.output_root,
            config.pointclouds_root,
            label_mapping,
            class2idx,
            config.ignore_index,
            config.frame_gap,
            config.parse_pointclouds,
            config.parse_depths,
        )
