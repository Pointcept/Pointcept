"""
Preprocessing Script for S3DIS

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import numpy as np
import re
import json
from PIL import Image
from scipy.spatial import cKDTree
from pathlib import Path
import camtools as ct
import shutil

try:
    import open3d
except ImportError:
    import warnings

    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings

    warnings.warn("Please install trimesh for parsing normal")

area_mesh_dict = {}


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


def correspondenceSave(mesh, output_dir, img_size, coords_gt, Ks, Ts):
    output_dir = Path(output_dir)
    (output_dir / "correspondence").mkdir(exist_ok=True)
    Ks = Ks[:, :3, :3]
    coords_gt_ = coords_gt
    pixels_ = []
    coords_ = []
    index_gt = [
        img_name.split(".")[0]
        for img_name in os.listdir(str(output_dir / "color"))
        if img_name.endswith(".png")
    ]
    index_gt.sort(key=lambda x: int(x.split("_frame_")[1].split("_domain_")[0]))
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
        np.save(
            output_dir / "correspondence" / "{}.npy".format(index_gt[i]),
            correspondences,
        )


def parse_room(
    room,
    angle,
    rgb_gap,
    raw_root,
    dataset_root,
    output_root,
    align_angle=True,
    parse_normal=False,
    parse_pointclouds=False,
    parse_depths=False,
):
    print("Parsing: {}".format(room))
    classes = [
        "ceiling",
        "floor",
        "wall",
        "beam",
        "column",
        "window",
        "door",
        "table",
        "chair",
        "sofa",
        "bookcase",
        "board",
        "clutter",
    ]
    class2label = {cls: i for i, cls in enumerate(classes)}
    source_dir = os.path.join(dataset_root, room)
    output_root_pointclouds = output_root
    output_root_imgs = os.path.join(output_root, "images")
    os.makedirs(output_root_imgs, exist_ok=True)
    save_path_imgs = os.path.join(output_root_imgs, room)
    os.makedirs(save_path_imgs, exist_ok=True)
    object_path_list = sorted(glob.glob(os.path.join(source_dir, "Annotations/*.txt")))

    room_coords = []
    room_colors = []
    room_normals = []
    room_semantic_gt = []
    room_instance_gt = []

    for object_id, object_path in enumerate(object_path_list):
        object_name = os.path.basename(object_path).split("_")[0]
        obj = np.loadtxt(object_path)
        coords = obj[:, :3]
        colors = obj[:, 3:6]
        # note: in some room there is 'stairs' class
        class_name = object_name if object_name in classes else "clutter"
        semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
        semantic_gt = semantic_gt.reshape([-1, 1])
        instance_gt = np.repeat(object_id, coords.shape[0])
        instance_gt = instance_gt.reshape([-1, 1])

        room_coords.append(coords)
        room_colors.append(colors)
        room_semantic_gt.append(semantic_gt)
        room_instance_gt.append(instance_gt)

    room_coords = np.ascontiguousarray(np.vstack(room_coords))
    room_colors = np.ascontiguousarray(np.vstack(room_colors))

    if parse_normal:
        x_min, z_max, y_min = np.min(room_coords, axis=0)
        x_max, z_min, y_max = np.max(room_coords, axis=0)
        z_max = -z_max
        z_min = -z_min
        max_bound = np.array([x_max, y_max, z_max]) + 0.1
        min_bound = np.array([x_min, y_min, z_min]) - 0.1
        bbox = open3d.geometry.AxisAlignedBoundingBox(
            min_bound=min_bound, max_bound=max_bound
        )
        # crop room
        room_mesh = (
            area_mesh_dict[os.path.dirname(room)]
            .crop(bbox)
            .transform(
                np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            )
        )
        vertices = np.array(room_mesh.vertices)
        faces = np.array(room_mesh.triangles)
        vertex_normals = np.array(room_mesh.vertex_normals)
        o3d_room_mesh = room_mesh
        room_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=vertex_normals
        )
        (closest_points, distances, face_id) = room_mesh.nearest.on_surface(room_coords)
        room_normals = room_mesh.face_normals[face_id]
    Ks = []
    Ts = []

    area = room.split("/")[0].lower()
    part = room.split("/")[1]
    if area == "area_5":
        camera_pose_paths = sorted(
            glob.glob(
                os.path.join(raw_root, f"{area}*", "data", "pose", f"*_{part}_*.json")
            )
        )
        camera_rgb_paths = sorted(
            glob.glob(
                os.path.join(raw_root, f"{area}*", "data", "rgb", f"*_{part}_*.png")
            )
        )
        camera_depth_paths = sorted(
            glob.glob(
                os.path.join(raw_root, f"{area}*", "data", "depth", f"*_{part}_*.png")
            )
        )
    else:
        camera_pose_paths = sorted(
            glob.glob(os.path.join(raw_root, area, "data", "pose", f"*_{part}_*.json"))
        )
        camera_rgb_paths = sorted(
            glob.glob(os.path.join(raw_root, area, "data", "rgb", f"*_{part}_*.png"))
        )
        camera_depth_paths = sorted(
            glob.glob(os.path.join(raw_root, area, "data", "depth", f"*_{part}_*.png"))
        )
    camera_pose_names = set(
        [i.split(f"_{part}_")[0].split("camera_")[1] for i in camera_pose_paths]
    )
    camera_rgb_names = set(
        [i.split(f"_{part}_")[0].split("camera_")[1] for i in camera_rgb_paths]
    )
    camera_names = camera_pose_names.intersection(camera_rgb_names)
    for camera_name in camera_names:
        camera_pose_paths_withname = [
            path for path in camera_pose_paths if camera_name in path
        ]
        camera_rgb_paths_withname = [
            path for path in camera_rgb_paths if camera_name in path
        ]
        camera_depth_paths_withname = [
            path for path in camera_depth_paths if camera_name in path
        ]
        pose_frame_ids = set(
            [re.search(r"frame_(\d+)", s).group(1) for s in camera_pose_paths_withname]
        )
        rgb_frame_ids = set(
            [re.search(r"frame_(\d+)", s).group(1) for s in camera_rgb_paths_withname]
        )
        frame_ids = pose_frame_ids.intersection(rgb_frame_ids)
        frame_ids_selected = sorted(list(map(int, list(frame_ids))))[::rgb_gap]
        camera_pose_prefix = camera_pose_paths_withname[0].split("frame_")[0] + "frame_"
        camera_pose_postfix = (
            "_domain" + camera_pose_paths_withname[0].split("_domain")[1]
        )
        camera_pose_path_selected = [
            camera_pose_prefix + str(id) + camera_pose_postfix
            for id in frame_ids_selected
        ]
        camera_rgb_prefix = camera_rgb_paths_withname[0].split("frame_")[0] + "frame_"
        camera_rgb_postfix = (
            "_domain" + camera_rgb_paths_withname[0].split("_domain")[1]
        )
        camera_rgb_path_selected = [
            camera_rgb_prefix + str(id) + camera_rgb_postfix
            for id in frame_ids_selected
        ]
        camera_depth_prefix = (
            camera_depth_paths_withname[0].split("frame_")[0] + "frame_"
        )
        camera_depth_postfix = (
            "_domain" + camera_depth_paths_withname[0].split("_domain")[1]
        )
        camera_depth_path_selected = [
            camera_depth_prefix + str(id) + camera_depth_postfix
            for id in frame_ids_selected
        ]
        image_shape = Image.open(camera_rgb_path_selected[0]).size
        save_path_imgs_camera = os.path.join(save_path_imgs, camera_name)
        imgs_output_path = os.path.join(save_path_imgs_camera, "color")
        os.makedirs(imgs_output_path, exist_ok=True)
        for img_path in camera_rgb_path_selected:
            shutil.copy2(img_path, imgs_output_path)
        if parse_depths:
            depths_output_path = os.path.join(save_path_imgs_camera, "depth")
            os.makedirs(depths_output_path, exist_ok=True)
            for depth_path in camera_depth_path_selected:
                shutil.copy2(depth_path, depths_output_path)
        Ks = []
        Ts = []
        Ks_output_path = os.path.join(save_path_imgs_camera, "intrinsic")
        Ts_output_path = os.path.join(save_path_imgs_camera, "pose")
        os.makedirs(Ks_output_path, exist_ok=True)
        os.makedirs(Ts_output_path, exist_ok=True)
        for id, posepath in enumerate(camera_pose_path_selected):
            with open(posepath, "r") as f:
                data = json.load(f)
            k_matrix = np.array(data["camera_k_matrix"])
            rt_matrix = np.array(data["camera_rt_matrix"])
            T_matrix = np.eye(4)
            T_matrix[:3, :] = rt_matrix
            Ks.append(k_matrix)
            Ts.append(np.linalg.inv(T_matrix))
            np.save(os.path.join(Ks_output_path, f"{id}.npy"), k_matrix)
            np.save(os.path.join(Ts_output_path, f"{id}.npy"), T_matrix)
        Ks = np.array(Ks)
        Ts = np.array(Ts)
        correspondenceSave(
            o3d_room_mesh,
            save_path_imgs_camera,
            (image_shape[1], image_shape[0]),
            room_coords,
            Ks,
            Ts,
        )

    if align_angle:
        angle = (2 - angle / 180) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        room_center = (np.max(room_coords, axis=0) + np.min(room_coords, axis=0)) / 2
        room_coords = (room_coords - room_center) @ np.transpose(rot_t) + room_center
        if parse_normal:
            room_normals = room_normals @ np.transpose(rot_t)

    if parse_pointclouds:
        os.makedirs(output_root_pointclouds, exist_ok=True)
        save_path_pointclouds = os.path.join(output_root_pointclouds, room)
        os.makedirs(save_path_pointclouds, exist_ok=True)
        room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
        room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))
        np.save(
            os.path.join(save_path_pointclouds, "coord.npy"),
            room_coords.astype(np.float32),
        )
        np.save(
            os.path.join(save_path_pointclouds, "color.npy"),
            room_colors.astype(np.uint8),
        )
        np.save(
            os.path.join(save_path_pointclouds, "segment.npy"),
            room_semantic_gt.astype(np.int16),
        )
        np.save(
            os.path.join(save_path_pointclouds, "instance.npy"),
            room_instance_gt.astype(np.int16),
        )

        if parse_normal:
            np.save(
                os.path.join(save_path_pointclouds, "normal.npy"),
                room_normals.astype(np.float32),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        choices=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        help="Splits need to process ([Area_1, Area_2, Area_3, Area_4, Area_5, Area_6]).",
    )
    parser.add_argument(
        "--dataset_root", required=True, help="Path to Stanford3dDataset_v1.2 dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--pointclouds_root",
        default="data/s3dis",
        type=str,
        help="Input path where previous pointclouds folder located",
    )
    parser.add_argument(
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--rgb_gap",
        default=5,
        help="gap of rgb (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    parser.add_argument(
        "--parse_pointclouds", action="store_true", help="Whether parse point clouds"
    )
    parser.add_argument(
        "--parse_depths", action="store_true", help="Whether parse depths"
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
    )
    parser.add_argument(
        "--thread_id",
        default=0,
        type=int,
        help="Thread id for parallel processing",
    )
    args = parser.parse_args()

    if args.parse_normal:
        assert args.raw_root is not None

    room_list = []
    angle_list = []

    # Load room information
    print("Loading room information ...")
    for split in args.splits:
        area_info = np.loadtxt(
            os.path.join(
                args.dataset_root,
                split,
                f"{split}_alignmentAngle.txt",
            ),
            dtype=str,
        )
        room_list += [os.path.join(split, room_info[0]) for room_info in area_info]
        angle_list += [int(room_info[1]) for room_info in area_info]

    if args.parse_normal:
        # load raw mesh file to extract normal
        print("Loading raw mesh file ...")
        for split in args.splits:
            if split != "Area_5":
                mesh_dir = os.path.join(args.raw_root, split.lower(), "3d", "rgb.obj")
                mesh = open3d.io.read_triangle_mesh(mesh_dir)
                mesh.triangle_uvs.clear()
                # trimesh_mesh = trimesh.load(mesh_dir, process=False)
            else:
                mesh_a_dir = os.path.join(
                    args.raw_root, f"{split.lower()}a", "3d", "rgb.obj"
                )
                mesh_b_dir = os.path.join(
                    args.raw_root, f"{split.lower()}b", "3d", "rgb.obj"
                )
                mesh_a = open3d.io.read_triangle_mesh(mesh_a_dir)
                mesh_a.triangle_uvs.clear()
                # trimesh_mesh_a = trimesh.load(mesh_a_dir, process=False)
                mesh_b = open3d.io.read_triangle_mesh(mesh_b_dir)
                mesh_b.triangle_uvs.clear()
                # trimesh_mesh_b = trimesh.load(mesh_b_dir, process=False)
                mesh_b = mesh_b.transform(
                    np.array(
                        [
                            [0, 0, -1, -4.09703582],
                            [0, 1, 0, 0],
                            [1, 0, 0, -6.22617759],
                            [0, 0, 0, 1],
                        ]
                    )
                )
                mesh = mesh_a + mesh_b
                # trimesh_mesh = trimesh_mesh_a +trimesh_mesh_b
            area_mesh_dict[split] = mesh
            # trimesh_area_mesh_dict[split] = trimesh_mesh
            print(f"{split} mesh is loaded")

    room_list_list = np.array_split(room_list, args.num_workers)
    room_list_ = room_list_list[args.thread_id]

    for i in range(len(room_list_)):
        parse_room(
            room_list_[i],
            angle_list[i],
            args.rgb_gap,
            args.raw_root,
            args.dataset_root,
            args.output_root,
            args.align_angle,
            args.parse_normal,
            args.parse_pointclouds,
            args.parse_depths,
        )
