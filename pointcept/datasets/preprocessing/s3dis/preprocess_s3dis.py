"""
Preprocessing Script for S3DIS
Parsing normal vectors has a large consumption of memory. Please reduce max_workers if memory is limited.

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import numpy as np

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

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

area_mesh_dict = {}


def parse_room(
    room, angle, dataset_root, output_root, align_angle=True, parse_normal=False
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
    save_path = os.path.join(output_root, room)
    os.makedirs(save_path, exist_ok=True)
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
        room_mesh = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_normals=vertex_normals
        )
        (closest_points, distances, face_id) = room_mesh.nearest.on_surface(room_coords)
        room_normals = room_mesh.face_normals[face_id]

    if align_angle:
        angle = (2 - angle / 180) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        room_center = (np.max(room_coords, axis=0) + np.min(room_coords, axis=0)) / 2
        room_coords = (room_coords - room_center) @ np.transpose(rot_t) + room_center
        if parse_normal:
            room_normals = room_normals @ np.transpose(rot_t)

    room_colors = np.ascontiguousarray(np.vstack(room_colors))
    room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
    room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))
    np.save(os.path.join(save_path, "coord.npy"), room_coords.astype(np.float32))
    np.save(os.path.join(save_path, "color.npy"), room_colors.astype(np.uint8))
    np.save(os.path.join(save_path, "segment.npy"), room_semantic_gt.astype(np.int16))
    np.save(os.path.join(save_path, "instance.npy"), room_instance_gt.astype(np.int16))

    if parse_normal:
        np.save(os.path.join(save_path, "normal.npy"), room_normals.astype(np.float32))


def main_process():
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
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
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
                mesh_dir = os.path.join(args.raw_root, split, "3d", "rgb.obj")
                mesh = open3d.io.read_triangle_mesh(mesh_dir)
                mesh.triangle_uvs.clear()
            else:
                mesh_a_dir = os.path.join(args.raw_root, f"{split}a", "3d", "rgb.obj")
                mesh_b_dir = os.path.join(args.raw_root, f"{split}b", "3d", "rgb.obj")
                mesh_a = open3d.io.read_triangle_mesh(mesh_a_dir)
                mesh_a.triangle_uvs.clear()
                mesh_b = open3d.io.read_triangle_mesh(mesh_b_dir)
                mesh_b.triangle_uvs.clear()
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
            area_mesh_dict[split] = mesh
            print(f"{split} mesh is loaded")

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(
        max_workers=args.num_workers
    )  # peak 110G memory when parsing normal.
    _ = list(
        pool.map(
            parse_room,
            room_list,
            angle_list,
            repeat(args.dataset_root),
            repeat(args.output_root),
            repeat(args.align_angle),
            repeat(args.parse_normal),
        )
    )


if __name__ == "__main__":
    main_process()
