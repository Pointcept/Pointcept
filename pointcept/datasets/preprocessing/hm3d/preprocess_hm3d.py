"""
Preprocessing Script for Habitat-Matterport 3D Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import argparse
import numpy as np
import trimesh
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

from hm3d_constants import CLASS_LABELS_40

MPCAT2INDEX = dict([(CLASS_LABELS_40[i], i) for i in range(40)])
MPCAT2INDEX["unlabeled"] = -1

CAT2INDEX = dict()
with open(Path(__file__).parent / "hm3dsem_category_mappings.tsv") as f:
    f.readline()  # raw_category	category	mpcat40
    lines = f.readlines()
for line in lines:
    line = line.strip().split("\t")
    if len(line) == 2:
        # L2164: "\tunknown\tunlabeled"
        line.insert(0, "")
    CAT2INDEX[line[1]] = MPCAT2INDEX[line[2]]


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


def handle_process(
    scene_path,
    output_root,
    density=0.02,
):
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
    for name, mesh in scene.geometry.items():
        room_id = "_".join(name.split("_")[1:3]).replace("group", "").replace("sub", "")
        if room_id not in room_dict.keys():
            room_dict[room_id] = trimesh.Scene()
        room_dict[room_id].add_geometry(mesh, geom_name=name.split("_")[0])
    del scene

    for room_id, scene in room_dict.items():
        # seed by scene_id and room_id e.g. 00802-000-002 -> 802000002
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
        save_path = Path(output_root) / split / "_".join([scene_name, room_id])
        os.makedirs(save_path, exist_ok=True)
        for key, value in data_dict.items():
            np.save(save_path / f"{key}.npy", value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Habitat-Matterport 3D dataset containing scene folders",
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where train/val folders will be located",
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
    args = parser.parse_args()

    scene_list = glob.glob(os.path.join(args.dataset_root, "*", "*.glb"))
    scene_list = [scene for scene in scene_list if not scene.endswith("semantic.glb")]
    assert len(scene_list) == 900

    # Preprocess data.
    print("Processing scenes...")
    pool = ProcessPoolExecutor(max_workers=args.num_workers)
    _ = list(
        pool.map(
            handle_process,
            scene_list,
            repeat(args.output_root),
            repeat(args.density),
        )
    )
