"""
Preprocessing ArkitScenes
"""
import os
import argparse
import glob
import plyfile
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

import torch


def read_plymesh(filepath):
    """Read ply file and return it as numpy array. Returns None if emtpy."""
    with open(filepath, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
    if plydata.elements:
        vertices = pd.DataFrame(plydata['vertex'].data).values
        faces = np.stack(plydata['face'].data['vertex_indices'], axis=0)
        return vertices, faces


def face_normal(vertex, face):
    v01 = vertex[face[:, 1]] - vertex[face[:, 0]]
    v02 = vertex[face[:, 2]] - vertex[face[:, 0]]
    vec = np.cross(v01, v02)
    length = np.sqrt(np.sum(vec ** 2, axis=1, keepdims=True)) + 1.0e-8
    nf = vec / length
    area = length * 0.5
    return nf, area


def vertex_normal(vertex, face):
    nf, area = face_normal(vertex, face)
    nf = nf * area

    nv = np.zeros_like(vertex)
    for i in range(face.shape[0]):
        nv[face[i]] += nf[i]

    length = np.sqrt(np.sum(nv ** 2, axis=1, keepdims=True)) + 1.0e-8
    nv = nv / length
    return nv


def parse_scene(scene_path, output_dir):
    print(f"Parsing scene {scene_path}")
    split = os.path.basename(os.path.dirname(os.path.dirname(scene_path)))
    scene_id = os.path.basename(os.path.dirname(scene_path))
    vertices, faces = read_plymesh(scene_path)
    coords = vertices[:, :3]
    colors = vertices[:, 3:6]
    data_dict = dict(coord=coords, color=colors, scene_id=scene_id)
    data_dict["normal"] = vertex_normal(coords, faces)
    torch.save(data_dict, os.path.join(output_dir, split, f"{scene_id}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help='Path to the ScanNet dataset containing scene folders')
    parser.add_argument('--output_root', required=True, help='Output path where train/val folders will be located')
    opt = parser.parse_args()
    # Create output directories
    train_output_dir = os.path.join(opt.output_root, 'Training')
    os.makedirs(train_output_dir, exist_ok=True)
    val_output_dir = os.path.join(opt.output_root, 'Validation')
    os.makedirs(val_output_dir, exist_ok=True)
    # Load scene paths
    scene_paths = sorted(glob.glob(opt.dataset_root + '/3dod/*/*/*_mesh.ply'))
    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    print('Processing scenes...')
    _ = list(pool.map(parse_scene, scene_paths, repeat(opt.output_root)))
