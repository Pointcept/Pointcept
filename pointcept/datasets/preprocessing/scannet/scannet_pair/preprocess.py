import os
import argparse
import glob
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from reader import reader
from point_cloud_extractor import extractor
from compute_full_overlapping import compute_full_overlapping


frame_skip = 25


def parse_sens(sens_dir, output_dir):
    scene_id = os.path.basename(os.path.dirname(sens_dir))
    print(f"Parsing sens data{sens_dir}")
    reader(sens_dir, os.path.join(output_dir, scene_id), frame_skip,
           export_color_images=True, export_depth_images=True, export_poses=True, export_intrinsics=True)
    extractor(os.path.join(output_dir, scene_id), os.path.join(output_dir, scene_id, "pcd"))
    compute_full_overlapping(output_dir, scene_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', required=True, help='Path to the ScanNet dataset containing scene folders')
    parser.add_argument('--output_root', required=True, help='Output path where train/val folders will be located')
    opt = parser.parse_args()
    sens_list = sorted(glob.glob(os.path.join(opt.dataset_root, "scans/scene*/*.sens")))
    # Preprocess data.
    pool = ProcessPoolExecutor(max_workers=mp.cpu_count())
    # pool = ProcessPoolExecutor(max_workers=1)
    print('Processing scenes...')
    _ = list(pool.map(parse_sens, sens_list, repeat(opt.output_root)))

    # sens_dir = "/home/gofinge/Documents/datasets/scannet/scans/scene0024_00/scene0024_00.sens"
    # output_dir = "/home/gofinge/Downloads"
    # parse_sens(sens_dir, output_dir)
