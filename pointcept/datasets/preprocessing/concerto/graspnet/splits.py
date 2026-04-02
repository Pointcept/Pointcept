# GraspNet : train: 0-99, test_seen: 100-189
# train: scene_0000 to scene_0100 (100 scenes)
# val:   scene_0100 to scene_0189 (90 scenes)
import os
import json
import argparse
from tqdm import tqdm


def generate_graspnet_splits(dataset_path, group_num=4):
    """
    Generates train/val/test JSON split files for the GraspNet dataset.
    """
    scenes_dir = os.path.join(dataset_path, "scenes")
    all_scenes = sorted(
        [f for f in os.scandir(scenes_dir) if f.is_dir()], key=lambda f: f.name
    )
    all_scene_names = [f.name for f in all_scenes]

    split_definitions = {
        "train": [s for s in all_scene_names if 100 > int(s.split("_")[1]) >= 0],
        "val": [s for s in all_scene_names if int(s.split("_")[1]) >= 100],
    }

    output_split_path = os.path.join(dataset_path, "splits")
    os.makedirs(output_split_path, exist_ok=True)
    print(f"JSON files will be saved in: {output_split_path}")

    for split, scene_names in split_definitions.items():
        print(f"Processing '{split}' split with {len(scene_names)} scenes...")
        split_dict = {}
        for scene_name in tqdm(scene_names, desc=f"Generating {split}.json"):
            kinect_path = os.path.join(scenes_dir, scene_name, "kinect")

            rgb_path = os.path.join(kinect_path, "rgb")
            depth_path = os.path.join(kinect_path, "depth")
            pose_path = os.path.join(kinect_path, "pose")
            K_file_path_abs = os.path.join(kinect_path, "camK.npy")
            K_file_path_rel = K_file_path_abs.replace(dataset_path, "data/graspnet")

            rgb_files = [f for f in os.listdir(rgb_path) if f.endswith(".png")]
            rgb_files = sorted(rgb_files, key=lambda x: int(os.path.splitext(x)[0]))

            rgb_file_paths = [
                os.path.join(rgb_path, f).replace(dataset_path, "data/graspnet")
                for f in rgb_files
            ]
            depth_file_paths = [
                os.path.join(depth_path, f).replace(dataset_path, "data/graspnet")
                for f in rgb_files
            ]
            pose_file_paths = [
                os.path.join(pose_path, f.replace(".png", ".npy")).replace(
                    dataset_path, "data/graspnet"
                )
                for f in rgb_files
            ]

            for i in range(0, len(rgb_file_paths), group_num):
                if i + 4 > len(rgb_file_paths):
                    continue

                chunk_key = f"{scene_name}_{i//group_num}"
                split_dict[chunk_key] = {
                    "images": rgb_file_paths[i : i + group_num],
                    "depths": depth_file_paths[i : i + group_num],
                    "Ts": pose_file_paths[i : i + group_num],
                    "Ks": [K_file_path_rel] * group_num,
                }

        output_json_file = os.path.join(output_split_path, f"{split}.json")
        with open(output_json_file, "w") as f:
            json.dump(split_dict, f, indent=4)
        print(f"Successfully created {output_json_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the GraspNet dataset root (containing the 'scenes' folder).",
    )
    parser.add_argument(
        "--group_num",
        default=4,
        type=int,
        help="Group Num.",
    )
    config = parser.parse_args()
    generate_graspnet_splits(config.dataset_root, config.group_num)
