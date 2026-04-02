import os
import json
import argparse


def get_splits_paths(dataset_path):
    # Get the names of all subfolders in the given folder
    pc_path = dataset_path
    splits = ["train"]
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    for split in splits:
        pc_split_path = os.path.join(pc_path, split)
        split_names = [f.name for f in os.scandir(pc_split_path) if f.is_dir()]
        pc_split_path = pc_split_path.replace(dataset_path, "data/hk")
        split_dict = {}
        for name in split_names:
            split_dict[f"{name}"] = {}
            split_dict[f"{name}"]["pointclouds"] = os.path.join(pc_split_path, name)
            split_dict[f"{name}"]["images"] = []
            split_dict[f"{name}"]["correspondences"] = []
        with open(os.path.join(split_path, f"{split}.json"), "w") as f:
            json.dump(split_dict, f, indent=4)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    config = parser.parse_args()
    get_splits_paths(config.dataset_root)
