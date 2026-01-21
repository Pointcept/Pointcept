import os
import json
import argparse

CAM_TYPES = ["FRONT_LEFT", "FRONT_RIGHT", "FRONT", "SIDE_LEFT", "SIDE_RIGHT"]


def get_splits_paths(dataset_path):
    # Get the names of all subfolders in the given folder
    im_path = os.path.join(dataset_path, "images")
    pc_path = dataset_path
    splits = ["training", "validation"]
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    for split in splits:
        im_split_path = os.path.join(im_path, split)
        pc_split_path = os.path.join(pc_path, split).replace(dataset_path, "data/waymo")
        split_names = sorted([f.name for f in os.scandir(im_split_path) if f.is_dir()])
        split_dict = {}
        for name in split_names:
            im_split_subpaths = os.path.join(im_split_path, name)
            subsplit_names = sorted(
                [f.name for f in os.scandir(im_split_subpaths) if f.is_dir()]
            )
            for subname in subsplit_names:
                im_split_name_path = os.path.join(im_split_path, name, subname, "color")
                co_split_name_path = os.path.join(
                    im_split_path, name, subname, "correspondence"
                )
                jpg_files = [cam_type + ".jpg" for cam_type in CAM_TYPES]
                # Get the full paths of the .jpg files
                jpg_file_paths = [
                    os.path.join(im_split_name_path, f).replace(
                        dataset_path, "data/waymo"
                    )
                    for f in jpg_files
                ]
                co_file_paths = [
                    os.path.join(co_split_name_path, f.replace(".jpg", ".npy")).replace(
                        dataset_path, "data/waymo"
                    )
                    for f in jpg_files
                ]
                split_dict[f"{name}_{subname}"] = {}
                split_dict[f"{name}_{subname}"]["pointclouds"] = os.path.join(
                    pc_split_path, name, subname
                )
                split_dict[f"{name}_{subname}"]["images"] = jpg_file_paths
                split_dict[f"{name}_{subname}"]["correspondences"] = co_file_paths
        with open(os.path.join(split_path, f"{split}.json"), "w") as f:
            json.dump(split_dict, f, indent=4)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the waymo dataset containing scene folders",
    )
    config = parser.parse_args()
    get_splits_paths(config.dataset_root)
