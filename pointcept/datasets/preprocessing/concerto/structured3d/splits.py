import os
import json
import argparse


def get_splits_paths(dataset_path):
    # Get the names of all subfolders in the given folder
    im_path = os.path.join(dataset_path, "images")
    pc_path = dataset_path
    splits = ["train", "val"]
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    for split in splits:
        im_split_path = os.path.join(im_path, split)
        pc_split_path = os.path.join(pc_path, split).replace(
            dataset_path, "data/structured3d"
        )
        split_names = [f.name for f in os.scandir(im_split_path) if f.is_dir()]
        split_dict = {}
        for name in split_names:
            cam_split_name_path = os.path.join(im_split_path, name)
            cam_names = [f.name for f in os.scandir(cam_split_name_path) if f.is_dir()]
            for cam_name in cam_names:
                im_split_name_path = os.path.join(
                    cam_split_name_path, cam_name, "color", "prsp"
                )
                co_split_name_path = os.path.join(
                    cam_split_name_path,
                    cam_name,
                    "correspondence",
                    "prsp_correspondence",
                )
                png_files = [
                    f for f in os.listdir(im_split_name_path) if f.endswith(".png")
                ]
                png_files = sorted(png_files, key=lambda x: int(x.split(".")[0]))
                # Get the full paths of the .png files
                png_file_paths = [
                    os.path.join(im_split_name_path, f).replace(
                        dataset_path, "data/structured3d"
                    )
                    for f in png_files
                ]
                co_file_paths = [
                    os.path.join(co_split_name_path, f.replace(".png", ".npy")).replace(
                        dataset_path, "data/structured3d"
                    )
                    for f in png_files
                ]
                split_dict[f"{name}_{cam_name}"] = {}
                split_dict[f"{name}_{cam_name}"]["pointclouds"] = os.path.join(
                    pc_split_path, name, cam_name
                )
                split_dict[f"{name}_{cam_name}"]["images"] = png_file_paths
                split_dict[f"{name}_{cam_name}"]["correspondences"] = co_file_paths
        with open(os.path.join(split_path, f"{split}.json"), "w") as f:
            json.dump(split_dict, f, indent=4)


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the Structured3D dataset containing scene folders",
    )
    config = parser.parse_args()
    get_splits_paths(config.dataset_root)
