import os
import json
import argparse


def get_splits_paths(dataset_path):
    # Get the names of all subfolders in the given folder
    co_path = os.path.join(dataset_path, "correspondences")
    im_path = os.path.join(dataset_path, "images")
    pc_path = dataset_path
    splits = ["train"]
    split_path = os.path.join(dataset_path, "splits")
    os.makedirs(split_path, exist_ok=True)
    for split in splits:
        co_split_path = os.path.join(co_path, split)
        im_split_path = os.path.join(im_path, split)
        pc_split_path = os.path.join(pc_path, split)
        split_names = [f.name for f in os.scandir(co_split_path) if f.is_dir()]
        split_dict = {}
        for name in split_names:
            print(f"processing {name}")
            co_name_split_path = os.path.join(co_split_path, name)
            img_names = [
                f.name.removesuffix(".npy")
                for f in os.scandir(co_name_split_path)
                if f.is_file() and f.name.endswith(".npy")
            ]
            img_names = sorted(img_names)
            png_file_paths = [
                os.path.join(im_split_path, name, img_name + ".png").replace(
                    dataset_path, "data/cap3d"
                )
                for img_name in img_names
            ]
            # Get the full paths of the .png files
            co_file_paths = [
                os.path.join(co_name_split_path, img_name + ".npy").replace(
                    dataset_path, "data/cap3d"
                )
                for img_name in img_names
            ]
            pc_file_path = os.path.join(pc_split_path, name + ".pt")
            file_check = True
            for png_file_path in png_file_paths:
                if not os.path.isfile(png_file_path):
                    file_check = False
            for co_file_path in co_file_paths:
                if not os.path.isfile(co_file_path):
                    file_check = False
            if not os.path.isfile(pc_file_path):
                file_check = False
                continue
            if file_check:
                split_dict[f"{name}"] = {}
                split_dict[f"{name}"]["pointclouds"] = pc_file_path
                split_dict[f"{name}"]["images"] = png_file_paths
                split_dict[f"{name}"]["correspondences"] = co_file_paths
            else:
                split_dict[f"{name}"] = {}
                split_dict[f"{name}"]["pointclouds"] = pc_file_path
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
