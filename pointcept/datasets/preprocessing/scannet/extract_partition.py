import json
import shutil
import argparse
import torch
import glob
import os.path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root",
        required=True,
        help="Path to the ScanNet dataset containing scene folders",
    )
    parser.add_argument(
        "--processed_root",
        required=True,
        help="Path to the processed ScanNet dataset, add partition to test data dict",
    )
    parser.add_argument(
        "--segmentor_root",
        required=True,
        help="Path to Felzenswalb and Huttenlocher's Graph Based Image Segmentation binary",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "val"],
        help="Split to process. [test / val]",
    )
    config = parser.parse_args()
    if config.split == "test":
        raw_split = "scans_test"
    else:
        raw_split = "scans"

    scene_list = glob.glob(os.path.join(config.processed_root, config.split, "*.pth"))
    os.makedirs(os.path.join(config.processed_root, "tmp"), exist_ok=True)

    for scene in scene_list:
        scene_name = os.path.basename(scene).split(".")[0]
        raw_scene = os.path.join(
            config.dataset_root,
            raw_split,
            scene_name,
            f"{scene_name}_vh_clean_2.ply",
        )
        tmp_scene = os.path.join(
            config.processed_root,
            "tmp",
            f"{scene_name}_vh_clean_2.ply",
        )
        # copy original scene to tmp folder
        shutil.copy(raw_scene, tmp_scene)
        # run segmentor
        process = os.popen(f"{config.segmentor_root} {tmp_scene}")
        print(process.read())
        process.close()
        # load partition file
        partition_file = tmp_scene.replace(".ply", ".0.010000.segs.json")
        with open(partition_file) as f:
            partition = json.load(f)["segIndices"]
        data_dict = torch.load(scene)
        data_dict["partition"] = partition
        torch.save(data_dict, scene)
        # clean tmp
        os.remove(partition_file)
        os.remove(tmp_scene)
        print(f"Adding partition information to {scene_name}")

    os.rmdir(os.path.join(config.processed_root, "tmp"))
