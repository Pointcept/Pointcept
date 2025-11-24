# Concerto
This repo is the official training codebase of the paper **_Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations_**. Concerto steps farther from [Sonata](https://github.com/facebookresearch/sonata), leveraging the synergy of 2D images and 3D point clouds and emerging superior representations.

*We recommend beginning with our inference [demo](https://github.com/Pointcept/Concerto)(not released currently), and it is also a tiny library for people who want to integrate pre-trained Concerto (as well as Sonata) into their project. This section is designed for users interested in reproducing our pre-training.*


<div align='left'>
<img src="https://raw.githubusercontent.com/pointcept/assets/main/concerto/teaser.png" alt="teaser" width="800" />
</div>

## Quick start

#### 1. Installation
Follow the instructions [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation).

Additional Package:
```bash
pip install transformers==4.50.3
pip install peft
```

#### 2. Dataset
Besides the same point cloud datasets as Sonata, Concerto includes the RE10K video dataset in training data. Each dataset has image assets and their correspondence to point clouds.

The whole data is organized as follows:
```
└── Default Dataset
    ├── images
    │   ├── train
    │   │   ├── color
    │   │   ├── correspondence
    │   │   ├── depth (optional)
    │   │   ├── intrinsic
    │   │   └── pose
    │   ├── val
    │   └── test
    ├── splits
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── train
    ├── val
    └── test
```
The `color`, `correspondence`, `depth`, `intrinsic`, and `pose` folders contain RGB images, point cloud correspondences, depth maps, camera intrinsics, and camera poses, respectively. The `train.json`, `val.json`, and `test.json` files in the _splits_ folder provide the indices for the training, validation, and test splits.

For the _intrinsic_ folder, if only a single file is presented, it indicates that all images share the same intrinsic parameters. If multiple files exist, each corresponds to a different image. In the case of the HM3D dataset, where camera parameters are simulated, all images share identical intrinsics across the entire dataset.

##### 2.1 [Option A] Preprocess Dataset from Scratch
For those who wish to do data preprocessing locally, please follow the instructions below:
- If you do not have the point cloud datasets for Sonata, you can process the next step without any modification. If you have already downloaded the processed Sonata datasets below, you can remove "-c" in the next step to only output image assets.
    - ScanNet v2 - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet-v2) ]
    - ScanNet++ v2 - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet) ]
    - S3DIS - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#s3dis) ]
    - ArkitScenes - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#arkitscenes) ]
    - HM3D - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#habitat---matterport-3d-hm3d) ]
    - Structured3D - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#structured3d) ]

- Now you can follow the instructions below to processe the image assets. 
    - ScanNet v2
        - Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
        - Run preprocessing code for raw ScanNet as follows:
        ```bash
        # RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset. PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/scannet/preprocess_scannet.sh -d ${RAW_SCANNET_DIR} -o ${PROCESSED_SCANNET_DIR} -n ${NUM_WORKERS} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/scannet/splits.py --dataset_root ${PROCESSED_SCANNET_DIR}
        ```
    - ScanNet++
        - Download the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset.
        - Run preprocessing code for raw ScanNet++ as follows:
        ```bash
        # For Concerto, we use the unsplitted point cloud data. There is no need to split the large scene because we crop the scenes in code according to the selected camera views.
        # RAW_SCANNETPP_DIR: the directory of downloaded ScanNet++ raw dataset. PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/scannetpp/preprocess_scannetpp.sh -d ${RAW_SCANNETPP_DIR} -o ${PROCESSED_SCANNETPP_DIR} -n ${NUM_WORKERS} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/scannetpp/splits.py --dataset_root ${PROCESSED_SCANNETPP_DIR}
        ```
    - S3DIS
        - Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
        - Fix error in `Area_5/office_19/Annotations/ceiling` Line 323474 (103.0�0000 => 103.000000).
        - (Optional) Download Full 2D-3D S3DIS dataset (no XYZ) from [here](https://github.com/alexsax/2D-3D-Semantics) for parsing normal.
        - Run preprocessing code for S3DIS as follows:
        ```bash
        # Default not use --align_angle
        # S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset. RAW_S3DIS_DIR: the directory of the Stanford2d3dDataset_noXYZ dataset. PROCESSED_S3DIS_DIR: the directory of the processed S3DIS dataset (output dir).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/s3dis/preprocess_s3dis.sh -d ${S3DIS_DIR} -r ${RAW_S3DIS_DIR} -o ${PROCESSED_S3DIS_DIR} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/s3dis/splits.py --dataset_root ${PROCESSED_S3DIS_DIR}
        ```
    - ARKitScenes
        - Download ArkitScenes 3DOD split with the following commands:
        ```bash
        # RAW_AS_DIR: the directory of downloaded Raw ArkitScenes dataset.
        git clone https://github.com/apple/ARKitScenes.git
        cd ARKitScenes
        python download_data.py 3dod --download_dir $RAW_AS_DIR --video_id_csv threedod/3dod_train_val_splits.csv
        ```
        - Run preprocessing code for ArkitScenes as follows:
        ```bash
        # RAW_AS_DIR: the directory of the downloaded ARKitScenes dataset. PROCESSED_AS_DIR: the directory of processed ArkitScenes dataset (output dir).
        # NUM_WORKERS: Number of workers for preprocessing, default same as CPU count (might OOM).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/arkitscenes/preprocess_arkitscenes.sh -d ${RAW_AS_DIR} -o ${PROCESSED_AS_DIR} -n ${NUM_WORKERS} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/arkitscenes/splits.py --dataset_root ${PROCESSED_ARKITSCENES_DIR}
        ```
    - Habitat - Matterport 3D (HM3D)
        - Download HM3D `hm3d-train-glb-v0.2.tar` and `hm3d-val-glb-v0.2.tar` with instuction [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) and unzip them.
        - Run preprocessing code for HM3D as follows:
        ```bash
        # We leverage the Habitat-Sim to simulate the camera views. The detailed installation for Habitat-Sim can be found at Habitat-Sim(https://github.com/facebookresearch/habitat-sim)
        # RAW_HM_DIR: the directory of downloaded HM3D dataset. PROCESSED_HM_DIR: the directory of processed HM3D dataset (output dir).
        # NUM_WORKERS: Number of workers for preprocessing, default same as CPU count (might OOM).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/hm3d/preprocess_hm3d.sh -d ${RAW_HM3D_DIR} -o ${PROCESSED_HM3D_DIR} -n ${NUM_WORKERS} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/hm3d/splits.py --dataset_root ${PROCESSED_HM3D_DIR}
        ```
    - Structured3D
        - Download Structured3D panorama related and perspective (full) related zip files by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSc0qtvh4vHSoZaW6UvlXYy79MbcGdZfICjh4_t4bYofQIVIdw/viewform?pli=1) (no need to unzip them).
        - Organize all downloaded zip file in one folder (`${STRUCT3D_DIR}`).
        - Run preprocessing code for Structured3D as follows:
        ```bash
        # RAW_STRUCT3D_DIR: the directory of downloaded Structured3D dataset. PROCESSED_STRUCT3D_DIR: the directory of processed Structured3D dataset (output dir).
        # NUM_WORKERS: Number of workers for preprocessing, default same as CPU count (might OOM).
        # Use -p to parse depths of the selected images.
        bash pointcept/datasets/preprocessing/concerto/structured3d/preprocess_structured3d.sh -d ${RAW_STRUCT3D_DIR} -o ${PROCESSED_STRUCT3D_DIR} -n ${NUM_WORKERS} -c
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/structured3d/splits.py --dataset_root ${PROCESSED_STRUCT3D_DIR}
        ```
    - RealEstate10K (RE10K)
        Additionally, we leverage video lifted RE10K by [VGGT](https://github.com/facebookresearch/vggt). Currently, the python file below only supports single data inference through VGGT. However, it is OK for you to start several python processes by setting num_workers and thread_id.
        - Install VGGT as a package
        ```bash
        git clone https://github.com/facebookresearch/vggt.git
        pip install -e .
        ```
        - Download RealEstate10K inPixelSplat format[ [Here](http://schadenfreude.csail.mit.edu:8000/)]
        - Preprocess the RE10K dataset
        ```bash
        # RAW_RE10K_DIR: the directory of downloaded RE10K dataset.
        # PROCESSED_RE10K_DIR: the directory of processed RE10K dataset (output dir).
        # using --parse_depths in the shell script to parse depths of the selected images.
        python pointcept/datasets/preprocessing/concerto/re10k/preprocess_re10k.py --dataset_root ${RAW_RE10K_DIR} --output_root ${PROCESSED_RE10K_DIR} --num_workers ${NUM_WORKERS} --thread_id ${THREAD_ID}
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/re10k/splits.py --dataset_root ${PROCESSED_RE10K_DIR}
        ```
    - SemanticKITTI
        - Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
        - SemanticKitti does not need any preprocessing.
        ```bash
        # SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
        # |- SEMANTIC_KITTI_DIR
        #   |- dataset
        #     |- sequences
        #       |- 00
        #       |- 01
        #       |- ...
        ```

    - nuScenes
        - Download the official [NuScene](https://www.nuscenes.org/nuscenes#download) dataset (with Lidar Segmentation) and organize the downloaded files as follows:
        ```bash
        NUSCENES_DIR
        │── samples
        │── sweeps
        │── lidarseg
        ...
        │── v1.0-trainval 
        │── v1.0-test
        ```
        - Run information preprocessing code (modified from OpenPCDet) for nuScenes as follows:
        ```bash
        # NUSCENES_DIR: the directory of downloaded nuScenes dataset.
        # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
        # MAX_SWEEPS: Max number of sweeps. Default: 10.
        pip install nuscenes-devkit pyquaternion
        python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py --dataset_root ${NUSCENES_DIR} --output_root ${PROCESSED_NUSCENES_DIR} --max_sweeps ${MAX_SWEEPS} --with_camera
        ```

        - Link raw dataset to processed NuScene dataset folder:
        ```bash
        # NUSCENES_DIR: the directory of downloaded nuScenes dataset.
        # PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
        ln -s ${NUSCENES_DIR} {PROCESSED_NUSCENES_DIR}/raw
        ```
        then the processed nuscenes folder is organized as follows:
        ```bash
        nuscene
        |── raw
            │── samples
            │── sweeps
            │── lidarseg
            ...
            │── v1.0-trainval
            │── v1.0-test
        |── info
        ```
    - Waymo
        - Download the official [Waymo](https://waymo.com/open/download/) dataset (v1.4.3) and organize the downloaded files as follows:
        ```bash
        WAYMO_RAW_DIR
        │── training
        │── validation
        │── testing
        ```
        - Install the following dependence:
        ```bash
        # If shows "No matching distribution found", download whl directly from Pypi and install the package.
        conda create -n waymo python=3.10 -y
        conda activate waymo
        pip install waymo-open-dataset-tf-2-12-0
        ```
        - Run the preprocessing code as follows:
        ```bash
        # WAYMO_DIR: the directory of the downloaded Waymo dataset.
        # PROCESSED_WAYMO_DIR: the directory of the processed Waymo dataset (output dir).
        # NUM_WORKERS: num workers for preprocessing
        python pointcept/datasets/preprocessing/waymo/preprocess_waymo.py --dataset_root ${WAYMO_DIR} --output_root ${PROCESSED_WAYMO_DIR} --splits training validation --num_workers ${NUM_WORKERS}
        # To generate the JSON file for the Concerto dataloader
        python pointcept/datasets/preprocessing/concerto/waymo/splits.py --dataset_root ${PROCESSED_WAYMO_DIR}
        ```

    - After all the preprocessing, link processed dataset above to the codebase.
    ```bash
    # PROCESSED_DIR: the directory of the processed dataset (output dir).
    # DATASET_NAME: the dataset name, which should be consistent with 'data_root' in the corresponding config, such as scannet, scannetpp, s3dis, arkitscenes, hm3d, structured3d, re10k, semantic_kitti, nuscenes, waymo
    mkdir data
    ln -s ${PROCESSED_DIR} ${CODEBASE_DIR}/data/${DATASET_NAME}
    ```

##### 2.2 [Option B] Download Preprocessed Dataset
You can download the preprocessed datasets from Huggingface. The downloaded datasets should be put in the folders named after the ${DATASET_NAME}, which is the string in front of .tar.gz.
    
- ScanNet v2 - [ [here](https://huggingface.co/datasets/Pointcept/concerto_scannet_compressed) ]
- ScanNet++ v2 - [ [here](https://huggingface.co/datasets/Pointcept/concerto_scannetpp_compressed) ]
- S3DIS - [ [here](https://huggingface.co/datasets/Pointcept/concerto_s3dis_compressed) ]
- ArkitScenes - [ [here](https://huggingface.co/datasets/Pointcept/concerto_arkitscenes_compressed) ]
- HM3D - [ [here](https://huggingface.co/datasets/Pointcept/concerto_hm3d_compressed) ]
- Structured3D - [ [here](https://huggingface.co/datasets/Pointcept/concerto_structured3d_compressed) ]
- RE10K - [ [here](https://huggingface.co/datasets/Pointcept/concerto_re10k_compressed) ]

Then you need to decompress the tar.gz using:
```bash
cat ${DATASET_NAME}/${DATASET_NAME}.tar.gz.* | tar -xzvf -
```
After decompressing, link processed dataset above to the codebase.
```bash
# PROCESSED_DIR: the directory of the processed dataset (output dir).
# DATASET_NAME: the dataset name, which should be consistent with 'data_root' in the corresponding config, such as scannet, scannetpp, s3dis, arkitscenes, hm3d, structured3d, re10k, semantic_kitti, nuscenes, waymo
mkdir data
ln -s ${PROCESSED_DIR} ${CODEBASE_DIR}/data/${DATASET_NAME}
```

#### 3. Pre-training
Enable pre-training with 32 GPUs (can be adjusted according to your hardware resources) by running the following script:
```bash
# default config
sh scripts/train.sh -m 4 -g 8 -d concerto -c pretrain-concerto-v1m1-0-base -n pretrain-concerto-v1m1-0-base

# (or) a large version PTv3 backbone
sh scripts/train.sh -m 4 -g 8 -d concerto -c pretrain-concerto-v1m1-1-large-base -n pretrain-concerto-v1m1-1-large-base

# (or) a large version PTv3 backbone with video-lifted RE10K data
sh scripts/train.sh -m 4 -g 8 -d concerto -c pretrain-concerto-v1m1-2-large-video -n pretrain-concerto-v1m1-2-large-video

# (or) if OOM
# half the batch size and half the base learning rate

# (or) if Nan appears
# try making grad clip lower as 2.0 or 1.0
# try making enable_amp=False
```

#### 4. Probing and Tuning
Our pre-trained model weight can be downloaded [here](https://huggingface.co/Pointcept/Concerto/blob/main/pretrain-concerto-v1m1-2-large-video.pth). Read [model card](https://huggingface.co/Pointcept/Concerto/blob/main/README.md) on huggingface for more information about different model weights. For `concerto_large/base/small/large_outdoor.pth` on [huggingface](https://huggingface.co/Pointcept/Concerto/tree/main), replace the lines below in the config to correctly load the model weight:
```bash
dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
```
to 
```bash
dict(
        type="CheckpointLoader",
        keywords="module",
        replacement="module.backbone",
    ),
```
Here are the example commands for probing and tuning on large model:
```bash
# Assume the pre-trained experiment is recorded in:
# exp/concerto/pretrain-concerto-v1m1-0-large-base

# ScanNet 
# linear probing
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0a-scannet-lin -n semseg-ptv3-large-v1m1-0a-scannet-lin -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# decoder probing
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0b-scannet-dec -n semseg-ptv3-large-v1m1-0b-scannet-dec -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# full fine-tuning
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0c-scannet-ft -n semseg-ptv3-large-v1m1-0c-scannet-ft -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# PPT multi-dataset joint fine-tuning (heavy, not necessary)
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0d-scannet-ppt -n semseg-ptv3-large-v1m1-0d-scannet-ppt -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# LoRA finetuning, need to change the backbone_path in config with your model that needs to be LoRA finetuned
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0f-scannet-ft-lora -n semseg-ptv3-large-v1m1-0f-scannet-ft-lora

# ScanNet Data Efficiency full fine-tuning
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e0-scannet-ft-la20 -n semseg-ptv3-large-v1m1-0e0-scannet-ft-la20 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e1-scannet-ft-la50 -n semseg-ptv3-large-v1m1-0e1-scannet-ft-la50 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e2-scannet-ft-la100 -n semseg-ptv3-large-v1m1-0e2-scannet-ft-la100 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e3-scannet-ft-la200 -n semseg-ptv3-large-v1m1-0e3-scannet-ft-la200 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e4-scannet-ft-lr1 -n semseg-ptv3-large-v1m1-0e4-scannet-ft-lr1 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e5-scannet-ft-lr5 -n semseg-ptv3-large-v1m1-0e5-scannet-ft-lr5 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e6-scannet-ft-lr10 -n semseg-ptv3-large-v1m1-0e6-scannet-ft-lr10 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e7-scannet-ft-lr20 -n semseg-ptv3-large-v1m1-0e7-scannet-ft-lr20 -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# LoRA finetuning, need to change the backbone_path in config with your model that needs to be LoRA finetuned
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e8-scannet-ft-lora-la20 -n semseg-ptv3-large-v1m1-0e8-scannet-ft-lora-la20
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e9-scannet-ft-lora-la50 -n semseg-ptv3-large-v1m1-0e9-scannet-ft-lora-la50
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e10-scannet-ft-lora-la100 -n semseg-ptv3-large-v1m1-0e10-scannet-ft-lora-la100
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e11-scannet-ft-lora-la200 -n semseg-ptv3-large-v1m1-0e11-scannet-ft-lora-la200
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e12-scannet-ft-lora-lr1 -n semseg-ptv3-large-v1m1-0e12-scannet-ft-lora-lr1
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e13-scannet-ft-lora-lr5 -n semseg-ptv3-large-v1m1-0e13-scannet-ft-lora-lr5
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e14-scannet-ft-lora-lr10 -n semseg-ptv3-large-v1m1-0e14-scannet-ft-lora-lr10
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-0e15-scannet-ft-lora-lr20 -n semseg-ptv3-large-v1m1-0e15-scannet-ft-lora-lr20

# ScanNet200
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-1a-scannet200-lin -n semseg-ptv3-large-v1m1-1a-scannet200-lin -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-1b-scannet200-dec -n semseg-ptv3-large-v1m1-1b-scannet200-dec -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-1c-scannet200-ft -n semseg-ptv3-large-v1m1-1c-scannet200-ft -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# LoRA finetuning, need to change the backbone_path in config with your model that needs to be LoRA finetuned
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-1e-scannet200-ft-lora -n semseg-ptv3-large-v1m1-1e-scannet200-ft-lora

# ScanNetpp
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-2a-scannetpp-lin -n semseg-ptv3-large-v1m1-2a-scannetpp-lin -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-2b-scannetpp-dec -n semseg-ptv3-large-v1m1-2b-scannetpp-dec -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-2c-scannetpp-ft -n semseg-ptv3-large-v1m1-2c-scannetpp-ft -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# LoRA finetuning, need to change the backbone_path in config with your model that needs to be LoRA finetuned
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-2g-scannetpp-ft-lora -n semseg-ptv3-large-v1m1-2g-scannetpp-ft-lora

# S3DIS Area 5
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-3a-s3dis-lin -n semseg-ptv3-large-v1m1-3a-s3dis-lin -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-3b-s3dis-dec -n semseg-ptv3-large-v1m1-3b-s3dis-dec -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-3c-s3dis-ft -n semseg-ptv3-large-v1m1-3c-s3dis-ft -w exp/concerto/pretrain-concerto-v1m1-0-large-base/model/model_last.pth
# LoRA finetuning, need to change the backbone_path in config with your model which need to be LoRA finetuned
sh scripts/train.sh -m 1 -g 8 -d concerto -c semseg-ptv3-large-v1m1-3e-s3dis-ft-lora -n semseg-ptv3-large-v1m1-3e-s3dis-ft-lora
```
The above configs have not been verified with one additional run. If you encounter any problems, please feel free to let me know. Also, comparing changes between each configuration would be helpful in handling the configuration system of Pointcept.

## Citation
If you find _Concerto_ useful to your research, please consider citing our line of works as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
```bib
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```

```bib
@inproceedings{wu2024ptv3,
    title={Point Transformer V3: Simpler, Faster, Stronger},
    author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2024ppt,
    title={Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training},
    author={Wu, Xiaoyang and Tian, Zhuotao and Wen, Xin and Peng, Bohao and Liu, Xihui and Yu, Kaicheng and Zhao, Hengshuang},
    booktitle={CVPR},
    year={2024}
}
```
```bib
@inproceedings{wu2023masked,
  title={Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning},
  author={Wu, Xiaoyang and Wen, Xin and Liu, Xihui and Zhao, Hengshuang},
  journal={CVPR},
  year={2023}
}
```
```bib
@inproceedings{wu2022ptv2,
    title={Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
    author={Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
    booktitle={NeurIPS},
    year={2022}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## License
- Concerto code is based on Sonata, which is released by Meta under the [Apache 2.0 license](https://github.com/facebookresearch/sonata/blob/main/LICENSE),
- Concerto weight is released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
(restricted by NC of datasets like HM3D, ArkitScenes).
- For commercial usage, please removing datasets restrict by non-commercial license.
