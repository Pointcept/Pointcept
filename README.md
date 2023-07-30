<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/logo.png" width="400">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

**Pointcept** is a powerful and flexible codebase for point cloud perception research. It is also an official implementation of the following paper:
- **Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning**   
*Xiaoyang Wu, Xin Wen, Xihui Liu, Hengshuang Zhao*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2023  
[ Pretrain ] [ MSC ] - [ [arXiv](https://arxiv.org/abs/2303.14191) ] [ [Bib](https://xywu.me/research/msc/bib.txt) ] &rarr; [here](#masked-scene-contrast)


- **Understanding Imbalanced Semantic Segmentation Through Neural Collapse** (3D Part)  
*Zhisheng Zhong\*, Jiequan Cui\*, Yibo Yang\*, Xiaoyang Wu, Xiaojuan Qi, Xiangyu Zhang, Jiaya Jia*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2023  
[ SemSeg ] [ CeCo ] - [ [arXiv](https://arxiv.org/abs/2301.01100) ] [ [Bib](https://xywu.me/research/ceco/bib.txt) ] [ [2D Part](https://github.com/dvlab-research/Imbalanced-Learning) ] &rarr; soon


- **Learning Context-aware Classifier for Semantic Segmentation** (3D Part)  
*Zhuotao Tian, Jiequan Cui, Li Jiang, Xiaojuan Qi, Xin Lai, Yixin Chen, Shu Liu, Jiaya Jia*  
AAAI Conference on Artificial Intelligence (**AAAI**) 2023 - Oral  
[ SemSeg ] [ CAC ] - [ [arXiv](https://arxiv.org/abs/2303.11633) ] [ [Bib](https://xywu.me/research/cac/bib.txt) ] [ [2D Part](https://github.com/tianzhuotao/CAC) ] &rarr; [here](#context-aware-classifier)


- **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling**   
*Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao*  
Conference on Neural Information Processing Systems (**NeurIPS**) 2022  
[ Backbone ] [ PTv2 ] - [ [arXiv](https://arxiv.org/abs/2210.05666) ] [ [Bib](https://xywu.me/research/ptv2/bib.txt) ] &rarr; [here](#point-transformers)


- **Point Transformer**   
*Hengshuang Zhao, Li Jiang, Jiaya Jia, Philip Torr, Vladlen Koltun*  
IEEE International Conference on Computer Vision (**ICCV**) 2021 - Oral  
[ Backbone ] [ PTv1 ] - [ [arXiv](https://arxiv.org/abs/2012.09164) ] [ [Bib](https://hszhao.github.io/papers/iccv21_pointtransformer_bib.txt) ] &rarr; [here](#point-transformers)

Additionally, **Pointcept** integrates the following excellent work (contain above):  
Backbone: 
[MinkUNet](https://github.com/NVIDIA/MinkowskiEngine) ([here](#sparseunet)),
[SpUNet](https://github.com/traveller59/spconv) ([here](#sparseunet)),
[SPVCNN](https://github.com/mit-han-lab/spvnas) ([here](#spvcnn)),
[PTv1](https://arxiv.org/abs/2012.09164) ([here](#point-transformers)),
[PTv2](https://arxiv.org/abs/2210.05666) ([here](#point-transformers)),
[StratifiedFormer](https://github.com/dvlab-research/Stratified-Transformer) ([here](#stratified-transformer)),
[OctFormer](https://github.com/octree-nn/octformer) ([here](#octformer)),
[Swin3D](https://github.com/microsoft/Swin3D) ([here](#swin3d));   
Semantic Segmentation:
[Mix3d](https://github.com/kumuji/mix3d) ([here](https://github.com/Pointcept/Pointcept/blob/main/configs/scannet/semseg-spunet-v1m1-0-base.py#L5)),
[CAC](https://arxiv.org/abs/2303.11633) ([here](#context-aware-classifier));  
Instance Segmentation: 
[PointGroup](https://github.com/dvlab-research/PointGroup) ([here](#pointgroup));  
Pre-training: 
[PointContrast](https://github.com/facebookresearch/PointContrast) ([here](#pointcontrast)), 
[Contrastive Scene Contexts](https://github.com/facebookresearch/ContrastiveSceneContexts) ([here](#contrastive-scene-contexts)),
[Masked Scene Contrast](https://arxiv.org/abs/2303.14191) ([here](#masked-scene-contrast));  
Datasets:
[ScanNet](http://www.scan-net.org/) ([here](#scannet-v2)), 
[ScanNet200](http://www.scan-net.org/) ([here](#scannet-v2)),
[S3DIS](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1) ([here](#s3dis)),
[ArkitScene](https://github.com/apple/ARKitScenes),
[SemanticKITTI](http://www.semantic-kitti.org/) ([here](#semantickitti)),
[nuScenes](https://www.nuscenes.org/nuscenes) ([here](#nuscenes)),
[ModelNet40](https://modelnet.cs.princeton.edu/) ([here](#modelnet)),
[Structured3D](https://structured3d-dataset.org/) ([here](#structured3d)).


## Highlights
- *Jul, 2023*: **PTv2** also achieved fully reproducible SOTA performance on NuScenes (val mIoU 80.2% and test mIoU 82.6%) and Semantic-KITTI (val mIoU 70.3%) semantic segmentation.
- *Mar, 2023*: We released our codebase, **Pointcept**, a highly potent tool for point cloud representation learning and perception. We welcome new work to join the _Pointcept_ family and highly recommend reading [Quick Start](#quick-start) before starting your trail.
- *Feb, 2023*: **MSC** and **CeCo** accepted by CVPR 2023. _MSC_ is a highly efficient and effective pretraining framework that facilitates cross-dataset large-scale pretraining, while _CeCo_ is a segmentation method specifically designed for long-tail datasets. Both approaches are compatible with all existing backbone models in our codebase, and we will soon make the code available for public use.
- *Jan, 2023*: **CAC**, oral work of AAAI 2023, has expanded its 3D result with the incorporation of Pointcept. This addition will allow CAC to serve as a pluggable segmentor within our codebase.
- *Sep, 2022*: **PTv2** accepted by NeurIPS 2022. It is a continuation of the Point Transformer. The proposed GVA theory can apply to most existing attention mechanisms, while Grid Pooling is also a practical addition to existing pooling methods.

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 or higher
- CUDA: 11.3 or higher
- PyTorch: 1.10.0 or higher

### Conda Environment

```bash
conda create -n pointcept python=3.8 -y
conda activate pointcept
conda install ninja -y
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
conda install h5py pyyaml -c anaconda -y
conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# PTv1 & PTv2 or precise eval
cd libs/pointops
# usual
python setup.py install
# docker & multi GPU arch
TORCH_CUDA_ARCH_LIST="ARCH LIST" python  setup.py install
# e.g. 7.5: RTX 3000; 8.0: a100 More available in: https://developer.nvidia.com/cuda-gpus
TORCH_CUDA_ARCH_LIST="7.5 8.0" python  setup.py install
cd ../..

# Open3D (visualization, optional)
pip install open3d
```

## Data Preparation

### ScanNet v2

The preprocessing support semantic and instance segmentation for both `ScanNet20`, `ScanNet200` and `ScanNet Data Efficient`.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).
python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

- (Alternative) Our preprocess data can also be downloaded [[here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EREuB1If2DNEjz43-rdaVf4B5toMaIViXv8gEbxr9ydeYA?e=ffXeG4)], please agree the official license before download it.

- (Optional) Download ScanNet Data Efficient files:
```bash
# download-scannet.py is the official download script
# or follow instruction here: https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation#download
python download-scannet.py --data_efficient -o ${RAW_SCANNET_DIR}
# unzip downloads
cd ${RAW_SCANNET_DIR}/tasks
unzip limited-annotation-points.zip
unzip limited-bboxes.zip
unzip limited-reconstruction-scenes.zip
# copy files to processed dataset folder
cp -r ${RAW_SCANNET_DIR}/tasks ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset.
mkdir data
ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Run preprocessing code for S3DIS as follows:

```bash
# S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2 dataset.
# RAW_S3DIS_DIR: the directory of Stanford2d3dDataset_noXYZ dataset. (optional, for parsing normal)
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).

# S3DIS without aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
# S3DIS with aligned angle
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --align_angle
# S3DIS with normal vector (recommended, normal is helpful)
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --parse_normal
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR} --raw_root ${RAW_S3DIS_DIR} --align_angle --parse_normal
```

- (Alternative) Our preprocess data can also be downloaded [[here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/ERtd0QAyLGNMs6vsM4XnebcBseQ8YTL0UTrMmp11PmQF3g?e=MsER95
)] (with normal vector and aligned angle), please agree the official license before download it.

- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```
### Structured3D

- Download Structured3D panorama related and perspective (full) related zip files by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSc0qtvh4vHSoZaW6UvlXYy79MbcGdZfICjh4_t4bYofQIVIdw/viewform?pli=1) (no need to unzip them).
- Organize all downloaded zip file in one folder (`${STRUCT3D_DIR}`).
- Run preprocessing code for Structured3D as follows:
```bash
# STRUCT3D_DIR: the directory of downloaded Structured3D dataset.
# PROCESSED_STRUCT3D_DIR: the directory of processed Structured3D dataset (output dir).
# NUM_WORKERS: Number for workers for preprocessing, default same as cpu count (might OOM).
export PYTHONPATH=./
python pointcept/datasets/preprocessing/structured3d/preprocess_structured3d.py --dataset_root ${STRUCT3D_DIR} --output_root ${PROCESSED_STRUCT3D_DIR} --num_workers ${NUM_WORKERS} --grid_size 0.01 --fuse_prsp --fuse_pano
```
Following the instruction of [Swin3D](https://arxiv.org/abs/2304.06906), we keep 25 categories with frequencies of more than 0.001, out of the original 40 categories.

[//]: # (- &#40;Alternative&#41; Our preprocess data can also be downloaded [[here]&#40;&#41;], please agree the official license before download it.)

- (Alternative) Our preprocess data can also be downloaded [[here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EaTAxo4SvEJFnDVuv9bOol4B0GCI806BeI4G-TF9Rp7lZw?e=drU9FN
)] (with perspective views and panorama view, 471.7G after unzipping), please agree the official license before download it.

- Link processed dataset to codebase.
```bash
# PROCESSED_STRUCT3D_DIR: the directory of processed Structured3D dataset (output dir).
mkdir data
ln -s ${PROCESSED_STRUCT3D_DIR} ${CODEBASE_DIR}/data/structured3d
```

### SemanticKITTI
- Download [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
```bash
# SEMANTIC_KITTI_DIR: the directory of SemanticKITTI dataset.
# |- SEMANTIC_KITTI_DIR
#   |- dataset
#     |- sequences
#       |- 00
#       |- 01
#       |- ...

mkdir -p data
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```

### nuScenes
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
- (Alternative) Our preprocess nuScenes information data can also be downloaded [[here](
)], please agree the official license before download it.

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

- Link processed dataset to codebase.
```bash
# PROCESSED_NUSCENES_DIR: the directory of processed nuScenes dataset (output dir).
mkdir data
ln -s ${PROCESSED_NUSCENES_DIR} ${CODEBASE_DIR}/data/nuscenes
```

### ModelNet
- Download [modelnet40_normal_resampled.zip](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and unzip
- Link dataset to codebase.
```bash
mkdir -p data
ln -s ${MODELNET_DIR} ${CODEBASE_DIR}/data/modelnet40_normal_resampled
```

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. However, to obtain precise evaluation results, testing is **essential**. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance. (Currently only support testing on a single GPU, I might add support to multi-gpus testing in the future version.)

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-pt-v2m2-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base weight=exp/scannet/semseg-pt-v2m2-0-base/models/model_best.pth
```

The TTA can be disabled by replace `data.test.test_cfg.aug_transform = [...]` with:

```python
data = dict(
    train = dict(...),
    val = dict(...),
    test = dict(
        ...,
        test_cfg = dict(
            ...,
            aug_transform = [
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)]
            ]
        )
    )
)
```

### Offset
`Offset` is the separator of point clouds in batch data, and it is similar to the concept of `Batch` in PyG. 
A visual illustration of batch and offset is as follows:
<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/offset_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/offset.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/Pointcept/Pointcept/main/docs/offset.png" width="480">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

## Model Zoo
### 1. Backbones and Semantic Segmentation
#### SparseUNet

_Pointcept_ provides `SparseUNet` implemented by `SpConv` and `MinkowskiEngine`. The SpConv version is recommended since SpConv is easy to install and faster than MinkowskiEngine. Meanwhile, SpConv is also widely applied in outdoor perception.

- **SpConv (recommend)**

The SpConv version `SparseUNet` in the codebase was fully rewrite from `MinkowskiEngine` version, example running script is as follows:

```bash
# ScanNet val
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# S3DIS (with normal)
sh scripts/train.sh -g 4 -d s3dis -c semseg-spunet-v1m1-0-cn-base -n semseg-spunet-v1m1-0-cn-base
# SemanticKITTI
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# nuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# ModelNet40
sh scripts/train.sh -g 2 -d modelnet40 -c cls-spunet-v1m1-0-base -n cls-spunet-v1m1-0-base

# ScanNet Data Efficient
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-la20 -n semseg-spunet-v1m1-2-efficient-la20
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-la50 -n semseg-spunet-v1m1-2-efficient-la50
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-la100 -n semseg-spunet-v1m1-2-efficient-la100
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-la200 -n semseg-spunet-v1m1-2-efficient-la200
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-lr1 -n semseg-spunet-v1m1-2-efficient-lr1
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-lr5 -n semseg-spunet-v1m1-2-efficient-lr5
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-lr10 -n semseg-spunet-v1m1-2-efficient-lr10
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-2-efficient-lr20 -n semseg-spunet-v1m1-2-efficient-lr20

# Profile model run time
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet-v1m1-0-enable-profiler -n semseg-spunet-v1m1-0-enable-profiler
```

- **MinkowskiEngine**

The MinkowskiEngine version `SparseUNet` in the codebase was modified from original MinkowskiEngine repo, and example running script is as follows:
1. Install MinkowskiEngine, refer https://github.com/NVIDIA/MinkowskiEngine
2. Training with the following example scripts:
```bash
# Uncomment "# from .sparse_unet import *" in "pointcept/models/__init__.py"
# Uncomment "# from .mink_unet import *" in "pointcept/models/sparse_unet/__init__.py"
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# SemanticKITTI
sh scripts/train.sh -g 2 -d semantic_kitti -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
```

#### Point Transformers
- **PTv2 mode2 (recommend)**

The original PTv2 was trained on 4 * RTX a6000 (48G memory). Even enabling AMP, the memory cost of the original PTv2 is slightly larger than 24G. Considering GPUs with 24G memory are much more accessible, I tuned the PTv2 on the latest Pointcept and made it runnable on 4 * RTX 3090 machines.

`PTv2 Mode2` enables AMP and disables _Position Encoding Multiplier_ & _Grouped Linear_. During our further research, we found that precise coordinates are not necessary for point cloud understanding (Replacing precise coordinates with grid coordinates doesn't influence the performance. Also, SparseUNet is an example). As for Grouped Linear, my implementation of Grouped Linear seems to cost more memory than the Linear layer provided by PyTorch. Benefiting from the codebase and better parameter tuning, we also relieve the overfitting problem. The reproducing performance is even better than the results reported in our paper.

Example running script is as follows:

```bash
# ptv2m2: PTv2 mode2, disable PEM & Grouped Linear, GPU memory cost < 24G (recommend)
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-3-lovasz -n semseg-pt-v2m2-3-lovasz

# ScanNet test benchmark (train on train set and val set)
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-1-benchmark-submit -n semseg-pt-v2m2-1-benchmark-submit
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# SemanticKITTI
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# nuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
```

- **PTv2 mode1**

`PTv2 mode1` is the original PTv2 we reported in our paper, example running script is as follows:

```bash
# ptv2m1: PTv2 mode1, Original PTv2, GPU memory cost > 24G
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m1-0-base -n semseg-pt-v2m1-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v2m1-0-base -n semseg-pt-v2m1-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v2m1-0-base -n semseg-pt-v2m1-0-base
```

- **PTv1**

The original PTv1 is also available in our Pointcept codebase. I haven't run PTv1 for a long time, but I have ensured that the example running script works well. 

```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v1-0-base -n semseg-pt-v1-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v1-0-base -n semseg-pt-v1-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v1-0-base -n semseg-pt-v1-0-base
```


#### Stratified Transformer
1. Additional requirements:
```bash
pip install torch-points3d
# fix dependence, caused by install torch-points3d 
pip uninstall SharedArray
pip install SharedArray==3.2.1

cd libs/pointops2
python setup.py install
cd ../..
```
2. Uncomment `# from .stratified_transformer import *` in `pointcept/models/__init__.py`.
3. Refer [Optional Installation](installation) to install dependence.
4. Training with the following example scripts:
```bash
# stv1m1: Stratified Transformer mode1, Modified from the original Stratified Transformer code.
# PTv2m2: Stratified Transformer mode2, My rewrite version (recommend).

# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-st-v1m2-0-refined -n semseg-st-v1m2-0-refined
sh scripts/train.sh -g 4 -d scannet -c semseg-st-v1m1-0-origin -n semseg-st-v1m1-0-origin
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-st-v1m2-0-refined -n semseg-st-v1m2-0-refined
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-st-v1m2-0-refined -n semseg-st-v1m2-0-refined
```

#### SPVCNN
`SPVCNN` is baseline model of [SPVNAS](https://github.com/mit-han-lab/spvnas), it is also a practical baseline for outdoor dataset.
1. Install torchsparse:
```bash
# refer https://github.com/mit-han-lab/torchsparse
# install method without sudo apt install
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```
2. Training with the following example scripts:
```bash
# SemanticKITTI
sh scripts/train.sh -g 2 -d semantic_kitti -c semseg-spvcnn-v1m1-0-base -n semseg-spvcnn-v1m1-0-base
```

#### OctFormer
OctFormer from _OctFormer: Octree-based Transformers for 3D Point Clouds_.
1. Additional requirements:
```bash
cd libs
git clone https://github.com/octree-nn/dwconv.git
pip install ./dwconv
pip install ocnn
```
2. Uncomment `# from .octformer import *` in `pointcept/models/__init__.py`.
2. Training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-octformer-v1m1-0-base -n semseg-octformer-v1m1-0-base
```

#### Swin3D
Swin3D from _Swin3D: A Pretrained Transformer Backbone for 3D Indoor Scene Understanding_. 
1. Additional requirements:
```bash
# 1. Install MinkEngine v0.5.4, follow readme in https://github.com/NVIDIA/MinkowskiEngine;
# 2. Install Swin3D, mainly for cuda operation:
cd libs
git clone https://github.com/microsoft/Swin3D.git
cd Swin3D
pip install ./
```
2. Uncomment `# from .swin3d import *` in `pointcept/models/__init__.py`.
3. Pre-Training with the following example scripts (Structured3D preprocessing refer [here](#structured3d)):
```bash
# Structured3D + Swin-S
sh scripts/train.sh -g 4 -d structured3d -c semseg-swin3d-v1m1-0-small -n semseg-swin3d-v1m1-0-small
# Structured3D + Swin-L
sh scripts/train.sh -g 4 -d structured3d -c semseg-swin3d-v1m1-1-large -n semseg-swin3d-v1m1-1-large

# Addition
# Structured3D + SpUNet
sh scripts/train.sh -g 4 -d structured3d -c semseg-spunet-v1m1-0-base -n semseg-spunet-v1m1-0-base
# Structured3D + PTv2
sh scripts/train.sh -g 4 -d structured3d -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
```
4. Fine-tuning with the following example scripts:
```bash
# ScanNet + Swin-S
sh scripts/train.sh -g 4 -d scannet -w exp/structured3d/semseg-swin3d-v1m1-1-large/model/model_last.pth -c semseg-swin3d-v1m1-0-small -n semseg-swin3d-v1m1-0-small
# ScanNet + Swin-L
sh scripts/train.sh -g 4 -d scannet -w exp/structured3d/semseg-swin3d-v1m1-1-large/model/model_last.pth -c semseg-swin3d-v1m1-1-large -n semseg-swin3d-v1m1-1-large

# S3DIS + Swin-S (here we provide config support S3DIS normal vector)
sh scripts/train.sh -g 4 -d s3dis -w exp/structured3d/semseg-swin3d-v1m1-1-large/model/model_last.pth -c semseg-swin3d-v1m1-0-small -n semseg-swin3d-v1m1-0-small
# S3DIS + Swin-L (here we provide config support S3DIS normal vector)
sh scripts/train.sh -g 4 -d s3dis -w exp/structured3d/semseg-swin3d-v1m1-1-large/model/model_last.pth -c semseg-swin3d-v1m1-1-large -n semseg-swin3d-v1m1-1-large
```

#### Context-Aware Classifier
`Context-Aware Classifier` is a segmentor that can further boost the performance of each backbone, as a replacement for `Default Segmentor`.  Training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-cac-v1m1-0-spunet-base -n semseg-cac-v1m1-0-spunet-base
sh scripts/train.sh -g 4 -d scannet -c semseg-cac-v1m1-1-spunet-lovasz -n semseg-cac-v1m1-1-spunet-lovasz
sh scripts/train.sh -g 4 -d scannet -c semseg-cac-v1m1-2-ptv2-lovasz -n semseg-cac-v1m1-2-ptv2-lovasz

# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-cac-v1m1-0-spunet-base -n semseg-cac-v1m1-0-spunet-base
sh scripts/train.sh -g 4 -d scannet200 -c semseg-cac-v1m1-1-spunet-lovasz -n semseg-cac-v1m1-1-spunet-lovasz
sh scripts/train.sh -g 4 -d scannet200 -c semseg-cac-v1m1-2-ptv2-lovasz -n semseg-cac-v1m1-2-ptv2-lovasz
```


### 2. Instance Segmentation
#### PointGroup
[PointGroup](https://github.com/dvlab-research/PointGroup) is a baseline framework for point cloud instance segmentation.
1. Additional requirements:
```bash
conda install -c bioconda google-sparsehash 
cd libs/pointgroup_ops
python setup.py install --include_dirs=${CONDA_PREFIX}/include
cd ../..
```
2. Uncomment `# from .point_group import *` in `pointcept/models/__init__.py`.
3. Training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c insseg-pointgroup-v1m1-0-spunet-base -n insseg-pointgroup-v1m1-0-spunet-base
# S3DIS
sh scripts/train.sh -g 4 -d scannet -c insseg-pointgroup-v1m1-0-spunet-base -n insseg-pointgroup-v1m1-0-spunet-base
```

### 3. Pre-training
#### Masked Scene Contrast
1. Pre-training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 8 -d scannet -c pretrain-msc-v1m1-0-spunet-base -n pretrain-msc-v1m1-0-spunet-base
```

2. Fine-tuning with the following example scripts:  
enable PointGroup ([here](#pointgroup)) before fine-tuning on instance segmentation task.
```bash
# ScanNet20 Semantic Segmentation
sh scripts/train.sh -g 8 -d scannet -w exp/scannet/pretrain-msc-v1m1-0-spunet-base/model/model_last.pth -c semseg-spunet-v1m1-4-ft -n semseg-msc-v1m1-0f-spunet-base
# ScanNet20 Instance Segmentation (enable PointGroup before running the script)
sh scripts/train.sh -g 4 -d scannet -w exp/scannet/pretrain-msc-v1m1-0-spunet-base/model/model_last.pth -c insseg-pointgroup-v1m1-0-spunet-base -n insseg-msc-v1m1-0f-pointgroup-spunet-base
```
3. Example log and weight: [[Pretrain](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EYvNV4XUJ_5Mlk-g15RelN4BW_P8lVBfC_zhjC_BlBDARg?e=UoGFWH)] [[Semseg](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/wuxy_connect_hku_hk/EQkDiv5xkOFKgCpGiGtAlLwBon7i8W6my3TIbGVxuiTttQ?e=tQFnbr)]

#### PointContrast
1. Preprocess and link ScanNet-Pair dataset (pair-wise matching with ScanNet raw RGB-D frame, ~1.5T):
```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_PAIR_DIR: the directory of processed ScanNet pair dataset (output dir).
python pointcept/datasets/preprocessing/scannet/scannet_pair/preprocess.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_PAIR_DIR}
ln -s ${PROCESSED_SCANNET_PAIR_DIR} ${CODEBASE_DIR}/data/scannet
```
2. Pre-training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 8 -d scannet -c pretrain-msc-v1m1-1-spunet-pointcontrast -n pretrain-msc-v1m1-1-spunet-pointcontrast
```
3. Fine-tuning refer [MSC](#masked-scene-contrast).

#### Contrastive Scene Contexts
1. Preprocess and link ScanNet-Pair dataset (refer [PointContrast](#pointcontrast)):
2. Pre-training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 8 -d scannet -c pretrain-msc-v1m2-0-spunet-csc -n pretrain-msc-v1m2-0-spunet-csc
```
3. Fine-tuning refer [MSC](#masked-scene-contrast).

## Citation
If you find _Pointcept_ useful to your research, please cite our work:
```
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## Acknowledgement
_Pointcept_ is designed by [Xiaoyang](https://xywu.me/), named by [Yixing](https://github.com/yxlao) and the logo is created by [Yuechen](https://julianjuaner.github.io/). It is derived from [Hengshuang](https://hszhao.github.io/)'s [Semseg](https://github.com/hszhao/semseg) and inspirited by several repos, e.g., [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
