<p align="center">
    <!-- pypi-strip -->
    <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/logo.png" width="400">
    <!-- pypi-strip -->
    </picture><br>
    <!-- /pypi-strip -->
</p>

[![Formatter](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml/badge.svg)](https://github.com/pointcept/pointcept/actions/workflows/formatter.yml)

**Pointcept** is a powerful and flexible codebase for point cloud perception research. It is also an official implementation of the following paper:
- 🚀 **Concerto: Joint 2D-3D Self-Supervised Learning Emerges Spatial Representations**  
*Yujia Zhang, Xiaoyang Wu, Yixing Lao, Chengyao Wang, Zhuotao Tian, Naiyan Wang, Hengshuang Zhao*   
Conference on Neural Information Processing Systems (**NeurIPS**) 2025  
[ Pretrain ] [Concerto] - [ [Project](https://pointcept.github.io/Concerto/) ] [ [Bib](https://xywu.me/research/concerto/bib.txt) ] [ [HF Demo](https://huggingface.co/spaces/Pointcept/Concerto) ] [ [Inference](https://github.com/Pointcept/Concerto) ] [ [Weight](https://huggingface.co/Pointcept/Concerto) ] &rarr; [here](#concerto)


- **Sonata: Self-Supervised Learning of Reliable Point Representations**  
*Xiaoyang Wu, Daniel DeTone, Duncan Frost, Tianwei Shen, Chris Xie, Nan Yang, Jakob Engel, Richard Newcombe, Hengshuang Zhao, Julian Straub*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2025 - Highlight  
[ Pretrain ] [Sonata] - [ [Project](https://xywu.me/sonata/) ] [ [arXiv](https://arxiv.org/abs/2503.16429) ] [ [Bib](https://xywu.me/research/sonata/bib.txt) ] [ [Demo](https://github.com/facebookresearch/sonata) ] [ [Weight](https://huggingface.co/facebook/sonata) ] &rarr; [here](#sonata)


- **Point Transformer V3: Simpler, Faster, Stronger**  
*Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2024 - Oral  
[ Backbone ] [PTv3] - [ [arXiv](https://arxiv.org/abs/2312.10035) ] [ [Bib](https://xywu.me/research/ptv3/bib.txt) ] [ [Project](https://github.com/Pointcept/PointTransformerV3) ] &rarr; [here](https://github.com/Pointcept/PointTransformerV3)


- **OA-CNNs: Omni-Adaptive Sparse CNNs for 3D Semantic Segmentation**  
*Bohao Peng, Xiaoyang Wu, Li Jiang, Yukang Chen, Hengshuang Zhao, Zhuotao Tian, Jiaya Jia*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2024  
[ Backbone ] [ OA-CNNs ] - [ [arXiv](https://arxiv.org/abs/2403.14418) ] [ [Bib](https://xywu.me/research/oacnns/bib.txt) ] &rarr; [here](#oa-cnns)


- **Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training**  
*Xiaoyang Wu, Zhuotao Tian, Xin Wen, Bohao Peng, Xihui Liu, Kaicheng Yu, Hengshuang Zhao*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2024  
[ Pretrain ] [PPT] - [ [arXiv](https://arxiv.org/abs/2308.09718) ] [ [Bib](https://xywu.me/research/ppt/bib.txt) ] &rarr; [here](#point-prompt-training-ppt)


- **Masked Scene Contrast: A Scalable Framework for Unsupervised 3D Representation Learning**  
*Xiaoyang Wu, Xin Wen, Xihui Liu, Hengshuang Zhao*  
IEEE Conference on Computer Vision and Pattern Recognition (**CVPR**) 2023  
[ Pretrain ] [ MSC ] - [ [arXiv](https://arxiv.org/abs/2303.14191) ] [ [Bib](https://xywu.me/research/msc/bib.txt) ] &rarr; [here](#masked-scene-contrast-msc)


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
[OACNNs](https://arxiv.org/abs/2403.14418) ([here](#oa-cnns)),
[PTv1](https://arxiv.org/abs/2012.09164) ([here](#point-transformers)),
[PTv2](https://arxiv.org/abs/2210.05666) ([here](#point-transformers)),
[PTv3](https://arxiv.org/abs/2312.10035) ([here](#point-transformers)),
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
[Masked Scene Contrast](https://arxiv.org/abs/2303.14191) ([here](#masked-scene-contrast-msc)),
[Point Prompt Training](https://arxiv.org/abs/2308.09718) ([here](#point-prompt-training-ppt)),
[Sonata](https://arxiv.org/abs/2503.16429) ([here](#sonata)),
[Concerto]() ([here](#concerto));  
Datasets:
[ScanNet](http://www.scan-net.org/) ([here](#scannet-v2)), 
[ScanNet200](http://www.scan-net.org/) ([here](#scannet-v2)),
[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) ([here](#scannet)),
[S3DIS](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1) ([here](#s3dis)),
[ArkitScene](https://github.com/apple/ARKitScenes) ([here](#arkitscenes)),
[HM3D](https://github.com/facebookresearch/habitat-matterport3d-dataset/) ([here](#habitat---matterport-3d-hm3d)),
[Matterport3D](https://niessner.github.io/Matterport/) ([here](#matterport3d)),
[Structured3D](https://structured3d-dataset.org/) ([here](#structured3d)),
[SemanticKITTI](http://www.semantic-kitti.org/) ([here](#semantickitti)),
[nuScenes](https://www.nuscenes.org/nuscenes) ([here](#nuscenes)),
[ModelNet40](https://modelnet.cs.princeton.edu/) ([here](#modelnet)),
[Waymo](https://waymo.com/open/) ([here](#waymo)).


## Highlights
- *Apr 2025* 🚀: We now support `wandb`, check the [Quick Start](#quick-start) training section for more information. (Thanks @Streakfull for his contribution!)
- *Mar 2025* 🚀: **Sonata** is accepted by CVPR 2025 and selected as one of the **Highlight** presentations (3.0% submissions)! We release the code with Pointcept v1.6.0. We release the pre-training **[code](#sonata)** along with Pointcept v1.6.0 and provide an easy-to-use pre-trained model for inference, tuning, and visualization in our project **[repository](https://github.com/facebookresearch/sonata)** hosted by Meta.
- *May 2024*: In v1.5.2, we redesigned the default structure for each dataset for better performance. Please **re-preprocess** datasets or **download** our preprocessed datasets from **[here](https://huggingface.co/Pointcept)**.
- *Apr 2024*: **PTv3** is selected as one of the 90 **Oral** papers (3.3% accepted papers, 0.78% submissions) by CVPR'24!
- *Mar 2024*: We release code for **OA-CNNs**, accepted by CVPR'24. Issue related to **OA-CNNs** can @Pbihao.
- *Feb 2024*: **PTv3** and **PPT** are accepted by CVPR'24, another **two** papers by our Pointcept team have also been accepted by CVPR'24 🎉🎉🎉. We will make them publicly available soon!
- *Dec 2023*: **PTv3** is released on arXiv, and the code is available in Pointcept. PTv3 is an efficient backbone model that achieves SOTA performances across indoor and outdoor scenarios.
- *Aug 2023*: **PPT** is released on arXiv. PPT presents a multi-dataset pre-training framework that achieves SOTA performance in both **indoor** and **outdoor** scenarios. It is compatible with various existing pre-training frameworks and backbones.  A **pre-release** version of the code is accessible; for those interested, please feel free to contact me directly for access.
- *Mar 2023*: We released our codebase, **Pointcept**, a highly potent tool for point cloud representation learning and perception. We welcome new work to join the _Pointcept_ family and highly recommend reading [Quick Start](#quick-start) before starting your trail.
- *Feb 2023*: **MSC** and **CeCo** accepted by CVPR 2023. _MSC_ is a highly efficient and effective pretraining framework that facilitates cross-dataset large-scale pretraining, while _CeCo_ is a segmentation method specifically designed for long-tail datasets. Both approaches are compatible with all existing backbone models in our codebase, and we will soon make the code available for public use.
- *Jan 2023*: **CAC**, oral work of AAAI 2023, has expanded its 3D result with the incorporation of Pointcept. This addition will allow CAC to serve as a pluggable segmentor within our codebase.
- *Sep 2022*: **PTv2** accepted by NeurIPS 2022. It is a continuation of the Point Transformer. The proposed GVA theory can apply to most existing attention mechanisms, while Grid Pooling is also a practical addition to existing pooling methods.

## Citation
If you find _Pointcept_ useful to your research, please cite our work as encouragement. (੭ˊ꒳​ˋ)੭✧
```
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished = {\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 and above.
- CUDA: 11.3 and above.
- PyTorch: 1.10.0 and above.

### Conda Environment
- **Method 1**: Utilize conda `environment.yml` to create a new environment with one line code:
  ```bash
  # Create and activate conda environment named as 'pointcept-torch2.5.0-cu12.4'
  # cuda: 12.4, pytorch: 2.5.0

  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate pointcept-torch2.5.0-cu12.4
  ```

- **Method 2**: Use our pre-built Docker image and refer to the supported tags [here](https://hub.docker.com/r/pointcept/pointcept/tags). Quickly verify the Docker image on your local machine with the following command:
  ```bash
  docker run --gpus all -it --rm pointcept/pointcept:v1.6.0-pytorch2.5.0-cuda12.4-cudnn9-devel bash
  git clone https://github.com/facebookresearch/sonata
  cd sonata
  export PYTHONPATH=./ && python demo/0_pca.py
  # Ignore the GUI error, we cannot expect a container to have its GUI, right?
  ```

- **Method 3**: Manually create a conda environment:
  ```bash
  conda create -n pointcept python=3.10 -y
  conda activate pointcept
  
  # (Optional) If no CUDA installed
  conda install nvidia/label/cuda-12.4.1::cuda conda-forge::cudnn conda-forge::gcc=13.2 conda-forge::gxx=13.2 -y
  
  conda install ninja -y
  # Choose version you want here: https://pytorch.org/get-started/previous-versions/
  conda install pytorch==2.5.0 torchvision==0.13.1 torchaudio==0.20.0 pytorch-cuda=12.4 -c pytorch -y
  conda install h5py pyyaml -c anaconda -y
  conda install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
  conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y
  pip install torch-geometric

  # spconv (SparseUNet)
  # refer https://github.com/traveller59/spconv
  pip install spconv-cu124

  # PPT (clip)
  pip install ftfy regex tqdm
  pip install git+https://github.com/openai/CLIP.git

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

The preprocessing supports semantic and instance segmentation for both `ScanNet20`, `ScanNet200`, and `ScanNet Data Efficient`.
- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:

  ```bash
  # RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
  # PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset (output dir).
  python pointcept/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
  ```
- (Optional) Download ScanNet Data Efficient files:
  ```bash
  # download-scannet.py is the official download script
  # or follow instructions here: https://kaldir.vc.in.tum.de/scannet_benchmark/data_efficient/documentation#download
  python download-scannet.py --data_efficient -o ${RAW_SCANNET_DIR}
  # unzip downloads
  cd ${RAW_SCANNET_DIR}/tasks
  unzip limited-annotation-points.zip
  unzip limited-reconstruction-scenes.zip
  # copy files to processed dataset folder
  mkdir ${PROCESSED_SCANNET_DIR}/tasks
  cp -r ${RAW_SCANNET_DIR}/tasks/points ${PROCESSED_SCANNET_DIR}/tasks
  cp -r ${RAW_SCANNET_DIR}/tasks/scenes ${PROCESSED_SCANNET_DIR}/tasks
  ```
- (Alternative) Our preprocess data can be directly downloaded [[here](https://huggingface.co/datasets/Pointcept/scannet-compressed)], please agree the official license before download it.

- Link processed dataset to codebase:
  ```bash
  # PROCESSED_SCANNET_DIR: the directory of the processed ScanNet dataset.
  mkdir data
  ln -s ${PROCESSED_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
  ```

### ScanNet++
- Download the [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) dataset.
- Run preprocessing code for raw ScanNet++ as follows:
  ```bash
  # RAW_SCANNETPP_DIR: the directory of downloaded ScanNet++ raw dataset.
  # PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
  # NUM_WORKERS: the number of workers for parallel preprocessing.
  python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py --dataset_root ${RAW_SCANNETPP_DIR} --output_root ${PROCESSED_SCANNETPP_DIR} --num_workers ${NUM_WORKERS}
  ```
- Sampling and chunking large point cloud data in train/val split as follows (only used for training):
  ```bash
  # PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
  # NUM_WORKERS: the number of workers for parallel preprocessing.
  python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers ${NUM_WORKERS}
  python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ${PROCESSED_SCANNETPP_DIR} --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers ${NUM_WORKERS}
  ```
- Link processed dataset to codebase:
  ```bash
  # PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet dataset.
  mkdir data
  ln -s ${PROCESSED_SCANNETPP_DIR} ${CODEBASE_DIR}/data/scannetpp
  ```

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2.zip` file and unzip it.
- Fix error in `Area_5/office_19/Annotations/ceiling` Line 323474 (103.0�0000 => 103.000000).
- (Optional) Download Full 2D-3D S3DIS dataset (no XYZ) from [here](https://github.com/alexsax/2D-3D-Semantics) for parsing normal.
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

- (Alternative) Our preprocess data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/s3dis-compressed
)] (with normal vector and aligned angle), please agree with the official license before downloading it.

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
  mkdir data
  ln -s ${PROCESSED_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
  ```


### ArkitScenes

- Download ArkitScenes 3DOD split with the following commands:
  ```bash
  # RAW_AS_DIR: the directory of downloaded Raw ArkitScenes dataset.
  git clone https://github.com/apple/ARKitScenes.git
  cd ARKitScenes
  python download_data.py 3dod --download_dir $RAW_AS_DIR --video_id_csv threedod/3dod_train_val_splits.csv
  ```
- Run preprocessing code for ArkitScenes as follows:
  ```bash
  # RAW_AS_DIR: the directory of downloaded ArkitScenes dataset.
  # PROCESSED_AS_DIR: the directory of processed ArkitScenes dataset (output dir).
  # NUM_WORKERS: Number for workers for preprocessing, default same as cpu count (might OOM).
  cd $POINTCEPT_DIR
  export PYTHONPATH=./
  python pointcept/datasets/preprocessing/arkitscenes/preprocess_arkitscenes_mesh.py --dataset_root $RAW_AS_DIR --output_root $PROCESSED_AS_DIR --num_workers $NUM_WORKERS
  ```

- (Alternative) Our preprocess data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/arkitscenes-compressed
)] please read and agree the official [license](https://github.com/apple/ARKitScenes?tab=License-1-ov-file#readme) before download it. (Unzip with the following command:  
 `find ./ -name '*.tar.gz' | xargs -n 1 -P 8 -I {} sh -c 'tar -xzvf {}'`)

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_AR_DIR: the directory of processed ArkitScenes dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_AR_DIR} ${CODEBASE_DIR}/data/arkitscenes
  ```

### Habitat - Matterport 3D (HM3D)

- Download HM3D `hm3d-train-glb-v0.2.tar` and `hm3d-val-glb-v0.2.tar` with instuction [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d) and unzip them.
- Run preprocessing code for HM3D as follows:
  ```bash
  # RAW_HM_DIR: the directory of downloaded HM3D dataset.
  # PROCESSED_HM_DIR: the directory of processed HM3D dataset (output dir).
  # NUM_WORKERS: Number for workers for preprocessing, default same as cpu count (might OOM).
  export PYTHONPATH=./
  python pointcept/datasets/preprocessing/hm3d/preprocess_hm3d.py --dataset_root $RAW_HM_DIR --output_root $PROCESSED_HM_DIR --density 0.02 --num_workers $NUM_WORKERS
  ```

- (Alternative) Our preprocess data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/hm3d-compressed
)] please read and agree the official [license](https://matterport.com/legal/matterport-end-user-license-agreement-academic-use-model-data) before download it. (Unzip with the following command:  
 `find ./ -name '*.tar.gz' | xargs -n 1 -P 4 -I {} sh -c 'tar -xzvf {}'`)

- Link processed dataset to codebase.
  ```bash
  # PROCESSED_HM_DIR: the directory of processed HM3D dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_HM_DIR} ${CODEBASE_DIR}/data/hm3d


### Matterport3D
- Follow [this page](https://niessner.github.io/Matterport/#download) to request access to the dataset.
- Download the "region_segmentation" type, which represents the division of a scene into individual rooms.
  ```bash
  # download-mp.py is the official download script
  # MATTERPORT3D_DIR: the directory of downloaded Matterport3D dataset.
  python download-mp.py -o {MATTERPORT3D_DIR} --type region_segmentations
  ```
- Unzip the region_segmentations data
  ```bash
  # MATTERPORT3D_DIR: the directory of downloaded Matterport3D dataset.
  python pointcept/datasets/preprocessing/matterport3d/unzip_matterport3d_region_segmentation.py --dataset_root {MATTERPORT3D_DIR}
  ```
- Run preprocessing code for Matterport3D as follows:
  ```bash
  # MATTERPORT3D_DIR: the directory of downloaded Matterport3D dataset.
  # PROCESSED_MATTERPORT3D_DIR: the directory of processed Matterport3D dataset (output dir).
  # NUM_WORKERS: the number of workers for this preprocessing.
  python pointcept/datasets/preprocessing/matterport3d/preprocess_matterport3d_mesh.py --dataset_root ${MATTERPORT3D_DIR} --output_root ${PROCESSED_MATTERPORT3D_DIR} --num_workers ${NUM_WORKERS}
  ```
- Link processed dataset to codebase.
  ```bash
  # PROCESSED_MATTERPORT3D_DIR: the directory of processed Matterport3D dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_MATTERPORT3D_DIR} ${CODEBASE_DIR}/data/matterport3d
  ```

Following the instruction of [OpenRooms](https://github.com/ViLab-UCSD/OpenRooms), we remapped Matterport3D's categories to ScanNet 20 semantic categories with the addition of a ceiling category.
* (Alternative) Our preprocess data can also be downloaded [here](https://huggingface.co/datasets/Pointcept/matterport3d-compressed), please agree the official license before download it.


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

- (Alternative) Our preprocess data can also be downloaded [[here](https://huggingface.co/datasets/Pointcept/structured3d-compressed
)] (with perspective views and panorama view, 471.7G after unzipping), please agree the official license before download it. (Unzip with the following command:  
 `find ./ -name '*.tar.gz' | xargs -n 1 -P 15 -I {} sh -c 'tar -xzvf {}'`)

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
https://huggingface.co/datasets/Pointcept/nuscenes-compressed)] (only processed information, still need to download raw dataset and link to the folder), please agree the official license before download it.

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

### Waymo
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
  ```

- Link processed dataset to the codebase.
  ```bash
  # PROCESSED_WAYMO_DIR: the directory of the processed Waymo dataset (output dir).
  mkdir data
  ln -s ${PROCESSED_WAYMO_DIR} ${CODEBASE_DIR}/data/waymo
  ```

### ModelNet
- Download [modelnet40_normal_resampled.zip](https://huggingface.co/datasets/Pointcept/modelnet40_normal_resampled-compressed) and unzip
- Link dataset to the codebase.
  ```bash
  mkdir -p data
  ln -s ${MODELNET_DIR} ${CODEBASE_DIR}/data/modelnet40_normal_resampled
  ```

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard, and checkpoints will also be saved into the experiment folder during the training process.
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
**Weights and Biases.**
Pointcept by default enables both `tensorboard` and `wandb`. There are some usage notes related to `wandb`:
1. Disable by set `enable_wandb=False`;
2. Sync with  `wandb` remote server by `wandb login` in the terminal or set `wandb_key=YOUR_WANDB_KEY` in config.
3. The project name is "Pointcept" by default, custom it to your research project name by setting `wandb_project=YOUR_PROJECT_NAME` (e.g. Sonata-Dev, PointTransformerV3-Dev)

### Testing
During training, model evaluation is performed on point clouds after grid sampling (voxelization), providing an initial assessment of model performance. ~~However, to obtain precise evaluation results, testing is **essential**~~ *(now we automatically run the testing process after training with the `PreciseEvaluation` hook)*. The testing process involves subsampling a dense point cloud into a sequence of voxelized point clouds, ensuring comprehensive coverage of all points. These sub-results are then predicted and collected to form a complete prediction of the entire point cloud. This approach yields  higher evaluation results compared to simply mapping/interpolating the prediction. In addition, our testing code supports TTA (test time augmentation) testing, which further enhances the stability of evaluation performance.

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```
For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-pt-v2m2-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-pt-v2m2-0-base.py --options save_path=exp/scannet/semseg-pt-v2m2-0-base weight=exp/scannet/semseg-pt-v2m2-0-base/model/model_best.pth
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
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png">
    <!-- /pypi-strip -->
    <img alt="pointcept" src="https://raw.githubusercontent.com/pointcept/assets/main/pointcept/offset.png" width="480">
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

The MinkowskiEngine version `SparseUNet` in the codebase was modified from the original MinkowskiEngine repo, and example running scripts are as follows:
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

#### OA-CNNs
Introducing Omni-Adaptive 3D CNNs (**OA-CNNs**), a family of networks that integrates a lightweight module to greatly enhance the adaptivity of sparse CNNs at minimal computational cost. Without any self-attention modules, **OA-CNNs** favorably surpass point transformers in terms of accuracy in both indoor and outdoor scenes, with much less latency and memory cost. Issue related to **OA-CNNs** can @Pbihao.
```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-oacnns-v1m1-0-base -n semseg-oacnns-v1m1-0-base
```

#### Point Transformers
- **PTv3**

[PTv3](https://arxiv.org/abs/2312.10035) is an efficient backbone model that achieves SOTA performances across indoor and outdoor scenarios. The full PTv3 relies on FlashAttention, while FlashAttention relies on CUDA 11.6 and above, make sure your local Pointcept environment satisfies the requirements.

If you can not upgrade your local environment to satisfy the requirements (CUDA >= 11.6), then you can disable FlashAttention by setting the model parameter `enable_flash` to `false` and reducing the `enc_patch_size` and `dec_patch_size` to a level (e.g. 128).

FlashAttention force disables RPE and forces the accuracy reduced to fp16. If you require these features, please disable `enable_flash` and adjust `enable_rpe`, `upcast_attention` and`upcast_softmax`.

Detailed instructions and experiment records (containing weights) are available on the [project repository](https://github.com/Pointcept/PointTransformerV3). Example running scripts are as follows:
```bash
# Scratched ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
# PPT joint training (ScanNet + Structured3D) and evaluate in ScanNet
sh scripts/train.sh -g 8 -d scannet -c semseg-pt-v3m1-1-ppt-extreme -n semseg-pt-v3m1-1-ppt-extreme

# Scratched ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
# Fine-tuning from  PPT joint training (ScanNet + Structured3D) with ScanNet200
# PTV3_PPT_WEIGHT_PATH: Path to model weight trained by PPT multi-dataset joint training
# e.g. exp/scannet/semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v3m1-1-ppt-ft -n semseg-pt-v3m1-1-ppt-ft -w ${PTV3_PPT_WEIGHT_PATH}

# Scratched ScanNet++
sh scripts/train.sh -g 4 -d scannetpp -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
# Scratched ScanNet++ test
sh scripts/train.sh -g 4 -d scannetpp -c semseg-pt-v3m1-1-submit -n semseg-pt-v3m1-1-submit


# Scratched S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
# an example for disbale flash_attention and enable rpe.
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-1-rpe -n semseg-pt-v3m1-0-rpe
# PPT joint training (ScanNet + S3DIS + Structured3D) and evaluate in ScanNet
sh scripts/train.sh -g 8 -d s3dis -c semseg-pt-v3m1-1-ppt-extreme -n semseg-pt-v3m1-1-ppt-extreme
# S3DIS 6-fold cross validation
# 1. The default configs are evaluated on Area_5, modify the "data.train.split", "data.val.split", and "data.test.split" to make the config evaluated on Area_1 ~ Area_6 respectively.
# 2. Train and evaluate the model on each split of areas and gather result files located in "exp/s3dis/EXP_NAME/result/Area_x.pth" in one single folder, noted as RECORD_FOLDER.
# 3. Run the following script to get S3DIS 6-fold cross validation performance:
export PYTHONPATH=./
python tools/test_s3dis_6fold.py --record_root ${RECORD_FOLDER}

# Scratched nuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
# Scratched Waymo
sh scripts/train.sh -g 4 -d waymo -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base

# More configs and exp records for PTv3 will be available soon.
```

Indoor semantic segmentation  
| Model | Benchmark | Additional Data | Num GPUs | Val mIoU | Config | Tensorboard | Exp Record |
| :---: | :---: |:---------------:| :---: | :---: | :---: | :---: | :---: |
| PTv3 | ScanNet |     &cross;     | 4 | 77.6% | [link](https://github.com/Pointcept/Pointcept/blob/main/configs/scannet/semseg-pt-v3m1-0-base.py) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tensorboard) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet-semseg-pt-v3m1-0-base) |
| PTv3 + PPT | ScanNet |     &check;     | 8 | 78.5% | [link](https://github.com/Pointcept/Pointcept/blob/main/configs/scannet/semseg-pt-v3m1-1-ppt-extreme.py) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tensorboard) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet-semseg-pt-v3m1-1-ppt-extreme) |
| PTv3 | ScanNet200 |     &cross;     | 4 | 35.3% | [link](https://github.com/Pointcept/Pointcept/blob/main/configs/scannet200/semseg-pt-v3m1-0-base.py) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tensorboard) |[link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/scannet200-semseg-pt-v3m1-0-base)|
| PTv3 | S3DIS (Area5) |     &cross;     | 4 | 73.6% | [link](https://github.com/Pointcept/Pointcept/blob/main/configs/s3dis/semseg-pt-v3m1-0-rpe.py) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tensorboard) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/s3dis-semseg-pt-v3m1-0-rpe) |
| PTv3 + PPT | S3DIS (Area5) |     &check;     | 8 | 75.4% | [link](https://github.com/Pointcept/Pointcept/blob/main/configs/s3dis/semseg-pt-v3m1-1-ppt-extreme.py) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tensorboard) | [link](https://huggingface.co/Pointcept/PointTransformerV3/tree/main/s3dis-semseg-pt-v3m1-1-ppt-extreme) |
_**\*Released model weights are trained for v1.5.1, weights for v1.5.2 and later is still ongoing.**_

- **PTv2 mode2**

The original PTv2 was trained on 4 * RTX a6000 (48G memory). Even enabling AMP, the memory cost of the original PTv2 is slightly larger than 24G. Considering GPUs with 24G memory are much more accessible, I tuned the PTv2 on the latest Pointcept and made it runnable on 4 * RTX 3090 machines.

`PTv2 Mode2` enables AMP and disables _Position Encoding Multiplier_ & _Grouped Linear_. During our further research, we found that precise coordinates are not necessary for point cloud understanding (Replacing precise coordinates with grid coordinates doesn't influence the performance. Also, SparseUNet is an example). As for Grouped Linear, my implementation of Grouped Linear seems to cost more memory than the Linear layer provided by PyTorch. Benefiting from the codebase and better parameter tuning, we also relieve the overfitting problem. The reproducing performance is even better than the results reported in our paper.

Example running scripts are as follows:

```bash
# ptv2m2: PTv2 mode2, disable PEM & Grouped Linear, GPU memory cost < 24G (recommend)
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-3-lovasz -n semseg-pt-v2m2-3-lovasz

# ScanNet test
sh scripts/train.sh -g 4 -d scannet -c semseg-pt-v2m2-1-submit -n semseg-pt-v2m2-1-submit
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# ScanNet++
sh scripts/train.sh -g 4 -d scannetpp -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# ScanNet++ test
sh scripts/train.sh -g 4 -d scannetpp -c semseg-pt-v2m2-1-submit -n semseg-pt-v2m2-1-submit
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# SemanticKITTI
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
# nuScenes
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v2m2-0-base -n semseg-pt-v2m2-0-base
```

- **PTv2 mode1**

`PTv2 mode1` is the original PTv2 we reported in our paper, example running scripts are as follows:

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
# Fix dependence, caused by installing torch-points3d 
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
`SPVCNN` is a baseline model of [SPVNAS](https://github.com/mit-han-lab/spvnas), it is also a practical baseline for outdoor datasets.
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
#### Concerto
Follow the instruction [here](https://github.com/Pointcept/Pointcept/tree/main/pointcept/models/concerto).

#### Sonata
Follow the instruction [here](https://github.com/Pointcept/Pointcept/tree/main/pointcept/models/sonata).

#### Masked Scene Contrast (MSC)
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

#### Point Prompt Training (PPT)
PPT presents a multi-dataset pre-training framework, and it is compatible with various existing pre-training frameworks and backbones.
1. PPT supervised joint training with the following example scripts:
```bash
# ScanNet + Structured3d, validate on ScanNet (S3DIS might cause long data time, w/o S3DIS for a quick validation) >= 3090 * 8 
sh scripts/train.sh -g 8 -d scannet -c semseg-ppt-v1m1-0-sc-st-spunet -n semseg-ppt-v1m1-0-sc-st-spunet
sh scripts/train.sh -g 8 -d scannet -c semseg-ppt-v1m1-1-sc-st-spunet-submit -n semseg-ppt-v1m1-1-sc-st-spunet-submit
# ScanNet + S3DIS + Structured3d, validate on S3DIS (>= a100 * 8)
sh scripts/train.sh -g 8 -d s3dis -c semseg-ppt-v1m1-0-s3-sc-st-spunet -n semseg-ppt-v1m1-0-s3-sc-st-spunet
# SemanticKITTI + nuScenes + Waymo, validate on SemanticKITTI (bs12 >= 3090 * 4 >= 3090 * 8, v1m1-0 is still on tuning)
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-ppt-v1m1-0-nu-sk-wa-spunet -n semseg-ppt-v1m1-0-nu-sk-wa-spunet
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-ppt-v1m2-0-sk-nu-wa-spunet -n semseg-ppt-v1m2-0-sk-nu-wa-spunet
sh scripts/train.sh -g 4 -d semantic_kitti -c semseg-ppt-v1m2-1-sk-nu-wa-spunet-submit -n semseg-ppt-v1m2-1-sk-nu-wa-spunet-submit
# SemanticKITTI + nuScenes + Waymo, validate on nuScenes (bs12 >= 3090 * 4; bs24 >= 3090 * 8, v1m1-0 is still on tuning))
sh scripts/train.sh -g 4 -d nuscenes -c semseg-ppt-v1m1-0-nu-sk-wa-spunet -n semseg-ppt-v1m1-0-nu-sk-wa-spunet
sh scripts/train.sh -g 4 -d nuscenes -c semseg-ppt-v1m2-0-nu-sk-wa-spunet -n semseg-ppt-v1m2-0-nu-sk-wa-spunet
sh scripts/train.sh -g 4 -d nuscenes -c semseg-ppt-v1m2-1-nu-sk-wa-spunet-submit -n semseg-ppt-v1m2-1-nu-sk-wa-spunet-submit
```

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
3. Fine-tuning refer [MSC](#masked-scene-contrast-msc).

#### Contrastive Scene Contexts
1. Preprocess and link ScanNet-Pair dataset (refer [PointContrast](#pointcontrast)):
2. Pre-training with the following example scripts:
```bash
# ScanNet
sh scripts/train.sh -g 8 -d scannet -c pretrain-msc-v1m2-0-spunet-csc -n pretrain-msc-v1m2-0-spunet-csc
```
3. Fine-tuning refer [MSC](#masked-scene-contrast-msc).

## Acknowledgement
_Pointcept_ is designed by [Xiaoyang](https://xywu.me/), named by [Yixing](https://github.com/yxlao) and the logo is created by [Yuechen](https://julianjuaner.github.io/). It is derived from [Hengshuang](https://hszhao.github.io/)'s [Semseg](https://github.com/hszhao/semseg) and inspirited by several repos, e.g., [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
