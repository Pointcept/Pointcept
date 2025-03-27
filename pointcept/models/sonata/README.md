# Sonata
This repo is the official training codebase of the paper **_Sonata: Self-Supervised Learning of Reliable Point Representations_**. Sonata is a powerful self-supervised learning framework (as well as our pre-trained model) that shows extraordinary parameter efficiency, data efficiency, and fine-tuning accuracy.

*We recommend beginning with our inference [demo](https://github.com/facebookresearch/sonata) and it is also a tiny library for people who want to integrate pre-trained Sonata into their project. This section is designed for users interested in reproduce our pre-training.*

[ **Pretrain** ] [ **Sonata** ] - [ [Homepage](https://xywu.me/sonata/) ] [ [Paper](https://arxiv.org/abs/2503.16429) ] [ [Inference Demo](https://github.com/facebookresearch/sonata) ] [ [Bib](#citation) ]

<div align='left'>
<img src="https://raw.githubusercontent.com/pointcept/assets/main/sonata/teaser.png" alt="teaser" width="800" />
</div>

## Quick start

#### 1. Installation
Follow the instruction [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation).

#### 2. Dataset
Downloading and preprocessing the following datasets:
- ScanNet v2 - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet-v2) ]
- ScanNet++ v2 - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet) ]
- S3DIS - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#s3dis) ]
- ArkitScenes - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#arkitscenes) ]
- HM3D - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#habitat---matterport-3d-hm3d) ]
- Structured3D - [ [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#structured3d) ]

The whole list of our supported datasets can be found [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#data-preparation) and some of the preprocessed datasets can be found [here](https://huggingface.co/Pointcept).


#### 3. Pre-training
Enable pre-training with 32 GPUs (minimal 16 GPUs) by running the following script:
```bash
# default config
sh scripts/train.sh -m 4 -g 8 -d sonata -c pretrain-sonata-v1m1-0-base -n pretrain-sonata-v1m1-0-base

# (or) a variant seem like have a better performance
sh scripts/train.sh -m 4 -g 8 -d sonata -c pretrain-sonata-v1m2-0-uni-teacher-head -n pretrain-sonata-v1m2-0-uni-teacher-head

# (or) if only have 16 gpus
sh scripts/train.sh -m 2 -g 8 -d sonata -c pretrain-sonata-v1m1-0-base -n pretrain-sonata-v1m1-0-base

# (or) if OOM
# half the batch size and half the base learning rate
```

#### 4. Probing and Tuning
After pre-training, you can probe or tune the model with the following script:
(Our pre-trained model weight can be downloaded [here](https://huggingface.co/facebook/sonata/blob/main/pretrain-sonata-v1m1-0-base.pth))
```bash
# Assume the pre-trained experiment is record in:
# exp/sonata/pretrain-sonata-v1m1-0-base

# ScanNet 
# linear probing
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0a-scannet-lin -n semseg-sonata-v1m1-0-base-0a-scannet-lin -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
# decoder probing
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0b-scannet-dec -n semseg-sonata-v1m1-0-base-0b-scannet-dec -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
# full fine-tuning
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0c-scannet-ft -n semseg-sonata-v1m1-0-base-0c-scannet-ft -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
# PPT multi-dataset joint fine-tuning (heavy, not necessary)
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0d-scannet-ppt -n semseg-sonata-v1m1-0-base-0d-scannet-ppt -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth

# ScanNet Data Efficiency full fine-tuning
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e0-scannet-ft-la20 -n semseg-sonata-v1m1-0-base-0e0-scannet-ft-la20 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e1-scannet-ft-la50 -n semseg-sonata-v1m1-0-base-0e1-scannet-ft-la50 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e2-scannet-ft-la100 -n semseg-sonata-v1m1-0-base-0e2-scannet-ft-la100 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e3-scannet-ft-la200 -n semseg-sonata-v1m1-0-base-0e3-scannet-ft-la200 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e4-scannet-ft-lr1 -n semseg-sonata-v1m1-0-base-0e4-scannet-ft-lr1 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e5-scannet-ft-lr5 -n semseg-sonata-v1m1-0-base-0e5-scannet-ft-lr5 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e6-scannet-ft-lr10 -n semseg-sonata-v1m1-0-base-0e6-scannet-ft-lr10 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-0e7-scannet-ft-lr20 -n semseg-sonata-v1m1-0-base-0e7-scannet-ft-lr20 -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth

# ScanNet200
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-1a-scannet200-lin -n semseg-sonata-v1m1-0-base-1a-scannet200-lin -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-1b-scannet200-dec -n semseg-sonata-v1m1-0-base-1b-scannet200-dec -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-1c-scannet200-ft -n semseg-sonata-v1m1-0-base-1c-scannet200-ft -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth

# ScanNetpp
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2a-scannetpp-lin -n semseg-sonata-v1m1-0-base-2a-scannetpp-lin -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2b-scannetpp-dec -n semseg-sonata-v1m1-0-base-2b-scannetpp-dec -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-2c-scannetpp-ft -n semseg-sonata-v1m1-0-base-2c-scannetpp-ft -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth

# S3DIS Area 5
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3a-s3dis-lin -n semseg-sonata-v1m1-0-base-3a-s3dis-lin -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3b-s3dis-dec -n semseg-sonata-v1m1-0-base-3b-s3dis-dec -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth
sh scripts/train.sh -m 1 -g 8 -d sonata -c semseg-sonata-v1m1-3c-s3dis-ft -n semseg-sonata-v1m1-0-base-3c-s3dis-ft -w exp/sonata/pretrain-sonata-v1m1-0-base/model/model_last.pth

# S3DIS 6-fold
# It's easy to adapt config for each split of S3DIS.
# Gathering all results file into one folder (RECORD_ROOT) 
# and run the following script to get the final result:
python tools/test_s3dis_6fold.py --record_root $RECORD_ROOT
```
The above configs are not verified with one additional running. If you encounter any problems, please feel free to let me know. Also, comparing changes between each configuration would be helpful in handling the configuration system of Pointcept.

## Citation
If you find _Sonata_ useful to your research, please consider citing our line of works as an acknowledgment. (੭ˊ꒳​ˋ)੭✧
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
- Sonata code is released by Meta under the [Apache 2.0 license](https://github.com/facebookresearch/sonata/blob/main/LICENSE),
- Sonata weight is released under the [CC-BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en)
(restricted by NC of datasets like HM3D, ArkitScenes).
- For commercial usage, please removing datasets restrict by non-commercial license.
