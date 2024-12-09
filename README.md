# Pointcept Installation
---
Official Repo link: https://gitlab.lrz.de/projects-phd/pointcept-scannetpp-v2

Our fork: https://gitlab.lrz.de/00000000014B31F0/pointcept (this has setup files to make things quicker)

## Base Setup
* clone repo
* `cd ./PointCept`
* `pip install poetry`
* `poetry install`
* `poetry shell`
---
## Pointcept Dependencies
* `pip install h5py pyyaml`
* `pip install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm`
* `pip install torch-cluster torch-scatter torch-sparse`
* `pip install torch-geometric`
* `pip install spconv-cu120` <---- This depends on the installed cuda version: https://pypi.org/project/spconv/
* `pip install ftfy regex tqdm`
* `pip install git+https://github.com/openai/CLIP.git`
* `pip install flash-attn`
* `pip install open3d`
* `cd libs/pointops`
* `python setup.py install`
---
## Check Installation
 At this point every thing should be ready to go. To check for a valid isntallation, running the notebook `setup_check.ipynb` should work correctly

----

## Data Preperation
official guide: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet
* Sampling and chunking large point cloud data in train/val split as follows (only used for training):
```
# RAW_SCANNETPP_DIR: the directory of downloaded ScanNet++ raw dataset.
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/scannetpp/preprocess_scannetpp.py --dataset_root ${RAW_SCANNETPP_DIR} --output_root ./raw_dataset --num_workers ${NUM_WORKERS}
```
*Note: it is very important to keep the `output folder` as `./raw_datset` because this is the assumed output folder in all the config files*

* Sampling and chunking large point cloud data in train/val split as follows (only used for training)
```
# PROCESSED_SCANNETPP_DIR: the directory of the processed ScanNet++ dataset (output dir).
# NUM_WORKERS: the number of workers for parallel preprocessing.
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ./raw_dataset --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split train --num_workers ${NUM_WORKERS}
python pointcept/datasets/preprocessing/sampling_chunking_data.py --dataset_root ./raw_dataset --grid_size 0.01 --chunk_range 6 6 --chunk_stride 3 3 --split val --num_workers ${NUM_WORKERS}

```
---
## Training

official guide: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#quick-start

* login to wandb from the terminal by `wandb login <your api key>`
* All commans should be run inside a poetry shell i.e run `poetry shell` in the project root.

* General command structure
```
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}

```
* *Note: you can use nohup to keep the process running in the background by running `nohup python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} & `*

* Ready to go training scripts
```
#ptv3 
python tools/train.py --config-file ./configs/scannetpp/semseg-pt-v3m1-0-base.py --num-gpus 1 --options save_path=./exp/ptv3/full-train-1 &

#OACnn
python tools/train.py --config-file ./configs/scannetpp/semseg-oacnn-v1m1-0-base.py --num-gpus 1 --options save_path=./exp/oacnn/full-train-1 &

#Octformer
python tools/train.py --config-file ./configs/scannetpp/semseg-octformer-v1m1-0-base.py --num-gpus 1 --options save_path=./exp/oct-former/full-train-1 &

# Context aware classifier
python tools/train.py --config-file ./configs/scannetpp/semseg-cac-v1m1-0-base.py --num-gpus 1 --options save_path=./exp/cac/full-train-1 &

# ptv2
python tools/train.py --config-file ./configs/scannetpp/semseg-pt-v2m2-0-base.py --num-gpus 1 --options save_path=./exp/ptv2/full-train-1 &

#spunet
python tools/train.py --config-file ./configs/scannetpp/semseg-spunet-v1m1-0-base.py --num-gpus 1 --options save_path=./exp/spUnet/full-train-1 &
```
### Notes
* Each config file contains relevant settings/hyperparams for the training experiment.
* The batch size for each experiment is set to maximize a single A500 24GB gpu, so it can be changed if a larger gpu is available.
* The wandb configs for each experiment are also in the config files `wandb_project_name`, `wandb_tags`, `enable_wandb`
---
## Evaluation

Official guide: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#testing
* Running the evaluation is very straight forward like for training

* General command structure: 
```
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}

```

* Ready to scripts
```

#ptv3 
python tools/test.py --config-file ./configs/scannetpp/semseg-pt-v3m1-0-base.py --num-gpus 1 --options save_path="./exp/ptv3_eval_val" weight="./exp/ptv3/full-train-1/model/model_best.pth"

#OACnn
python tools/test.py --config-file ./configs/scannetpp/semseg-oacnn-v1m1-0-base.py --num-gpus 1 --options save_path="./exp/oacnn_eval_val" weight="./exp/oacnn/full-train-1/model/model_best.pth"


semseg-octformer-v1m1-0-base.p
#Octformer
python tools/test.py --config-file ./configs/scannetpp/semseg-octformer-v1m1-0-base.py --num-gpus 1 --options save_path="./exp/octformer_eval_val" weight="./exp/oct-former/full-train-1/model/model_best.pth"
...
```
