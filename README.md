# Pointcept Installation
---
Official Repo link: https://github.com/Pointcept/Pointcept

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
For the dataset preperation, it is best to follow the official guide: https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet

---
## Training

For training, it is best to follow the official guide:
https://github.com/Pointcept/Pointcept?tab=readme-ov-file#scannet


