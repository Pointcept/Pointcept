TORCH_VERSION=2.5.0
CUDA_VERSION=12.4
CUDNN_VERSION=9

ARGS=`getopt -o t:c: -l torch:,cuda:,cudnn: -n "$0" -- "$@"`
[ $? != 0 ] && exit 1
eval set -- "${ARGS}"
while true ; do
  case "$1" in
    -t | --torch)
      TORCH_VERSION=$2
      shift 2
      ;;
    -c | --cuda)
      CUDA_VERSION=$2
      shift 2
      ;;
    --cudnn)
      CUDNN_VERSION=$2
      shift 2
      ;;
    --)
      break
      ;;
    *)
      echo "Invalid option: $1"
      exit 1
      ;;
  esac
done

CUDA_VERSION_NO_DOT=`echo ${CUDA_VERSION} | tr -d "."`
BASE_TORCH_TAG=${TORCH_VERSION}-cuda${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel
IMG_TAG=pointcept/pointcept:v1.6.0-pytorch${BASE_TORCH_TAG}

echo "TORCH VERSION: ${TORCH_VERSION}"
echo "CUDA VERSION: ${CUDA_VERSION}"
echo "CUDNN VERSION: ${CUDNN_VERSION}"


cat > ./Dockerfile <<- EOM
FROM pytorch/pytorch:${BASE_TORCH_TAG}

# Fix nvidia-key error issue (NO_PUBKEY A4B469963BF863CC)
RUN rm /etc/apt/sources.list.d/*.list

# Installing apt packages
RUN export DEBIAN_FRONTEND=noninteractive \
	&& apt -y update --no-install-recommends \
	&& apt -y install --no-install-recommends \
	  git wget tmux vim zsh build-essential cmake ninja-build libopenblas-dev libsparsehash-dev \
	&& apt autoremove -y \
	&& apt clean -y \
	&& export DEBIAN_FRONTEND=dialog

# Install Pointcept environment
RUN conda install h5py pyyaml tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor matplotlib black open3d -c conda-forge -y

RUN pip install --upgrade pip
RUN pip install timm
RUN pip install torch-geometric
RUN pip install torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cu${CUDA_VERSION_NO_DOT}.html
RUN pip install spconv-cu${CUDA_VERSION_NO_DOT}
RUN pip install git+https://github.com/octree-nn/ocnn-pytorch.git
RUN pip install ftfy regex tqdm
RUN pip install git+https://github.com/openai/CLIP.git

# Build swin3d
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install -U git+https://github.com/microsoft/Swin3D.git -v

# Build FlashAttention2
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install git+https://github.com/Dao-AILab/flash-attention.git

# Build pointops
RUN git clone https://github.com/Pointcept/Pointcept.git
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install Pointcept/libs/pointops -v

# Build pointgroup_ops
RUN TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0" pip install Pointcept/libs/pointgroup_ops -v

EOM

docker build . -f ./Dockerfile -t $IMG_TAG