TORCH_VERSION=2.0.1
CUDA_VERSION=11.7
CUDNN_VERSION=8

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
IMG_TAG=pointcept/pointcept:pytorch${BASE_TORCH_TAG}

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
RUN conda install h5py pyyaml -c anaconda -y
RUN conda install sharedarray tensorboard tensorboardx yapf addict einops scipy plyfile termcolor timm -c conda-forge -y
RUN conda install pytorch-cluster pytorch-scatter pytorch-sparse -c pyg -y

RUN pip install --upgrade pip
RUN pip install torch-geometric
RUN pip install spconv-cu${CUDA_VERSION_NO_DOT}
RUN pip install open3d

# Build MinkowskiEngine
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git
WORKDIR /workspace/MinkowskiEngine
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" python setup.py install --blas=openblas --force_cuda
WORKDIR /workspace

# Build pointops
RUN git clone https://github.com/Pointcept/Pointcept.git
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointops -v

# Build pointgroup_ops
RUN TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0+PTX 8.0" pip install Pointcept/libs/pointgroup_ops -v

# Build swin3d
RUN TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0" pip install -U git+https://github.com/microsoft/Swin3D.git -v
EOM

docker build . -f ./Dockerfile -t $IMG_TAG