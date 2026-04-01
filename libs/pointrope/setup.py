"""
setup.py script for PointROPE

Author: Yuanwen Yue (yuayue@ethz.ch)
Please cite our work if the code is helpful to you.
"""

from setuptools import setup
from torch import cuda
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# compile for all possible CUDA architectures
all_cuda_archs = cuda.get_gencode_flags().replace("compute=", "arch=").split()
# alternatively, you can list cuda archs that you want, eg:
# check https://developer.nvidia.com/cuda-gpus to find your arch
# all_cuda_archs = [
#     '-gencode', 'arch=compute_90,code=sm_90',
#     # '-gencode', 'arch=compute_75,code=sm_75',
#     # '-gencode', 'arch=compute_80,code=sm_80',
#     # '-gencode', 'arch=compute_86,code=sm_86'
# ]

setup(
    name="pointrope",
    ext_modules=[
        CUDAExtension(
            name="pointrope",
            sources=[
                "pointrope.cpp",
                "kernels.cu",
            ],
            extra_compile_args=dict(
                nvcc=["-O3", "--ptxas-options=-v", "--use_fast_math"] + all_cuda_archs,
                cxx=["-O3"],
            ),
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
