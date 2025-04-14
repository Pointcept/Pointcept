from setuptools import find_packages, setup
import os.path as osp
import torch
from glob import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_sources(module, surfix="*.c*"):
    # src_dir = osp.join(*module.split('.'), 'src')
    src_dir = osp.join("src")
    cuda_dir = osp.join(src_dir, "cuda")
    cpu_dir = osp.join(src_dir, "cpu")
    return (
        glob(osp.join(src_dir, surfix))
        + glob(osp.join(cuda_dir, surfix))
        + glob(osp.join(cpu_dir, surfix))
    )


def get_include_dir(module):
    include_dir = osp.join(*module.split("."), "include")
    if osp.exists(include_dir):
        return [osp.abspath(include_dir)]
    else:
        return []


def make_extension(name, module):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please ensure CUDA is installed and accessible."
        )
    return CUDAExtension(
        name=name,
        sources=get_sources(module),
        include_dirs=get_include_dir(module),
        extra_compile_args={
            "cxx": ["-g"],
            "nvcc": [
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
        define_macros=[("WITH_CUDA", None)],
    )


setup(
    name="pointgroup_ops_sp",
    packages=["pointgroup_ops_sp"],
    package_dir={"pointgroup_ops_sp": "functions"},
    ext_modules=[
        make_extension(name="pointgroup_ops_sp_cuda", module="pointgroup_ops_sp")
    ],
    cmdclass={"build_ext": BuildExtension},
)
