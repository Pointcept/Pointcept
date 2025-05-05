import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars("OPT")
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointseg",
    version="1.0",
    install_requires=["torch"],
    packages=["pointseg"],
    package_dir={"pointseg": "pointseg"},
    ext_modules=[
        CppExtension(
            name="pointseg._C",
            sources=sources,
            extra_compile_args=["-O3"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
