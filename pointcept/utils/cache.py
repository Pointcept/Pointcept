"""
Data Cache Utils

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os

try:
    import SharedArray
except ImportError:
    SharedArray = None

try:
    from multiprocessing.shared_memory import ShareableList
except ImportError:
    import warnings

    warnings.warn("Please update python version >= 3.8 to enable shared_memory")
import numpy as np


def shared_array(name, var=None):
    if var is not None:
        # check exist
        if os.path.exists(f"/dev/shm/{name}"):
            return SharedArray.attach(f"shm://{name}")
        # create shared_array
        data = SharedArray.create(f"shm://{name}", var.shape, dtype=var.dtype)
        data[...] = var[...]
        data.flags.writeable = False
    else:
        data = SharedArray.attach(f"shm://{name}").copy()
    return data


def shared_dict(name, var=None):
    name = str(name)
    assert "." not in name  # '.' is used as sep flag
    data = {}
    if var is not None:
        assert isinstance(var, dict)
        keys = var.keys()
        # current version only cache np.array
        keys_valid = []
        for key in keys:
            if isinstance(var[key], np.ndarray):
                keys_valid.append(key)
        keys = keys_valid

        ShareableList(sequence=keys, name=name + ".keys")
        for key in keys:
            if isinstance(var[key], np.ndarray):
                data[key] = shared_array(name=f"{name}.{key}", var=var[key])
    else:
        keys = list(ShareableList(name=name + ".keys"))
        for key in keys:
            data[key] = shared_array(name=f"{name}.{key}")
    return data
