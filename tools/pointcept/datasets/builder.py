"""
Dataset Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry

DATASETS = Registry("datasets")


def build_dataset(cfg):
    """Build datasets."""
    return DATASETS.build(cfg)
