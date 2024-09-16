"""
Hook Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.utils.registry import Registry


HOOKS = Registry("hooks")


def build_hooks(cfg):
    hooks = []
    for hook_cfg in cfg:
        hooks.append(HOOKS.build(hook_cfg))
    return hooks
