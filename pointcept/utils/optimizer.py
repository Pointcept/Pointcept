"""
Optimizer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
from .registry import Registry

OPTIMIZERS = Registry("optimizers")


OPTIMIZERS.register_module(module=torch.optim.SGD, name="SGD")
OPTIMIZERS.register_module(module=torch.optim.Adam, name="Adam")
OPTIMIZERS.register_module(module=torch.optim.AdamW, name="AdamW")


def build_optimizer(cfg, model, params_dicts=None):
    if params_dicts is None:
        cfg.params = model.parameters()
    else:
        cfg.params = [dict(params=[])]
        for i in range(len(params_dicts)):
            cfg.params.append(dict(params=[], lr=params_dicts[i].lr_scale * cfg.lr))

        for n, p in model.named_parameters():
            flag = False
            for i in range(len(params_dicts)):
                if params_dicts[i].keyword in n:
                    cfg.params[i+1]["params"].append(p)
                    flag = True
                    break
            if not flag:
                cfg.params[0]["params"].append(p)
    return OPTIMIZERS.build(cfg=cfg)
