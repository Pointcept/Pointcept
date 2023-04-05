"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import random
import numpy as np
import argparse
import collections

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.datasets.utils import collate_fn
from pointcept.utils.config import Config, DictAction
from pointcept.utils.logger import get_root_logger
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.engines.test import TEST


def get_parser():
    parser = argparse.ArgumentParser(description='Pointcept Test Process')
    parser.add_argument('--config-file', default="", metavar="FILE", help="path to config file")
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    args = parser.parse_args()
    return args


def main():
    args = get_parser()

    # config_parser
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    os.makedirs(cfg.save_path, exist_ok=True)

    # default_setup
    set_seed(cfg.seed)
    cfg.batch_size_val_per_gpu = cfg.batch_size_test  # TODO: add support to multi gpu test
    cfg.num_worker_per_gpu = cfg.num_worker  # TODO: add support to multi gpu test

    # tester init
    weight_name = os.path.basename(cfg.weight).split(".")[0]
    logger = get_root_logger(log_file=os.path.join(cfg.save_path, "test-{}.log".format(weight_name)))
    logger.info("=> Loading config ...")
    logger.info(f"Save path: {cfg.save_path}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    # build model
    logger.info("=> Building model ...")
    model = build_model(cfg.model).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")

    # build dataset
    logger.info("=> Building test dataset & dataloader ...")
    test_dataset = build_dataset(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size_val_per_gpu,
                                              shuffle=False,
                                              num_workers=cfg.num_worker_per_gpu,
                                              pin_memory=True,
                                              collate_fn=collate_fn)

    # load checkpoint
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for name, value in state_dict.items():
            if name.startswith("module."):
                name = name[7:]  # module.xxx.xxx -> xxx.xxx
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> loaded weight '{}' (epoch {})".format(cfg.weight, checkpoint['epoch']))
        cfg.epochs = checkpoint['epoch']  # TODO: move to self
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(cfg.weight))
    TEST.build(cfg.test)(cfg, test_loader, model)


if __name__ == '__main__':
    main()
