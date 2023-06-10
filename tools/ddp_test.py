"""
Main Testing Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import collections

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import pointcept.utils.comm as comm
from pointcept.engines.defaults import default_argument_parser, default_config_parser, default_setup, create_ddp_model
from pointcept.engines.launch import launch
from pointcept.models import build_model
from pointcept.datasets import build_dataset
from pointcept.utils.logger import get_root_logger
from pointcept.engines.test import TEST


def main_worker(cfg):
    cfg = default_setup(cfg)

    # tester init
    weight_name = os.path.basename(cfg.weight).split(".")[0]
    logger = get_root_logger(log_file=os.path.join(cfg.save_path, "test-{}.log".format(weight_name)))
    logger.info("=> Loading config ...")
    logger.info(f"Save path: {cfg.save_path}")
    logger.info(f"Config:\n{cfg.pretty_text}")

    tester = TEST.build(cfg.test)
    # build model
    logger.info("=> Building model ...")
    model = build_model(cfg.model).cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")
    model = create_ddp_model(model.cuda(), broadcast_buffers=False, find_unused_parameters=False)

    # load checkpoint
    if os.path.isfile(cfg.weight):
        checkpoint = torch.load(cfg.weight)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for name, value in state_dict.items():
            if name.startswith("module."):
                if comm.get_world_size() == 1:
                    name = name[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    name = "module." + name  # xxx.xxx -> module.xxx.xxx
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict, strict=True)
        logger.info("=> Loaded weight '{}' (epoch {})".format(cfg.weight, checkpoint['epoch']))
        cfg.test_epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))

    # build dataset
    logger.info("=> Building test dataset & dataloader ...")
    test_dataset = build_dataset(cfg.data.test)
    if comm.get_world_size() > 1:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        test_sampler = None
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.batch_size_test_per_gpu,
                                              shuffle=False,
                                              num_workers=cfg.batch_size_test_per_gpu,
                                              pin_memory=True,
                                              sampler=test_sampler,
                                              collate_fn=tester.collate_fn)
    tester(cfg, test_loader, model)


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
