"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import shutil
import argparse
import multiprocessing as mp
from importlib import import_module
from torch.nn.parallel import DistributedDataParallel


import pointcept.utils.comm as comm
from pointcept.utils.env import get_random_seed, set_seed
from pointcept.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model,** kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = None if seed is None else num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--num-machines", type=int, default=1, help="total number of machines"
    )
    parser.add_argument(
        "--machine-rank",
        type=int,
        default=0,
        help="the rank of this machine (unique per machine)",
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 **15 + 2** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 **14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    return parser


def default_config_parser(file_path, options):
    from pointcept.models.builder import MODELS    # 导入模块注册器
    # 1. Determine the real path of the original configuration file (source file)
    if os.path.isfile(file_path):
        real_config_path = file_path  # Source file: original config file (e.g., config/s3dis/xxx.py)
    else:
        sep = file_path.find("-")
        real_config_path = os.path.join(file_path[:sep], file_path[sep + 1:])
    # Verify the original configuration file exists
    if not os.path.isfile(real_config_path):
        raise FileNotFoundError(f"Original configuration file does not exist: {real_config_path}")

    # 2. Parse the original configuration first
    cfg = Config.fromfile(real_config_path)

    # 3. Merge options first (key modification: merge options in advance to get correct save_path)
    if options is not None:
        cfg.merge_from_dict(options)

    # 4. Determine the save path for the configuration file (target file) - now using merged save_path
    save_path = cfg.save_path
    os.makedirs(save_path, exist_ok=True)  # Ensure the save directory exists
    model_save_dir = os.path.join(save_path, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    config_save_path = os.path.join(save_path, "config.py")  # Target file: config.py in the save directory

    # 5. Copy the original configuration file (now using correct save_path)
    if not cfg.test_only:  # Only copy in training mode
        # Check if source and target files are the same (avoid SameFileError)
        if os.path.abspath(real_config_path) == os.path.abspath(config_save_path):
            print(f"Source configuration file is the same as the target path, skipping copy: {real_config_path}")
        else:
            if cfg.resume:  # Resume mode: copy only if target file does not exist
                if not os.path.exists(config_save_path):
                    shutil.copy2(real_config_path, config_save_path)
                    print(f"Resume mode: Copying original configuration file to: {config_save_path}")
                else:
                    print(f"Resume mode: Configuration file already exists, skipping copy: {config_save_path}")
            else:  # Non-resume mode: force copy (overwrite old file)
                shutil.copy2(real_config_path, config_save_path)
                print(f"Training mode: Copying original configuration file to: {config_save_path}")
    else:
        print("Testing mode: Skipping configuration file copy")

    # 6. Handle random seed
    if cfg.seed is None:
        cfg.seed = get_random_seed()
    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    # 7. Copy backbone file (now using correct save_path)
    try:
        backbone_type = cfg.model.backbone.type
        if not backbone_type:
            raise ValueError("model.backbone.type not found in config")
        backbone_cls = MODELS.get(backbone_type)
        if backbone_cls is None:
            raise KeyError(f"Backbone type {backbone_type} not found in MODELS registry")
        module_path = backbone_cls.__module__
        module = import_module(module_path)
        backbone_file_path = module.__file__  # Source file: original file where backbone is located
        if not backbone_file_path:
            raise FileNotFoundError(f"Failed to get module file corresponding to {backbone_type}")
        if backbone_file_path.endswith(".pyc"):
            backbone_file_path = backbone_file_path[:-1]  # Remove .pyc suffix
        dest_file = os.path.join(save_path, os.path.basename(backbone_file_path))  # Target file: file with the same name in save directory
        
        # Check if source and target files are the same
        if os.path.abspath(backbone_file_path) == os.path.abspath(dest_file):
            print(f"Source backbone file is the same as the target path, skipping copy: {backbone_file_path}")
        else:
            if not os.path.exists(dest_file):
                shutil.copy2(backbone_file_path, dest_file)
                print(f"Copied backbone module file to: {dest_file}")
            else:
                print(f"Backbone module file already exists, skipping copy: {dest_file}")
    except Exception as e:
        print(f"Warning: Failed to copy backbone module file, reason: {str(e)}")

    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    cfg.batch_size_test_per_gpu = (
        cfg.batch_size_test // world_size if cfg.batch_size_test is not None else 1
    )
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed + rank * cfg.num_worker_per_gpu
    set_seed(seed)
    return cfg