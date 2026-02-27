"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import subprocess
from pathlib import Path
from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from clearml import Task, InputModel

def fast_rsync(nas_path, local_path):
    """
    Optimized rsync for high-speed local transfer.
    -a: archive mode
    -u: update (skip files that are newer on receiver)
    -h: human-readable
    """
    nas_path = str(Path(nas_path).absolute())
    local_path = str(Path(local_path).absolute())
    
    print(f"🚀 [Data Sync] Synchronizing {nas_path} -> {local_path}...")
    os.makedirs(local_path, exist_ok=True)
    
    # We use a trailing slash on nas_path to copy contents, not the folder itself
    cmd = f"rsync -au --info=progress2 {nas_path}/ {local_path}/"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"✅ [Data Sync] Transfer to {local_path} complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ [Data Sync] Error during rsync: {e}")
        raise

def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

def main():
    task: Task = Task.init(
            output_uri=True,
            auto_connect_frameworks=False,
        )
    args = default_argument_parser().parse_args()
    config_path = task.connect_configuration(
        args.config_file,
        name="Pretraining configuration",
        description="Configuration for pretraining.",
    )
    cfg = default_config_parser(config_path, args.options)

    if getattr(args, "input_model", None):
        print("Loading model from input model: ", args.input_model)
        original_model = InputModel(model_id=args.input_model)
        task.connect(original_model)
        checkpoint_path = original_model.get_weights(
            raise_on_error=False, force_download=False, extract_archive=False
        )
        cfg.weight = checkpoint_path

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
