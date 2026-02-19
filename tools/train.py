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
from clearml import Task

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

    # try:
    #     # --- LOCAL DATA SYNC LOGIC ---
    #     # Define your local scratch space (usually /tmp or /scratch)
    #     # Using /tmp/pointcept_data to keep the node clean
    #     base_local_path = "/tmp/pointcept_data"
        
    #     # We loop through the datasets in the config to sync each one
    #     # This handles your PercivDataset and MagnaDataset entries
    #     for dataset_cfg in cfg.data.train.datasets:
    #         nas_root = dataset_cfg.data_root
            
    #         # Create a unique local subfolder name based on the NAS path hash or name
    #         folder_name = os.path.basename(nas_root.strip("/"))
    #         local_root = os.path.join(base_local_path, folder_name)
            
    #         # Perform the sync (only on the main process to avoid collisions)
    #         fast_rsync(nas_root, local_root)
            
    #         # Crucial: Update the config to point to the LOCAL path
    #         dataset_cfg.data_root = local_root
    # except Exception as e:
    #     print(f"❌ [Data Sync] Failed to sync data localy: {e}. Using default paths.")
    #     pass
    # # -----------------------------

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
