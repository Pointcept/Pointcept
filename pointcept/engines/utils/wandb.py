import wandb
import numpy as np


class Wandb():
    def __init__(self,
                 enabled,
                 project_name,
                 logger,
                 tags=[],
                 cfg={},
                 use_step_logging=False,
                 print_every=10
                 ):
        self.global_step = 0
        self.epoch = 0
        self.enabled = enabled
        self.use_step_logging = use_step_logging
        self.logger = logger
        self.project_name = project_name
        self.tags = tags
        self.cfg = cfg
        self.running_metrics = {}
        self.print_every = print_every

    def set_epoch(self, epoch):
        self.epoch = epoch + 1

    def set_global_step(self, global_step):
        self.global_step = global_step+1
        if self.global_step % self.print_every == 0:
            self.log_running_metrics()

    def init(self):
        if not self.enabled:
            self.logger.info("Wandb not enabled")
            return None
        wandb.init(project=self.project_name,
                   tags=self.tags,
                   config=self.cfg,
                   name=self.cfg.run_name)
        wandb.log({"Test/Log": 500}, step=0)

    def log(self, data_dict):
        if not self.enabled:
            return
        step = self.global_step if self.use_step_logging else self.epoch
        wandb.log(data_dict, step=step)

    def add_to_running_metrics(self, key, val):
        current_values = self.running_metrics.get(key, np.array([]))
        next_values = np.append(current_values, val)
        self.running_metrics[key] = next_values

    def log_running_metrics(self):
        for key, values in self.running_metrics.items():
            mean_value = np.mean(values)
            self.log({f"freq/{key}": mean_value})
        self.running_metrics = {}
