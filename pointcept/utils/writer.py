from pathlib import Path
from typing import Union
from tensorboardX import SummaryWriter

from pointcept.utils.config import Config


class ExperimentWriter(object):
    def __init__(
        self,
        save_path: Union[str, Path],
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = None,
        wandb_entity: str = None,
        wandb_config: Union[dict, Config] = None,
        wandb_group: str = None,
        wandb_name: str = None,
        wandb_id: str = None,
    ):
        """Abstraction over the TensorBoard writer or wandb logger.

        Args:
            save_path (str | Path): Path to save the local experiment logs.
            use_tensorboard (bool, optional): Whether to use tensorboard. Defaults to True.
            use_wandb (bool, optional): Whether to use Weights & Biases. Defaults to False.
            wandb_project (str, optional): wandb project name. Defaults to None.
            wandb_entity (str, optional): wandb entity name. Defaults to None.
            wandb_config (dict, optional): hyperparameter config to log for the run. Defaults to None.
            wandb_group (str, optional): wandb group name. Defaults to None.
            wandb_name (str, optional): wandb run name, if not set, inferred from save_path. Defaults to None.
            wandb_id (str, optional): Set this to continue logging to a run / overwrite a run. Defaults to None.
        """
        assert (
            use_tensorboard or use_wandb
        ), "At least one of use_tensorboard or use_wandb must be True."
        if use_wandb:
            assert wandb_project, "wandb_project is required for logging to wandb."

        self.wandb = None
        if use_wandb:
            try:
                import wandb

                assert wandb_project, "wandb_project is required for logging to wandb."
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    config=wandb_config,
                    group=wandb_group,
                    name=wandb_name if wandb_name else Path(save_path).name,
                    dir=save_path,
                    id=wandb_id,
                )
                self.wandb = wandb
            except ImportError:
                raise Exception("Tried to use wandb logger but wandb is not installed.")

        self.tb_writer = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(save_path)

    def add_scalar(self, tag: str, val: float, step: float = None):
        if self.wandb:
            self.wandb.log({tag: val}, step=step)
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, val, step)

    def close(self):
        if self.wandb:
            self.wandb.finish()
        if self.tb_writer:
            self.tb_writer.close()
