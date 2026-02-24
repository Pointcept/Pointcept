"""
Misc Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import sys
import glob
import os
import shutil
import time
import gc
import wandb
import clearml
from clearml import Logger, Task, OutputModel
import torch
import torch.utils.data
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
if sys.version_info >= (3, 10):
    from collections.abc import Sequence
else:
    from collections import Sequence
from pointcept.utils.timer import Timer
from pointcept.utils.comm import is_main_process, synchronize
from pointcept.utils.cache import shared_dict
from pointcept.utils.scheduler import CosineScheduler
import pointcept.utils.comm as comm

from .default import HookBase
from .builder import HOOKS


@HOOKS.register_module()
class PCAVisualizationHook(HookBase):
    def __init__(self, interval=500, save_path="visuals/pca_bev", x_lim=(-50, 50), y_lim=(0, 100)):
        self.interval = interval
        self.save_path = save_path
        self.x_lim = x_lim  # Lateral (left/right)
        self.y_lim = y_lim  # Longitudinal (forward)
        self._total_iter = 0


    def before_step(self):
        curr_iter = self.trainer.comm_info.get("iter", 0)
        # Inject the flag into the data_dict so the model knows to return features
        if curr_iter % self.interval == 0:
            self.trainer.comm_info.get("input_dict")["visualize"] = True

    def after_step(self):
        curr_iter = self.trainer.comm_info.get("iter", 0)
        rank = self.trainer.comm_info.get("rank", 0)
        self._total_iter += 1
        # Check if the flag was set and we are on master
        if curr_iter % self.interval == 0 and rank == 0:
            # Pointcept stores the model's return value in trainer.comm_info["model_output_dict"]
            output_dict = self.trainer.comm_info.get("model_output_dict", {})
            
            if "student_feat" in output_dict:
                self._plot_debug_frame(self._total_iter, output_dict)
            
            # Clean up the flag for the next step
            if "visualize" in self.trainer.comm_info.get("input_dict", {}):
                del self.trainer.comm_info.get("input_dict")["visualize"]

    @torch.no_grad()
    def _plot_debug_frame(self, curr_iter, output_dict):
        # 1. Extraction: Get Student and Teacher data
        # Use .float() to convert BFloat16 to Float32 before calling .cpu().numpy()
        s_feat = output_dict["student_feat"].float().cpu().numpy()
        s_coord = output_dict["student_feat_coord"].float().cpu().numpy()
        t_feat = output_dict["teacher_feat"].float().cpu().numpy()
        t_coord = output_dict["teacher_feat_coord"].float().cpu().numpy()
    
        input_dict = self.trainer.comm_info.get("input_dict")       
        num_clusters = 8

        # --- Feature Processing Function ---
        def process_features(feat):
            # PCA Projection
            pca = PCA(n_components=3)
            rgb = pca.fit_transform(feat)
            rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-8)
            
            # K-Means
            kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
            clusters = kmeans.fit_predict(feat)
            return rgb, clusters

        s_rgb, s_clusters = process_features(s_feat)
        t_rgb, t_clusters = process_features(t_feat)

        # 2. Image Context
        img_paths = input_dict.get("image_path", [])
        img = None
        if len(img_paths) > 0 and os.path.exists(img_paths[0]):
            img = cv2.cvtColor(cv2.imread(img_paths[0]), cv2.COLOR_BGR2RGB)

        # 3. Create 2x3 Grid (Row 0: Student | Row 1: Teacher)
        # Col 0: Camera | Col 1: PCA | Col 2: K-Means
        fig, axes = plt.subplots(2, 3, figsize=(30, 18), facecolor='black')
        
        # --- Helper for plotting columns ---
        def plot_row(row_idx, coord, rgb, clusters, label_prefix):
            # Column 0: Camera (Plot on both rows for alignment or leave bottom empty)
            if img is not None:
                axes[row_idx, 0].imshow(img)
                axes[row_idx, 0].set_title(f"{label_prefix} Camera Ref", color='white', fontsize=15)
            axes[row_idx, 0].axis('off')

            # Column 1: PCA BEV
            axes[row_idx, 1].scatter(-coord[:, 1], coord[:, 0], c=rgb, s=8, alpha=0.8)
            self._format_bev_ax(axes[row_idx, 1], curr_iter, f"{label_prefix} PCA (RGB)")

            # Column 2: K-Means BEV
            axes[row_idx, 2].scatter(-coord[:, 1], coord[:, 0], c=clusters, s=8, cmap='tab10', alpha=0.8)
            self._format_bev_ax(axes[row_idx, 2], curr_iter, f"{label_prefix} K-Means (K={num_clusters})")

        # Plot Student Row
        plot_row(0, s_coord, s_rgb, s_clusters, "STUDENT")
        # Plot Teacher Row
        plot_row(1, t_coord, t_rgb, t_clusters, "TEACHER")

        # 4. Save and Report
        save_dir = os.path.join(self.trainer.writer.logdir, self.save_path)
        save_path = os.path.join(save_dir, f"debug_{curr_iter:07d}.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight', facecolor='black')

        if Task.current_task() is not None:
            logger = Logger.current_logger()
            rgb_img = Image.open(save_path).convert("RGB")
            logger.report_image("Visualization", "Student_Teacher_Comparison", image=rgb_img, iteration=curr_iter)
        
        plt.close(fig)

    def _format_bev_ax(self, ax, curr_iter, title_suffix):
        """Helper to keep BEV axes consistent."""
        ax.set_xlim([-self.x_lim[1], -self.x_lim[0]]) # Flipped lateral limits
        ax.set_ylim(self.y_lim)
        ax.set_facecolor('#0a0a0a')
        ax.set_title(f"Iter {curr_iter} | {title_suffix}", color='white', fontsize=15)
        ax.set_xlabel("Right <--- Lateral (m) ---> Left", color='white')
        ax.set_ylabel("Longitudinal (Forward m)", color='white')
        ax.grid(color='gray', linestyle='--', alpha=0.2)
        ax.tick_params(colors='white')

@HOOKS.register_module()
class IterationTimer(HookBase):
    def __init__(self, warmup_iter=1):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def before_train(self):
        self._start_time = time.perf_counter()
        _remain_epoch = self.trainer.max_epoch - self.trainer.start_epoch
        self._remain_iter = _remain_epoch * len(self.trainer.train_loader)

    def before_epoch(self):
        self._iter_timer.reset()

    def before_step(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def after_step(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


@HOOKS.register_module()
class InformationWriter(HookBase):
    def __init__(self):
        self.curr_iter = 0
        self.model_output_keys = []
        self.clearml_logger = Logger.current_logger()
    def before_train(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        if self.trainer.writer is not None and self.trainer.cfg.enable_wandb:
            wandb.define_metric("params/*", step_metric="Iter")
            wandb.define_metric("train_batch/*", step_metric="Iter")
            wandb.define_metric("train/*", step_metric="Epoch")

    def before_step(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def after_step(self):
        if "model_output_dict" in self.trainer.comm_info.keys():
            model_output_dict = self.trainer.comm_info["model_output_dict"]
            # Convert to set and subtract the keys you want to exclude
            self.model_output_keys = set(model_output_dict.keys()) - {"student_feat", "student_feat_coord", "teacher_feat", "teacher_feat_coord"}

            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, model_output_dict[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)
        self.trainer.logger.info(self.trainer.comm_info["iter_info"])
        self.trainer.comm_info["iter_info"] = ""  # reset iter info
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/lr", lr, self.curr_iter)
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train_batch/" + key,
                    self.trainer.storage.history(key).val,
                    self.curr_iter,
                )
            if self.trainer.cfg.enable_wandb:

                wandb.log(
                    {"Iter": self.curr_iter, "params/lr": lr}, step=self.curr_iter
                )
                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Iter": self.curr_iter,
                            f"train_batch/{key}": self.trainer.storage.history(key).val,
                        },
                        step=wandb.run.step,
                    )
            self.clearml_logger.report_scalar(title="Iter", series="Iter", iteration=self.curr_iter, value=self.curr_iter)
            self.clearml_logger.report_scalar(title="params/lr", series="params/lr", iteration=self.curr_iter, value=lr)
            for key in self.model_output_keys:
                self.clearml_logger.report_scalar(title="train_batch/" + key, series="train_batch/" + key, iteration=self.curr_iter, value=self.trainer.storage.history(key).val)
    def after_epoch(self):
        epoch_info = "Train result: "
        for key in self.model_output_keys:
            epoch_info += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).avg
            )
        self.trainer.logger.info(epoch_info)
        if self.trainer.writer is not None:
            for key in self.model_output_keys:
                self.trainer.writer.add_scalar(
                    "train/" + key,
                    self.trainer.storage.history(key).avg,
                    self.trainer.epoch + 1,
                )

            if self.trainer.cfg.enable_wandb:

                for key in self.model_output_keys:
                    wandb.log(
                        {
                            "Epoch": self.trainer.epoch + 1,
                            f"train/{key}": self.trainer.storage.history(key).avg,
                        },
                        step=wandb.run.step,
                    )
            for key in self.model_output_keys:
                self.clearml_logger.report_scalar(title=f"train epoch/{key}", series=f"train epoch/{key}", iteration=self.trainer.epoch + 1, value=self.trainer.storage.history(key).avg)


@HOOKS.register_module()
class CheckpointSaver(HookBase):
    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save model last
        self.task = Task.current_task()
        self.task_name = self.task.name
    def after_epoch(self):
        if is_main_process():
            is_best = False
            if self.trainer.cfg.evaluate:
                current_metric_value = self.trainer.comm_info["current_metric_value"]
                current_metric_name = self.trainer.comm_info["current_metric_name"]
                if current_metric_value > self.trainer.best_metric_value:
                    self.trainer.best_metric_value = current_metric_value
                    is_best = True
                    self.trainer.logger.info(
                        "Best validation {} updated to: {:.4f}".format(
                            current_metric_name, current_metric_value
                        )
                    )
                self.trainer.logger.info(
                    "Currently Best {}: {:.4f}".format(
                        current_metric_name, self.trainer.best_metric_value
                    )
                )

            filename = os.path.join(
                self.trainer.cfg.save_path, "model", "model_last.pth"
            )
            self.trainer.logger.info("Saving checkpoint to: " + filename)
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": (
                        self.trainer.scaler.state_dict()
                        if self.trainer.cfg.enable_amp
                        else None
                    ),
                    "best_metric_value": self.trainer.best_metric_value,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
            if is_best:
                shutil.copyfile(
                    filename,
                    os.path.join(self.trainer.cfg.save_path, "model", "model_best.pth"),
                )
                om = OutputModel(
                    task=self.task, name=f"{self.task_name}_best", framework="pytorch"
                )

                om.update_weights(
                    weights_filename=filename, async_enable=False
                )
                print(f"Uploaded weights from {filename} to clearml (model ID: {om.id})")

                # Upload training configuration
                om.update_design(config_dict=self.trainer.cfg)
            if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0:
                shutil.copyfile(
                    filename,
                    os.path.join(
                        self.trainer.cfg.save_path,
                        "model",
                        f"epoch_{self.trainer.epoch + 1}.pth",
                    ),
                )


@HOOKS.register_module()
class CheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
                weights_only=False,
            )
            
            # Get current model state to check shapes
            current_model_dict = self.trainer.model.state_dict()
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            
            weight = OrderedDict()
            mismatched_keys = []

            for key, value in checkpoint["state_dict"].items():
                # Standardize key names (handling 'module.' prefix)
                if not key.startswith("module."):
                    key = "module." + key
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement, 1)
                if comm.get_world_size() == 1 and key.startswith("module."):
                    key = key[7:]

                # --- SHAPE CHECK LOGIC ---
                if key in current_model_dict:
                    if value.shape != current_model_dict[key].shape:
                        mismatched_keys.append(
                            f"{key} (Checkpoint: {list(value.shape)} vs Model: {list(current_model_dict[key].shape)})"
                        )
                        continue  # Skip this key, leaving model's random weights intact
                    weight[key] = value
                else:
                    # Key exists in checkpoint but not in current model
                    if self.strict:
                        weight[key] = value

            # Log warnings for skipped keys
            if mismatched_keys:
                self.trainer.logger.warning(
                    f"⚠️ Found {len(mismatched_keys)} keys with shape mismatches. "
                    f"These will remain randomized: {mismatched_keys}"
                )

            # Load the filtered weights
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            
            self.trainer.logger.info(f"Successfully loaded {len(weight)} tensors.")
            self.trainer.logger.info(f"Missing keys (not in checkpoint): {load_state_info[0]}")
            self.trainer.logger.info(f"Unexpected keys (not in model): {load_state_info[1]}")

            # --- Resume logic remains the same ---
            if self.trainer.cfg.resume:
                self.trainer.logger.info(f"Resuming train at eval epoch: {checkpoint['epoch']}")
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")


@HOOKS.register_module()
class PreciseEvaluator(HookBase):
    def __init__(self, test_last=False):
        self.test_last = test_last

    def after_train(self):
        from pointcept.engines.test import TESTERS

        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Precise Evaluation >>>>>>>>>>>>>>>>"
        )
        torch.cuda.empty_cache()
        cfg = self.trainer.cfg
        test_cfg = dict(cfg=cfg, model=self.trainer.model, **cfg.test)
        tester = TESTERS.build(test_cfg)
        if self.test_last:
            self.trainer.logger.info("=> Testing on model_last ...")
        else:
            self.trainer.logger.info("=> Testing on model_best ...")
            best_path = os.path.join(
                self.trainer.cfg.save_path, "model", "model_best.pth"
            )
            checkpoint = torch.load(best_path, weights_only=False)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if not key.startswith("module."):
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                weight[key] = value
            tester.model.load_state_dict(weight, strict=True)
        tester.test()


@HOOKS.register_module()
class DataCacheOperator(HookBase):
    def __init__(self, data_root, split):
        self.data_root = data_root
        self.split = split
        self.data_list = self.get_data_list()

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split))
        else:
            raise NotImplementedError
        return data_list

    def get_cache_name(self, data_path):
        data_name = data_path.replace(os.path.dirname(self.data_root), "")
        return "pointcept" + data_name.replace(os.path.sep, "-")

    def before_train(self):
        self.trainer.logger.info(
            f"=> Caching dataset: {self.data_root}, split: {self.split} ..."
        )
        if is_main_process():
            dataset = self.trainer.train_loader.dataset
            for i in range(len(dataset)):
                data_dict = dataset[i]
                name = data_dict["name"]
                shared_dict(f"Pointcept-{name}", data_dict)
        synchronize()


@HOOKS.register_module()
class RuntimeProfiler(HookBase):
    def __init__(
        self,
        forward=True,
        backward=True,
        interrupt=False,
        warm_up=2,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.forward = forward
        self.backward = backward
        self.interrupt = interrupt
        self.warm_up = warm_up
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import profile, record_function, ProfilerActivity

        for i, input_dict in enumerate(self.trainer.train_loader):
            if i == self.warm_up + 1:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            if self.forward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as forward_prof:
                    with record_function("model_inference"):
                        output_dict = self.trainer.model(input_dict)
            else:
                output_dict = self.trainer.model(input_dict)
            loss = output_dict["loss"]
            if self.backward:
                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                ) as backward_prof:
                    with record_function("model_inference"):
                        loss.backward()
            self.trainer.logger.info(f"Profile: [{i + 1}/{self.warm_up + 1}]")
        if self.forward:
            self.trainer.logger.info(
                "Forward profile: \n"
                + str(
                    forward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            forward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "forward_trace.json")
            )

        if self.backward:
            self.trainer.logger.info(
                "Backward profile: \n"
                + str(
                    backward_prof.key_averages().table(
                        sort_by=self.sort_by, row_limit=self.row_limit
                    )
                )
            )
            backward_prof.export_chrome_trace(
                os.path.join(self.trainer.cfg.save_path, "backward_trace.json")
            )
        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class RuntimeProfilerV2(HookBase):
    def __init__(
        self,
        interrupt=False,
        wait=1,
        warmup=1,
        active=10,
        repeat=1,
        sort_by="cuda_time_total",
        row_limit=30,
    ):
        self.interrupt = interrupt
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.sort_by = sort_by
        self.row_limit = row_limit

    def before_train(self):
        self.trainer.logger.info("Profiling runtime ...")
        from torch.profiler import (
            profile,
            record_function,
            ProfilerActivity,
            schedule,
            tensorboard_trace_handler,
        )

        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=tensorboard_trace_handler(self.trainer.cfg.save_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.start()
        for i, input_dict in enumerate(self.trainer.train_loader):
            if i >= (self.wait + self.warmup + self.active) * self.repeat:
                break
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with record_function("model_forward"):
                output_dict = self.trainer.model(input_dict)
                loss = output_dict["loss"]
            with record_function("model_backward"):
                loss.backward()
            prof.step()
            self.trainer.logger.info(
                f"Profile: [{i + 1}/{(self.wait + self.warmup + self.active) * self.repeat}]"
            )
        self.trainer.logger.info(
            "Profile: \n"
            + str(
                prof.key_averages().table(
                    sort_by=self.sort_by, row_limit=self.row_limit
                )
            )
        )
        prof.stop()

        if self.interrupt:
            sys.exit(0)


@HOOKS.register_module()
class WeightDecaySchedular(HookBase):
    def __init__(
        self,
        base_value=0.04,
        final_value=0.2,
    ):
        self.base_value = base_value
        self.final_value = final_value
        self.scheduler = None

    def before_train(self):
        curr_step = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.scheduler = CosineScheduler(
            base_value=self.base_value,
            final_value=self.final_value,
            total_iters=self.trainer.cfg.scheduler.total_steps,
        )
        self.scheduler.iter = curr_step

    def before_step(self):
        wd = self.scheduler.step()
        for param_group in self.trainer.optimizer.param_groups:
            param_group["weight_decay"] = wd
        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("params/wd", wd, self.scheduler.iter)


@HOOKS.register_module()
class GarbageHandler(HookBase):
    def __init__(self, interval=150, disable_auto=True, empty_cache=False):
        self.interval = interval
        self.disable_auto = disable_auto
        self.empty_cache = empty_cache
        self.iter = 1

    def before_train(self):
        if self.disable_auto:
            gc.disable()
            self.trainer.logger.info("Disable automatic garbage collection")

    def before_epoch(self):
        self.iter = 1

    def after_step(self):
        if self.iter % self.interval == 0:
            gc.collect()
            if self.empty_cache:
                torch.cuda.empty_cache()
            self.trainer.logger.info("Garbage collected")
        self.iter += 1

    def after_train(self):
        gc.collect()
        torch.cuda.empty_cache()
