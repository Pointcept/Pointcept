"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import os
import numpy as np
import torch


from uuid import uuid4

import pointcept.utils.comm as comm
from collections import OrderedDict

from pointcept.engines.hooks.evaluator import HookBase
from pointcept.engines.hooks.evaluator import HOOKS

from pointcept.utils.comm import is_main_process
from pointcept.models.sgiFormer.utils.data import process_label, process_instance
from pointcept.utils.misc import make_dirs
from configs._base_.dataset.scannetpp import CLASS_LABELS_PP, INST_LABELS_PP
import json

# Custom Hook


@HOOKS.register_module()
class SPInsEvaluator(HookBase):
    def __init__(
        self,
        segment_ignore_index=(-1,),
        semantic_ignore_index=(-1,),
        instance_ignore_index=-1,
        use_eval_train=False,
        write_cls_ap=False,
        enable_write_results=False,
    ):
        self.segment_ignore_index = segment_ignore_index
        self.semantic_ignore_index = semantic_ignore_index
        self.instance_ignore_index = instance_ignore_index

        self.valid_class_names = None  # update in before train
        self.overlaps = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
        self.min_region_sizes = 100
        self.distance_threshes = float("inf")
        self.distance_confs = -float("inf")
        self.enable_write_results = enable_write_results
        self.use_eval_train = use_eval_train
        self.write_cls_ap = write_cls_ap

    def before_train(self):
        self.valid_class_names = [
            self.trainer.cfg.data.names[i]
            for i in range(self.trainer.cfg.data.num_classes)
            if i not in self.segment_ignore_index
        ]
        if self.trainer.cfg.dataset_type == "ScanNet200SpDataset":
            assert (
                self.trainer.cfg.data.names[1] == "chair"
                and self.trainer.cfg.data.names[2] == "floor"
                and self.valid_class_names[0] == "floor"
            )
            self.valid_class_names[0] = "chair"

    def after_epoch(self):
        torch.cuda.empty_cache()
        self.eval()
        if self.use_eval_train and self.trainer.train_eval_loader is not None:
            self.eval_train()

    def associate_instances(self, pred, segment, instance):
        segment = segment.cpu().numpy()
        instance = instance.cpu().numpy()
        void_mask = np.in1d(segment, self.semantic_ignore_index)

        assert (
            pred["pred_classes"].shape[0]
            == pred["pred_scores"].shape[0]
            == pred["pred_masks"].shape[0]
        )
        if pred["pred_masks"].size > 0:  # Empty predictions
            assert pred["pred_masks"].shape[1] == segment.shape[0] == instance.shape[0]

        # get gt instances
        gt_instances = dict()
        for name in self.valid_class_names:
            gt_instances[name] = []
        instance_ids, idx, counts = np.unique(
            instance, return_index=True, return_counts=True
        )
        segment_ids = segment[idx]
        for i in range(len(instance_ids)):
            if instance_ids[i] == self.instance_ignore_index:
                continue
            if segment_ids[i] in self.semantic_ignore_index:
                continue
            gt_inst = dict()
            gt_inst["instance_id"] = instance_ids[i]
            gt_inst["segment_id"] = segment_ids[i]
            gt_inst["dist_conf"] = 0.0
            gt_inst["med_dist"] = -1.0
            gt_inst["vert_count"] = counts[i]
            gt_inst["matched_pred"] = []
            gt_instances[self.valid_class_names[segment_ids[i]]].append(gt_inst)

        # get pred instances and associate with gt
        pred_instances = dict()
        for name in self.valid_class_names:
            pred_instances[name] = []
        instance_id = 0
        for i in range(len(pred["pred_classes"])):
            if pred["pred_classes"][i] in self.semantic_ignore_index:
                continue
            pred_inst = dict()
            pred_inst["uuid"] = uuid4()
            pred_inst["instance_id"] = instance_id
            pred_inst["segment_id"] = pred["pred_classes"][i]
            pred_inst["confidence"] = pred["pred_scores"][i]
            pred_inst["mask"] = np.not_equal(pred["pred_masks"][i], 0)
            pred_inst["vert_count"] = np.count_nonzero(pred_inst["mask"])
            pred_inst["void_intersection"] = np.count_nonzero(
                np.logical_and(void_mask, pred_inst["mask"])
            )
            if pred_inst["vert_count"] < self.min_region_sizes:
                continue  # skip if empty
            segment_name = self.valid_class_names[pred_inst["segment_id"]]
            matched_gt = []
            for gt_idx, gt_inst in enumerate(gt_instances[segment_name]):
                intersection = np.count_nonzero(
                    np.logical_and(
                        instance == gt_inst["instance_id"], pred_inst["mask"]
                    )
                )
                if intersection > 0:
                    gt_inst_ = gt_inst.copy()
                    pred_inst_ = pred_inst.copy()
                    gt_inst_["intersection"] = intersection
                    pred_inst_["intersection"] = intersection
                    matched_gt.append(gt_inst_)
                    gt_inst["matched_pred"].append(pred_inst_)
            pred_inst["matched_gt"] = matched_gt
            pred_instances[segment_name].append(pred_inst)
            instance_id += 1
        return gt_instances, pred_instances

    def evaluate_matches(self, scenes):
        overlaps = self.overlaps
        min_region_sizes = [self.min_region_sizes]
        dist_threshes = [self.distance_threshes]
        dist_confs = [self.distance_confs]

        # results: class x overlap
        ap_table = np.zeros(
            (len(dist_threshes), len(self.valid_class_names), len(overlaps)), float
        )
        pr_rc = np.zeros((2, len(self.valid_class_names), len(overlaps)), float)
        for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
            zip(min_region_sizes, dist_threshes, dist_confs)
        ):
            for oi, overlap_th in enumerate(overlaps):
                pred_visited = {}
                for scene in scenes:
                    for _ in scene["pred"]:
                        for label_name in self.valid_class_names:
                            for p in scene["pred"][label_name]:
                                if "uuid" in p:
                                    pred_visited[p["uuid"]] = False
                for li, label_name in enumerate(self.valid_class_names):
                    y_true = np.empty(0)
                    y_score = np.empty(0)
                    hard_false_negatives = 0
                    has_gt = False
                    has_pred = False
                    for scene in scenes:
                        pred_instances = scene["pred"][label_name]
                        gt_instances = scene["gt"][label_name]
                        # filter groups in ground truth
                        gt_instances = [
                            gt
                            for gt in gt_instances
                            if gt["vert_count"] >= min_region_size
                            and gt["med_dist"] <= distance_thresh
                            and gt["dist_conf"] >= distance_conf
                        ]
                        if gt_instances:
                            has_gt = True
                        if pred_instances:
                            has_pred = True

                        cur_true = np.ones(len(gt_instances))
                        cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                        cur_match = np.zeros(len(gt_instances), dtype=bool)
                        # collect matches
                        for gti, gt in enumerate(gt_instances):
                            found_match = False
                            for pred in gt["matched_pred"]:
                                # greedy assignments
                                if pred_visited[pred["uuid"]]:
                                    continue
                                overlap = float(pred["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - pred["intersection"]
                                )
                                if overlap > overlap_th:
                                    confidence = pred["confidence"]
                                    # if already have a prediction for this gt,
                                    # the prediction with the lower score is automatically a false positive
                                    if cur_match[gti]:
                                        max_score = max(cur_score[gti], confidence)
                                        min_score = min(cur_score[gti], confidence)
                                        cur_score[gti] = max_score
                                        # append false positive
                                        cur_true = np.append(cur_true, 0)
                                        cur_score = np.append(cur_score, min_score)
                                        cur_match = np.append(cur_match, True)
                                    # otherwise set score
                                    else:
                                        found_match = True
                                        cur_match[gti] = True
                                        cur_score[gti] = confidence
                                        pred_visited[pred["uuid"]] = True
                            if not found_match:
                                hard_false_negatives += 1
                        # remove non-matched ground truth instances
                        cur_true = cur_true[cur_match]
                        cur_score = cur_score[cur_match]

                        # collect non-matched predictions as false positive
                        for pred in pred_instances:
                            found_gt = False
                            for gt in pred["matched_gt"]:
                                overlap = float(gt["intersection"]) / (
                                    gt["vert_count"]
                                    + pred["vert_count"]
                                    - gt["intersection"]
                                )
                                if overlap > overlap_th:
                                    found_gt = True
                                    break
                            if not found_gt:
                                num_ignore = pred["void_intersection"]
                                for gt in pred["matched_gt"]:
                                    if gt["segment_id"] in self.semantic_ignore_index:
                                        num_ignore += gt["intersection"]
                                    # TODO:
                                    # if gt["instance_id"] < 0:
                                    #     num_ignore += gt["intersection"]
                                    # small ground truth instances
                                    if (
                                        gt["vert_count"] < min_region_size
                                        or gt["med_dist"] > distance_thresh
                                        or gt["dist_conf"] < distance_conf
                                    ):
                                        num_ignore += gt["intersection"]
                                proportion_ignore = (
                                    float(num_ignore) / pred["vert_count"]
                                )
                                # if not ignored append false positive
                                if proportion_ignore <= overlap_th:
                                    cur_true = np.append(cur_true, 0)
                                    confidence = pred["confidence"]
                                    cur_score = np.append(cur_score, confidence)

                        # append to overall results
                        y_true = np.append(y_true, cur_true)
                        y_score = np.append(y_score, cur_score)

                    # compute average precision
                    if has_gt and has_pred:
                        # compute precision recall curve first

                        # sorting and cumsum
                        score_arg_sort = np.argsort(y_score)
                        y_score_sorted = y_score[score_arg_sort]
                        y_true_sorted = y_true[score_arg_sort]
                        y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                        # unique thresholds
                        (thresholds, unique_indices) = np.unique(
                            y_score_sorted, return_index=True
                        )
                        num_prec_recall = len(unique_indices) + 1

                        # prepare precision recall
                        num_examples = len(y_score_sorted)
                        # https://github.com/ScanNet/ScanNet/pull/26
                        # all predictions are non-matched but also all of them are ignored and not counted as FP
                        # y_true_sorted_cumsum is empty
                        # num_true_examples = y_true_sorted_cumsum[-1]
                        num_true_examples = (
                            y_true_sorted_cumsum[-1]
                            if len(y_true_sorted_cumsum) > 0
                            else 0
                        )
                        precision = np.zeros(num_prec_recall)
                        recall = np.zeros(num_prec_recall)

                        # deal with the first point
                        y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                        # deal with remaining
                        for idx_res, idx_scores in enumerate(unique_indices):
                            cumsum = y_true_sorted_cumsum[idx_scores - 1]
                            tp = num_true_examples - cumsum
                            fp = num_examples - idx_scores - tp
                            fn = cumsum + hard_false_negatives
                            p = float(tp) / (tp + fp)
                            r = float(tp) / (tp + fn)
                            precision[idx_res] = p
                            recall[idx_res] = r

                        # first point in curve is artificial
                        precision[-1] = 1.0
                        recall[-1] = 0.0

                        # compute optimal precision and recall, based on f1_score
                        f1_score = (
                            2 * precision * recall / (precision + recall + 0.0001)
                        )
                        f1_argmax = f1_score.argmax()
                        best_pr = precision[f1_argmax]
                        best_rc = recall[f1_argmax]

                        # compute average of precision-recall curve
                        recall_for_conv = np.copy(recall)
                        recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                        recall_for_conv = np.append(recall_for_conv, 0.0)

                        stepWidths = np.convolve(
                            recall_for_conv, [-0.5, 0, 0.5], "valid"
                        )
                        # integrate is now simply a dot product
                        ap_current = np.dot(precision, stepWidths)

                    elif has_gt:
                        ap_current = 0.0
                        best_pr = 0
                        best_rc = 0
                    else:
                        ap_current = float("nan")
                        best_pr = float("nan")
                        best_rc = float("nan")
                    ap_table[di, li, oi] = ap_current
                    pr_rc[0, li, oi] = best_pr
                    pr_rc[1, li, oi] = best_rc
        d_inf = 0
        o50 = np.where(np.isclose(self.overlaps, 0.5))
        o25 = np.where(np.isclose(self.overlaps, 0.25))
        oAllBut25 = np.where(np.logical_not(np.isclose(self.overlaps, 0.25)))
        ap_scores = dict()
        ap_scores["all_ap"] = np.nanmean(ap_table[d_inf, :, oAllBut25])
        ap_scores["all_ap_50%"] = np.nanmean(ap_table[d_inf, :, o50])
        ap_scores["all_ap_25%"] = np.nanmean(ap_table[d_inf, :, o25])
        ap_scores["all_prec_50%"] = np.nanmean(pr_rc[0, :, o50])
        ap_scores["all_rec_50%"] = np.nanmean(pr_rc[1, :, o50])
        ap_scores["classes"] = {}
        for li, label_name in enumerate(self.valid_class_names):
            ap_scores["classes"][label_name] = {}
            ap_scores["classes"][label_name]["ap"] = np.average(
                ap_table[d_inf, li, oAllBut25]
            )
            ap_scores["classes"][label_name]["ap50%"] = np.average(
                ap_table[d_inf, li, o50]
            )
            ap_scores["classes"][label_name]["ap25%"] = np.average(
                ap_table[d_inf, li, o25]
            )
            ap_scores["classes"][label_name]["prec50%"] = np.average(pr_rc[0, li, o50])
            ap_scores["classes"][label_name]["rec50%"] = np.average(pr_rc[1, li, o50])
        return ap_scores

    def print_results(self, ap_scores):
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_prec_50 = ap_scores["all_prec_50%"]
        all_rec_50 = ap_scores["all_rec_50%"]
        loss_avg = self.trainer.storage.history("val_loss").avg

        sep = ""
        col1 = ":"
        lineLen = 66
        self.trainer.logger.info("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>10}".format("AP") + sep
        line += "{:>10}".format("AP_50%") + sep
        line += "{:>10}".format("AP_25%") + sep
        line += "{:>10}".format("Prec_50%") + sep
        line += "{:>10}".format("Rec_50%") + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            prec_50 = ap_scores["classes"][label_name]["prec50%"]
            rec_50 = ap_scores["classes"][label_name]["rec50%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>10.3f}".format(ap) + sep
            line += sep + "{:>10.3f}".format(ap_50) + sep
            line += sep + "{:>10.3f}".format(ap_25) + sep
            line += sep + "{:>10.3f}".format(prec_50) + sep
            line += sep + "{:>10.3f}".format(rec_50) + sep
            self.trainer.logger.info(line)
            if self.write_cls_ap:
                self.trainer.wandb.log({f"val_AP_cls/{i}-{label_name}": ap})
                self.trainer.wandb.log({f"val_AP25_cls/{i}-{label_name}": ap_25})
                self.trainer.wandb.log({f"val_AP50_cls/{i}-{label_name}": ap_50})

        self.trainer.logger.info("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>10.3f}".format(all_ap) + sep
        line += "{:>10.3f}".format(all_ap_50) + sep
        line += "{:>10.3f}".format(all_ap_25) + sep
        line += "{:>10.3f}".format(all_prec_50) + sep
        line += "{:>10.3f}".format(all_rec_50) + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)
        self.trainer.wandb.log({"val/loss": loss_avg})
        self.trainer.wandb.log({"val/mAP": all_ap})
        self.trainer.wandb.log({"val/AP50": all_ap_50})
        self.trainer.wandb.log({"val/AP25": all_ap_25})

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.val_loader):
            key = "offset" if "offset" in input_dict else next(iter(input_dict))
            # currently only support bs 1 for each GPU
            assert len(input_dict[key]) == 1
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)

            loss = output_dict["loss"]

            # map to origin
            assert ("origin_segment" and "origin_instance") in input_dict.keys()
            instance = process_instance(
                input_dict["origin_instance"].clone(),
                input_dict["origin_segment"].clone(),
                self.segment_ignore_index,
            )
            segment = process_label(
                input_dict["origin_segment"].clone(), self.segment_ignore_index
            )

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))
            if self.enable_write_results:
                self.write_results(output_dict, input_dict)

            self.trainer.storage.put_scalar("val_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1, max_iter=len(self.trainer.val_loader), loss=loss.item()
                )
            )
            torch.cuda.empty_cache()

        loss_avg = self.trainer.storage.history("val_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        self.print_results(ap_scores)
        current_epoch = self.trainer.epoch + 1

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("val/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(
                "val/mAP", ap_scores["all_ap"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/AP50", ap_scores["all_ap_50%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/AP25", ap_scores["all_ap_25%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/Prec50", ap_scores["all_prec_50%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "val/Rec50", ap_scores["all_rec_50%"], current_epoch
            )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        # save for saver
        self.trainer.comm_info["current_metric_value"] = ap_scores["all_ap_50%"]
        # save for saver
        self.trainer.comm_info["current_metric_name"] = "AP50"

    def after_train(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}".format("AP50", self.trainer.best_metric_value)
        )

    def write_results(self, model_output, input_dict):
        current_epoch = self.trainer.epoch + 1
        scores = model_output["pred_scores"]
        masks = model_output["pred_masks"]
        classes = self.map_predict_classes_to_segment_classes(
            model_output["pred_classes"]
        )
        scan_name = input_dict["name"][0]
        mask_dir = "predicted_masks"

        root_dir = os.path.join(
            self.trainer.cfg.save_path, "result", str(current_epoch), "submit"
        )
        mask_dir = os.path.join(root_dir, mask_dir)
        make_dirs(mask_dir)
        result_file_path = os.path.join(root_dir, f"{scan_name}.txt")
        with open(result_file_path, "w") as result_file:
            for i, (score, mask, cls) in enumerate(zip(scores, masks, classes)):
                binary_mask = mask.astype(np.uint8)
                rle = self.rle_encode(binary_mask)

                mask_filename = f"{scan_name}_{i:03d}.json"
                mask_file_path = os.path.join(mask_dir, mask_filename)
                with open(mask_file_path, "w") as mask_file:
                    json.dump(rle, mask_file)

                # Write the instance's information to the results file
                relative_path = os.path.join("predicted_masks", mask_filename)
                result_file.write(f"{relative_path} {cls} {score:.3f}\n")
        save_pred_dir = os.path.join(
            self.trainer.cfg.save_path, "result", str(current_epoch), scan_name
        )
        os.makedirs(save_pred_dir, exist_ok=True)
        np.save(os.path.join(save_pred_dir, "pred_scores.npy"), np.array(scores))
        np.save(os.path.join(save_pred_dir, "pred_masks.npy"), np.array(masks))
        np.save(os.path.join(save_pred_dir, "pred_classes.npy"), np.array(classes))

    def rle_encode(self, mask):
        length = mask.shape[0]
        mask = np.concatenate([[0], mask, [0]])
        runs = np.where(mask[1:] != mask[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        counts = " ".join(str(x) for x in runs)
        rle = dict(length=length, counts=counts)
        return rle

    def map_predict_classes_to_segment_classes(self, predicted_classes):
        text = [INST_LABELS_PP[idx] for idx in predicted_classes]
        labels = [CLASS_LABELS_PP.index(c) for c in text]

        return np.array(labels)

    def eval_train(self):
        self.trainer.logger.info(
            ">>>>>>>>>>>>>>>> Start Evaluation Train Eval >>>>>>>>>>>>>>>>"
        )
        self.trainer.model.eval()
        scenes = []
        for i, input_dict in enumerate(self.trainer.train_eval_loader):
            # assert (
            #     len(input_dict["offset"]) == 1
            # )  # currently only support bs 1 for each GPU
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output_dict = self.trainer.model(input_dict)
                torch.cuda.empty_cache()

            loss = output_dict["loss"]

            # map to origin
            assert ("origin_segment" and "origin_instance") in input_dict.keys()
            instance = process_instance(
                input_dict["origin_instance"].clone(),
                input_dict["origin_segment"].clone(),
                self.segment_ignore_index,
            )
            segment = input_dict["origin_segment"]

            gt_instances, pred_instance = self.associate_instances(
                output_dict, segment, instance
            )
            scenes.append(dict(gt=gt_instances, pred=pred_instance))

            self.trainer.storage.put_scalar("trainEval_loss", loss.item())
            self.trainer.logger.info(
                "Test: [{iter}/{max_iter}] "
                "Loss {loss:.4f} ".format(
                    iter=i + 1,
                    max_iter=len(self.trainer.train_eval_loader),
                    loss=loss.item(),
                )
            )
            torch.cuda.empty_cache()

        loss_avg = self.trainer.storage.history("trainEval_loss").avg
        comm.synchronize()
        scenes_sync = comm.gather(scenes, dst=0)
        scenes = [scene for scenes_ in scenes_sync for scene in scenes_]
        ap_scores = self.evaluate_matches(scenes)
        self.print_results_train(ap_scores)
        current_epoch = self.trainer.epoch + 1

        if self.trainer.writer is not None:
            self.trainer.writer.add_scalar("trainEval/loss", loss_avg, current_epoch)
            self.trainer.writer.add_scalar(
                "trainEval/mAP", ap_scores["all_ap"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "trainEval/AP50", ap_scores["all_ap_50%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "trainEval/AP25", ap_scores["all_ap_25%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "trainEval/Prec50", ap_scores["all_prec_50%"], current_epoch
            )
            self.trainer.writer.add_scalar(
                "trainEval/Rec50", ap_scores["all_rec_50%"], current_epoch
            )
        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")

    def print_results_train(self, ap_scores):
        all_ap = ap_scores["all_ap"]
        all_ap_50 = ap_scores["all_ap_50%"]
        all_ap_25 = ap_scores["all_ap_25%"]
        all_prec_50 = ap_scores["all_prec_50%"]
        all_rec_50 = ap_scores["all_rec_50%"]
        loss_avg = self.trainer.storage.history("trainEval_loss").avg

        sep = ""
        col1 = ":"
        lineLen = 66
        self.trainer.logger.info("#" * lineLen)
        line = ""
        line += "{:<15}".format("what") + sep + col1
        line += "{:>10}".format("AP") + sep
        line += "{:>10}".format("AP_50%") + sep
        line += "{:>10}".format("AP_25%") + sep
        line += "{:>10}".format("Prec_50%") + sep
        line += "{:>10}".format("Rec_50%") + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)

        for i, label_name in enumerate(self.valid_class_names):
            ap = ap_scores["classes"][label_name]["ap"]
            ap_50 = ap_scores["classes"][label_name]["ap50%"]
            ap_25 = ap_scores["classes"][label_name]["ap25%"]
            prec_50 = ap_scores["classes"][label_name]["prec50%"]
            rec_50 = ap_scores["classes"][label_name]["rec50%"]
            line = "{:<15}".format(label_name) + sep + col1
            line += sep + "{:>10.3f}".format(ap) + sep
            line += sep + "{:>10.3f}".format(ap_50) + sep
            line += sep + "{:>10.3f}".format(ap_25) + sep
            line += sep + "{:>10.3f}".format(prec_50) + sep
            line += sep + "{:>10.3f}".format(rec_50) + sep
            self.trainer.logger.info(line)
            self.trainer.wandb.log({f"trainEval_AP_cls/{i}-{label_name}": ap})
            self.trainer.wandb.log({f"trainEval_AP25_cls/{i}-{label_name}": ap_25})
            self.trainer.wandb.log({f"trainEval_AP50_cls/{i}-{label_name}": ap_50})

        self.trainer.logger.info("-" * lineLen)
        line = "{:<15}".format("average") + sep + col1
        line += "{:>10.3f}".format(all_ap) + sep
        line += "{:>10.3f}".format(all_ap_50) + sep
        line += "{:>10.3f}".format(all_ap_25) + sep
        line += "{:>10.3f}".format(all_prec_50) + sep
        line += "{:>10.3f}".format(all_rec_50) + sep
        self.trainer.logger.info(line)
        self.trainer.logger.info("#" * lineLen)
        self.trainer.wandb.log({"trainEval/loss": loss_avg})
        self.trainer.wandb.log({"trainEval/mAP": all_ap})
        self.trainer.wandb.log({"trainEval/AP50": all_ap_50})
        self.trainer.wandb.log({"trainEval/AP25": all_ap_25})


@HOOKS.register_module()
class CustomCheckpointLoader(HookBase):
    def __init__(self, keywords="", replacement=None, strict=False, skip_key=None):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict
        self.skip_key = skip_key

    def before_train(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )

            # modified from gorilla
            # get model state_dict from checkpoint
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "net" in checkpoint:
                state_dict = checkpoint["net"]
            else:
                state_dict = checkpoint

            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict()
            for key, value in state_dict.items():
                if not key.startswith("module."):
                    if comm.get_world_size() >= 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                # Now all keys contain "module." no matter DDP or not.
                if self.keywords in key:
                    key = key.replace(self.keywords, self.replacement)
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
                if self.skip_key is not None and self.skip_key in key:
                    continue
                weight[key] = value

            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
