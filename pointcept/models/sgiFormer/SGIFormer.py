"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from pointcept.models.sgiFormer.utils.data import (
    process_label,
    process_instance,
    split_offset,
)

from ..builder import MODELS, build_model
from .utils import mask_matrix_nms


@MODELS.register_module("SGIFormer")
class SGIFormer(nn.Module):
    """SGIFormer"""

    def __init__(
        self,
        backbone,
        decoder=None,
        criterion=None,
        fix_module=[],
        semantic_num_classes=18,
        semantic_ignore_index=-1,
        segment_ignore_index=(-1, 0, 1),
        instance_ignore_index=-1,
        topk_insts=200,
        score_thr=0.0,
        npoint_thr=100,
        sp_score_thr=0.55,
        nms=True,
    ):
        super().__init__()

        # ignore info
        self.semantic_num_classes = semantic_num_classes
        self.semantic_ignore_index = semantic_ignore_index
        self.segment_ignore_index = segment_ignore_index
        self.instance_ignore_index = instance_ignore_index

        # backbone and pooling
        self.backbone = build_model(backbone)
        # decoder
        self.decoder = build_model(decoder)

        self.criterion = build_model(criterion)

        self.topk_insts = topk_insts
        self.score_thr = score_thr
        self.npoint_thr = npoint_thr
        self.sp_score_thr = sp_score_thr
        self.nms = nms

        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def forward(self, input_dict):
        pt_offset = input_dict["origin_offset"].int()
        vx_offset = input_dict["offset"].int()

        feats = self.backbone(input_dict)
        input_dict["feats"] = feats

        assert "inverse" in input_dict.keys()
        inv = split_offset(input_dict["inverse"], pt_offset)
        sp = split_offset(input_dict["superpoint"], pt_offset)
        sp = [torch.unique(_sp, return_inverse=True)[1] for _sp in sp]
        vx_feat = split_offset(feats, vx_offset)

        sp_feat = [
            scatter_mean(_feat[_inv], _sp, dim=0)
            for _feat, _inv, _sp in zip(vx_feat, inv, sp)
        ]

        input_dict["sp"] = sp
        input_dict["sp_feat"] = sp_feat
        input_dict["vx_feat"] = vx_feat
        input_dict["vx_coord"] = split_offset(input_dict["coord"], vx_offset)
        input_dict["inv"] = inv

        out = self.decoder(input_dict)

        # prepare target
        if ("origin_segment" and "origin_instance") in input_dict.keys():
            target = self.prepare_target(input_dict)
            return_dict = self.criterion(out, target)
        else:
            return_dict = {}

        return_dict["seg_logits"] = feats

        if not self.training:
            return_dict = self.prediction(out, return_dict, sp)

        return return_dict

    def prepare_target(self, input_dict):
        pt_offset = input_dict["origin_offset"].int()

        pt_ins = split_offset(input_dict["origin_instance"], pt_offset)
        pt_sem = split_offset(input_dict["origin_segment"], pt_offset)
        sp = input_dict["sp"]

        target = dict()
        target["inst_gt"] = []
        target["vx_gt"] = dict()

        vx_sem = process_label(input_dict["segment"].clone(), self.segment_ignore_index)
        vx_sem[vx_sem == self.semantic_ignore_index] = self.semantic_num_classes
        vx_coords = input_dict["coord"]
        vx_ins_cent = input_dict["instance_centroid"]

        target["vx_gt"]["labels"] = vx_sem
        target["vx_gt"]["coords"] = vx_coords
        bias_gt = vx_ins_cent - vx_coords
        target["vx_gt"]["bias_gt"] = bias_gt
        mask = (input_dict["instance"] != self.instance_ignore_index).float()
        target["vx_gt"]["ins_mask"] = mask

        for p_ins, p_cls, p_sp in zip(pt_ins, pt_sem, sp):
            p_ins = process_instance(
                p_ins.clone(), p_cls.clone(), self.segment_ignore_index
            )
            p_cls = process_label(p_cls.clone(), self.segment_ignore_index)

            # create gt instance markup
            p_ins_mask = p_ins.clone()

            if torch.sum(p_ins_mask == self.instance_ignore_index) != 0:
                p_ins_mask[p_ins_mask == self.instance_ignore_index] = (
                    torch.max(p_ins_mask) + 1
                )
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)[:, :-1]
            else:
                p_ins_mask = torch.nn.functional.one_hot(p_ins_mask)

            if p_ins_mask.shape[1] != 0:
                p_ins_mask = p_ins_mask.T
                sp_ins_mask = scatter_mean(p_ins_mask.float(), p_sp, dim=-1)
                sp_ins_mask = sp_ins_mask > 0.5
            else:
                sp_ins_mask = p_ins_mask.new_zeros(
                    (0, p_sp.max() + 1), dtype=torch.bool
                )

            insts = p_ins.unique()[1:]
            gt_labels = insts.new_zeros(len(insts))

            for inst in insts:
                index = p_cls[p_ins == inst][0]
                gt_labels[inst] = index
            target["inst_gt"].append(dict(labels=gt_labels, masks=sp_ins_mask))
        return target

    def prediction(self, out, return_dict, sp=None):
        scores, masks, classes = self.predict_by_feat(out, sp)
        masks = masks.cpu().detach().numpy()
        classes = classes.cpu().detach().numpy()

        sort_scores = scores.sort(descending=True)
        sort_scores_index = sort_scores.indices.cpu().detach().numpy()
        sort_scores_values = sort_scores.values.cpu().detach().numpy()
        sort_classes = classes[sort_scores_index]
        sorted_masks = masks[sort_scores_index]

        return_dict["pred_scores"] = sort_scores_values
        return_dict["pred_masks"] = sorted_masks
        return_dict["pred_classes"] = sort_classes

        return return_dict

    def predict_by_feat(self, out, sp=None):
        cls_preds = out["labels"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]

        if out["scores"] is not None:
            scores *= out["scores"][0]
        labels = (
            torch.arange(self.semantic_num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(len(cls_preds), 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(self.topk_insts, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.semantic_num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / (
            (mask_pred > 0).sum(1) + 1e-6
        )
        scores = scores * mask_scores

        if self.nms:
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel="linear"
            )

        if sp is not None:
            mask_pred_sigmoid = mask_pred_sigmoid[:, sp[0]]
        mask_pred = mask_pred_sigmoid > self.sp_score_thr

        # score_thr
        score_mask = scores > self.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return scores, mask_pred, labels
