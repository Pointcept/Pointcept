"""
# This file includes code adapted from SPFormer:
# https://github.com/sunjiahao1999/SPFormer
# Original author: Sun Jiahao (@sunjiahao1999)
"""

import functools

try:
    import pointgroup_ops_sp.sp_ops as pointops
except ImportError:
    pointops = None
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from .utils import cuda_cast
from .backbone import ResidualBlock, UBlock
from .loss import Criterion
from .query_decoder import QueryDecoder
from ..builder import MODELS

import numpy as np


@MODELS.register_module("SPFormer")
class SPFormer(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool="mean",
        num_class=18,
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()
        assert (
            pointops is not None
        ), "Please install the pointgoup_ops_sp from the lib folder"
        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(
            norm_fn(media), nn.ReLU(inplace=True)
        )
        self.pool = pool
        self.num_class = num_class

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(SPFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm1d only
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode="loss"):
        batch_copy = batch.copy()  # To avoid modifying the original dictionary
        batch_copy.pop("name", None)
        batch_copy.pop("origin_segment", None)
        batch_copy.pop("origin_instance", None)
        if not self.training:
            loss, loss_dict, sp_feats = self.loss(**batch_copy)
            loss_dict["loss"] = loss
            pred = self.predict(**batch_copy)
            inst = pred["pred_instances"]
            y = inst
            result = {"conf": [], "label_id": [], "pred_mask": [], "scan_id": []}
            for item in y:
                for key, value in item.items():
                    result[key].append(value)
            inst = result
            return_dict = {
                "pred_scores": np.array(inst["conf"]),
                "pred_classes": np.array(inst["label_id"]),
                "pred_masks": np.array(inst["pred_mask"]),
                "loss": loss,
            }

            return return_dict
        loss, loss_dict, sp_feats = self.loss(**batch_copy)
        loss_dict["loss"] = loss
        loss_dict["seg_logits"] = sp_feats
        return loss_dict

    @cuda_cast
    def loss(
        self,
        scan_ids,
        voxel_coords,
        p2v_map,
        v2p_map,
        spatial_shape,
        feats,
        insts,
        superpoints,
        batch_offsets,
    ):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, batch_size
        )

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        out = self.decoder(sp_feats, batch_offsets)
        # import pdb
        # pdb.set_trace()
        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict, sp_feats.detach()

    @cuda_cast
    def predict(
        self,
        scan_ids,
        voxel_coords,
        p2v_map,
        v2p_map,
        spatial_shape,
        feats,
        insts,
        superpoints,
        batch_offsets,
    ):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, batch_size
        )

        sp_feats = self.extract_feat(input, superpoints, p2v_map)
        out = self.decoder(sp_feats, batch_offsets)

        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out["labels"]
        pred_masks = out["masks"]
        pred_scores = out["scores"]
        # import pdb
        # pdb.set_trace()
        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]
        labels = (
            torch.arange(self.num_class, device=scores.device)
            .unsqueeze(0)
            .repeat(self.decoder.num_query, 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False
        )
        labels = labels[topk_idx]
        # labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode="floor")
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred["scan_id"] = scan_ids[0]
            pred["label_id"] = cls_pred[i]
            pred["conf"] = score_pred[i]
            pred["pred_mask"] = mask_pred[i]
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(
            scan_id=scan_ids[0],
            pred_instances=pred_instances,
            gt_instances=gt_instances,
        )

    def extract_feat(self, x, superpoints, v2p_map):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == "mean":
            x = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == "max":
            x, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        return x
