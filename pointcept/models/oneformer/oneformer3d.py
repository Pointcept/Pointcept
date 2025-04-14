"""
# This file includes code adapted from OneFormer:
# https://github.com/SHI-Labs/OneFormer
"""

import numpy as np
import torch
import torch.nn.functional as F
import spconv.pytorch as spconv
from torch_scatter import scatter_mean
from .mask_matrix_nms import mask_matrix_nms
from ..builder import MODELS, build_model
import torch.nn as nn

try:
    import pointgroup_ops_sp.sp_ops as pointops
except ImportError:
    pointops = None


@MODELS.register_module("OneFormer3D")
class ScanNetOneFormer3D(nn.Module):

    def __init__(
        self,
        in_channels=6,
        num_channels=32,
        voxel_size=0.02,
        num_classes=84,
        min_spatial_shape=128,
        query_thr=0.5,
        backbone=None,
        decoder=None,
        criterion=None,
        train_cfg=None,
        test_cfg=None,
        data_preprocessor=None,
        init_cfg=None,
    ):
        super().__init__()
        assert (
            pointops is not None
        ), "Please install the pointgoup_ops_sp from the lib folder"
        self.unet = build_model(backbone)
        self.decoder = MODELS.build(decoder)
        self.criterion = MODELS.build(criterion)
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.min_spatial_shape = min_spatial_shape
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, num_channels)

    def _init_layers(self, in_channels, num_channels):
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )

    def extract_feat(self, x, superpoints, inverse_mapping, batch_offsets):
        """Extract features from sparse tensor.

        Args:
            x (SparseTensor): Input sparse tensor of shape
                (n_points, in_channels).
            superpoints (Tensor): of shape (n_points,).
            inverse_mapping (Tesnor): of shape (n_points,).
            batch_offsets (List[int]): of len batch_size + 1.

        Returns:
            List[Tensor]: of len batch_size,
                each of shape (n_points_i, n_channels).
        """
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = scatter_mean(x.features[inverse_mapping], superpoints, dim=0)
        out = []
        for i in range(len(batch_offsets) - 1):
            out.append(x[batch_offsets[i] : batch_offsets[i + 1]])
        return out

    def forward(self, batch):
        torch.cuda.empty_cache()
        batch_copy = batch.copy()  # To avoid modifying the original dictionary
        batch_copy.pop("name", None)
        batch_copy.pop("origin_segment", None)
        batch_copy.pop("origin_instance", None)

        if not self.training:
            loss_dict, x = self.loss(**batch_copy)
            inst = self.predict(**batch_copy)
            return_dict = {
                "pred_scores": inst["conf"].cpu().numpy(),
                "pred_classes": inst["label_id"].cpu().numpy(),
                "pred_masks": inst["pred_mask"].cpu().numpy(),
                "loss": loss_dict["loss"],
            }
            # import pdb
            # pdb.set_trace()
            return return_dict

        loss_dict, _ = self.loss(**batch_copy)

        return loss_dict

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
        x = spconv.SparseConvTensor(
            voxel_feats, voxel_coords.int(), spatial_shape, batch_size
        )

        x = self.extract_feat(x, superpoints, p2v_map, batch_offsets)

        queries, insts = self._select_queries(x, insts)
        # import pdb
        # pdb.set_trace()
        x = self.decoder(x, queries)
        # import pdb
        # pdb.set_trace()
        loss = self.criterion(x, insts)
        # loss["seg_logits"] = x["scores"]

        return loss, x

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

        sp_feats = self.extract_feat(input, superpoints, p2v_map, batch_offsets)

        out = self.decoder(sp_feats, sp_feats)

        pred = self.predict_by_feat(out, superpoints, scan_ids)
        return pred

    def _select_queries(self, x, gt_instances):
        """Select queries for train pass.

        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, n_channels).
            gt_instances (List[InstanceData_]): of len batch_size.
                Ground truth which can contain `labels` of shape (n_gts_i,),
                `sp_masks` of shape (n_gts_i, n_points_i).

        Returns:
            Tuple:
                List[Tensor]: Queries of len batch_size, each queries of shape
                    (n_queries_i, n_channels).
                List[InstanceData_]: of len batch_size, each updated
                    with `query_masks` of shape (n_gts_i, n_queries_i).
        """
        queries = []
        # import pdb
        # pdb.set_trace()
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                # ids = torch.randperm(len(x[i]))[:n]
                # gt_instances[i].gt_spmasks = gt_instances[i].gt_spmasks.to(
                #     x[i].device)
                queries.append(x[i][ids])
                if gt_instances[i].gt_spmasks.numel() > 0:
                    gt_instances[i].query_masks = (
                        gt_instances[i].gt_spmasks.to(x[i].device)[:, ids].bool()
                    )
                else:
                    gt_instances[i].query_masks = torch.empty(
                        (0, len(ids)), device=x[i].device, dtype=torch.bool
                    )
            else:
                queries.append(x[i])
                # gt_instances[i].gt_spmasks = gt_instances[i].gt_spmasks.to(
                #     x[i].device)
                if gt_instances[i].gt_spmasks.numel() > 0:
                    gt_instances[i].query_masks = (
                        gt_instances[i].gt_spmasks.to(x[i].device).bool()
                    )
                else:
                    gt_instances[i].query_masks = torch.empty(
                        (
                            0,
                            (
                                gt_instances[i].gt_spmasks.shape[1]
                                if gt_instances[i].gt_spmasks.dim() > 1
                                else 0
                            ),
                        ),
                        device=x[i].device,
                        dtype=torch.bool,
                    )
        # import pdb
        # pdb.set_trace()
        return queries, gt_instances

    def predict_by_feat(self, out, superpoints, scene_ids):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.

        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out["cls_preds"][0]
        pred_masks = out["masks"][0]

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out["scores"][0] is not None:
            scores *= out["scores"][0]
        labels = (
            torch.arange(self.num_classes, device=scores.device)
            .unsqueeze(0)
            .repeat(len(cls_preds), 1)
            .flatten(0, 1)
        )
        scores, topk_idx = scores.flatten(0, 1).topk(
            self.test_cfg.topk_insts, sorted=False
        )
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode="floor")
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        if self.test_cfg.get("obj_normalization", None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / (
                (mask_pred > 0).sum(1) + 1e-6
            )
            scores = scores * mask_scores

        if self.test_cfg.get("nms", None):
            kernel = self.test_cfg.matrix_nms_kernel
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(
                mask_pred_sigmoid, labels, scores, kernel=kernel
            )

        mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > 0.0
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        scores = scores[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        # return mask_pred, labels, scores
        return {"conf": scores, "label_id": labels, "pred_mask": mask_pred}
