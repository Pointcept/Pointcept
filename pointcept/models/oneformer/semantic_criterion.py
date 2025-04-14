"""
# This file includes code adapted from OneFormer:
# https://github.com/SHI-Labs/OneFormer
"""

import torch
import torch.nn.functional as F
from ..builder import MODELS, build_model


@MODELS.register_module("OneFormer-ScanNetSemanticCriterion")
class ScanNetSemanticCriterion:
    """Semantic criterion for ScanNet.

    Args:
        ignore_index (int): Ignore index.
        loss_weight (float): Loss weight.
    """

    def __init__(self, ignore_index, loss_weight):
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (dict): Predictions with List `sem_preds`
                of len batch_size, each of shape
                (n_queries_i, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with `sp_masks` of shape
                (n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic loss value.
        """
        losses = []
        for pred_mask, gt_mask in zip(pred["sem_preds"], insts):
            if self.ignore_index >= 0:
                pred_mask = pred_mask[:, :-1]
            losses.append(
                F.cross_entropy(
                    pred_mask,
                    gt_mask.gt_spmasks.float().argmax(0),
                    ignore_index=self.ignore_index,
                )
            )
        loss = self.loss_weight * torch.mean(torch.stack(losses))
        return dict(seg_loss=loss)
