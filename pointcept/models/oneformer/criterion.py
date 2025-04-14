"""
# This file includes code adapted from OneFormer:
# https://github.com/SHI-Labs/OneFormer
"""

from ..builder import MODELS
from .structures import InstanceData_


@MODELS.register_module("OneFormer-ScanNetUnifiedCriterion")
class ScanNetUnifiedCriterion:
    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion):
        self.num_semantic_classes = num_semantic_classes
        self.inst_criterion = MODELS.build(inst_criterion)

    def __call__(self, pred, insts):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `gt_spmasks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        inst_gts = []
        for i in range(len(pred["masks"])):

            inst_gt = InstanceData_()

            inst_gt.gt_spmasks = insts[i].gt_spmasks.to(insts[i].query_masks.device)
            inst_gt.labels_3d = insts[i].gt_labels.to(insts[i].query_masks.device)

            if insts[i].get("query_masks") is not None:
                inst_gt.query_masks = insts[i].query_masks

            inst_gts.append(inst_gt)

        loss_inst = self.inst_criterion(pred, inst_gts)

        loss = loss_inst["inst_loss"]
        loss_dict = {
            "loss": loss,
        }
        return loss_dict
