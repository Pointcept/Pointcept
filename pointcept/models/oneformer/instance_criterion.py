"""
# This file includes code adapted from OneFormer:
# https://github.com/SHI-Labs/OneFormer
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from ..builder import MODELS, build_model
from .structures import InstanceData_


def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).

    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    pos_loss = torch.einsum("nc,mc->nm", pos, targets)
    neg_loss = torch.einsum("nc,mc->nm", neg, (1 - targets))
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).

    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).

    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).

    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()


@MODELS.register_module("OneFormer-InstanceCriterion")
class InstanceCriterion:
    """Instance criterion.

    Args:
        matcher (Callable): Class for matching queries with gt.
        loss_weight (List[float]): 4 weights for query classification,
            mask bce, mask dice, and score losses.
        non_object_weight (float): no_object weight for query classification.
        num_classes (int): number of classes.
        fix_dice_loss_weight (bool): Whether to fix dice loss for
            batch_size != 4.
        iter_matcher (bool): Whether to use separate matcher for
            each decoder layer.
        fix_mean_loss (bool): Whether to use .mean() instead of .sum()
            for mask losses.

    """

    def __init__(
        self,
        matcher,
        loss_weight,
        non_object_weight,
        num_classes,
        fix_dice_loss_weight,
        iter_matcher,
        fix_mean_loss=False,
    ):
        self.matcher = build_model(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss.

        Args:
            aux_outputs (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `gt_spmasks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).

        Returns:
            Tensor: loss value.
        """
        cls_preds = aux_outputs["cls_preds"]
        pred_scores = aux_outputs["scores"]
        pred_masks = aux_outputs["masks"]

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(scores=cls_preds[i], masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d, masks=insts[i].gt_spmasks
                )
                if insts[i].get("query_masks") is not None:
                    gt_instances.query_masks = insts[i].query_masks

                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )
            cls_target[idx_q.long()] = inst.labels_3d[idx_gt.long()]
            cls_losses.append(
                F.cross_entropy(
                    cls_pred,
                    cls_target,
                    cls_pred.new_tensor(self.class_weight),
                    ignore_index=-1,
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices
        ):
            if len(inst) == 0:
                continue
            idx_q = idx_q.long()
            idx_gt = idx_gt.long()
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

        return loss

    # todo: refactor pred to InstanceData_
    def __call__(self, pred, insts):
        """Loss main function.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks.
            insts (List):
                Ground truth of len batch_size, each InstanceData_ with
                    `gt_spmasks` of shape (n_gts_i, n_points_i)
                    `labels_3d` of shape (n_gts_i,)
                    `query_masks` of shape (n_gts_i, n_queries_i).

        Returns:
            Dict: with instance loss value.
        """
        cls_preds = pred["cls_preds"]
        pred_scores = pred["scores"]
        pred_masks = pred["masks"]

        # match
        indices = []
        for i in range(len(insts)):
            pred_instances = InstanceData_(scores=cls_preds[i], masks=pred_masks[i])
            gt_instances = InstanceData_(
                labels=insts[i].labels_3d, masks=insts[i].gt_spmasks
            )
            if insts[i].get("query_masks") is not None:
                gt_instances.query_masks = insts[i].query_masks

            indices.append(self.matcher(pred_instances, gt_instances))

        # class loss
        cls_losses = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )

            cls_target[idx_q.long()] = inst.labels_3d[idx_gt.long()]

            cls_losses.append(
                F.cross_entropy(
                    cls_pred,
                    cls_target,
                    cls_pred.new_tensor(self.class_weight),
                    ignore_index=-1,
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(
            pred_masks, pred_scores, insts, indices
        ):
            if len(inst) == 0:
                continue
            idx_q = idx_q.long()
            idx_gt = idx_gt.long()
            pred_mask = mask[idx_q]
            tgt_mask = inst.gt_spmasks[idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        # todo: actually .mean() should be better
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = 0

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4

            if self.fix_mean_loss:
                mask_bce_loss = mask_bce_loss * len(pred_masks) / len(mask_bce_losses)
                mask_dice_loss = (
                    mask_dice_loss * len(pred_masks) / len(mask_dice_losses)
                )
        else:
            mask_bce_loss = 0
            mask_dice_loss = 0

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

        if "aux_outputs" in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        return {"inst_loss": loss}


@MODELS.register_module("OneFormer-QueryClassificationCost")
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight


@MODELS.register_module("OneFormer-MaskBCECost")
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@MODELS.register_module("OneFormer-MaskDiceCost")
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).

        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@MODELS.register_module("OneFormer-SparseMatcher")
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(build_model(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))

        cost_values = []
        for cost in self.costs:
            c = cost(pred_instances, gt_instances)
            print(c.shape, "COST SHAPOE")
            cost_values.append(c)
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        import pdb

        pdb.set_trace()
        cost_value = torch.where(gt_instances.query_masks.T, cost_value, self.inf)

        cost_value = torch.where(
            torch.isnan(cost_value) | torch.isinf(cost_value), cost_value, self.inf
        )

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True, largest=False
        ).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]


LARGE_CONSTANT = 1e6


@MODELS.register_module("OneFormer-HungarianMatcher")
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """

    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            self.costs.append(build_model(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return torch.empty((0,)), torch.empty((0,))

        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        C = torch.stack(cost_values).sum(dim=0)
        if torch.isnan(C).any() or torch.isinf(C).any():
            # print(
            #     "Warning: Detected NaN or Inf in cost_value. Replacing with large values.")

            C = torch.where(torch.isnan(C) | torch.isinf(C), LARGE_CONSTANT, C)
        query_ids, object_ids = linear_sum_assignment(C.cpu())
        return torch.Tensor(query_ids).long(), torch.Tensor(object_ids).long()
