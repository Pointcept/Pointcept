"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from pointcept.models.builder import MODELS, build_model
from torch.cuda.amp import autocast

LARGE_CONSTANT = 1e6


@torch.jit.script
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


@torch.jit.script
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


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (torch.Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (torch.Tensor): A float tensor with the same shape as inputs. Stores the binary
            classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
        alpha (float, optional): Weighting factor in range (0, 1) to balance
            positive vs negative examples. Default: 0.25.
        gamma (float): Exponent of the modulating factor (1 - p_t) to
            balance easy vs hard examples. Default: 2.

    Returns:
        torch.Tensor: The computed sigmoid focal loss.
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean()


@MODELS.register_module("InstanceCriterion")
class InstanceCriterion:
    def __init__(
        self,
        matcher,
        loss_weight,
        non_object_weight,
        num_classes,
        fix_dice_loss_weight,
        iter_matcher,
        fix_mean_loss=False,
        loss_cls_type="ce_loss",
    ):
        self.matcher = build_model(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.loss_cls_type = loss_cls_type

    def get_layer_loss(self, aux_outputs, insts, indices=None):
        """Per layer auxiliary loss."""
        cls_preds = aux_outputs["labels"]
        pred_masks = aux_outputs["masks"]

        pred_insts = []
        for i in range(len(cls_preds)):
            pred_insts.append(dict(scores=cls_preds[i], masks=pred_masks[i]))

        if indices is None:
            indices = []
            for i in range(len(insts)):
                indices.append(self.matcher(pred_insts[i], insts[i]))

        cls_losses = []
        for i, (cls_pred, inst, (idx_q, idx_gt)) in enumerate(
            zip(cls_preds, insts, indices)
        ):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )
            cls_target[idx_q] = inst["labels"][idx_gt]

            assert self.loss_cls_type == "ce_loss"
            cls_losses.append(
                F.cross_entropy(
                    cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for i, (mask, inst, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks, insts, indices)
        ):
            if inst["masks"].shape[0] == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst["masks"][idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if aux_outputs["scores"] is None:
                continue

            pred_score = aux_outputs["scores"][i][idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)

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
            mask_bce_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)
            mask_dice_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )

        return loss

    def loss_vx_bias(self, vx_bias, target):
        bias_gt = target["bias_gt"]
        mask = target["ins_mask"]
        bias_pred = vx_bias

        bias_dist = torch.sum(torch.abs(bias_pred - bias_gt), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * mask) / (torch.sum(mask) + 1e-8)

        loss_bias = bias_l1_loss
        return loss_bias

    def __call__(self, pred, target):
        """Loss main function."""
        insts = target["inst_gt"]
        vx_gt = target["vx_gt"]
        losses = {}
        cls_preds = pred["labels"]
        pred_masks = pred["masks"]
        pred_insts = []
        assert len(cls_preds) == len(pred_masks), "Number of batchs should be the same."
        for i in range(len(cls_preds)):
            pred_insts.append(dict(scores=cls_preds[i], masks=pred_masks[i]))

        # match
        indices = []
        for i in range(len(insts)):
            indices.append(self.matcher(pred_insts[i], insts[i]))

        # class loss
        cls_losses = []
        for i, (cls_pred, inst, (idx_q, idx_gt)) in enumerate(
            zip(cls_preds, insts, indices)
        ):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long
            )
            cls_target[idx_q] = inst["labels"][idx_gt]

            assert self.loss_cls_type == "ce_loss"
            cls_losses.append(
                F.cross_entropy(
                    cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)
                )
            )
        cls_loss = torch.mean(torch.stack(cls_losses))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for i, (mask, inst, (idx_q, idx_gt)) in enumerate(
            zip(pred_masks, insts, indices)
        ):
            if inst["masks"].shape[0] == 0:
                continue
            pred_mask = mask[idx_q]
            tgt_mask = inst["masks"][idx_gt]
            mask_bce_losses.append(
                F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float())
            )
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))

            # check if skip objectness loss
            if pred["scores"] is None:
                continue

            pred_score = pred["scores"][i][idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))
        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)

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
            mask_bce_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)
            mask_dice_loss = torch.tensor(0.0, requires_grad=True, device=mask.device)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )
        losses["loss_cls"] = cls_loss
        losses["loss_mask"] = mask_bce_loss
        losses["loss_dice"] = mask_dice_loss
        losses["loss_score"] = score_loss

        if "aux_outputs" in pred:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs in enumerate(pred["aux_outputs"]):
                loss += self.get_layer_loss(aux_outputs, insts, indices)

        if "vx_seg" in pred:
            if pred["vx_seg"] is not None:
                loss_vx_cls = F.cross_entropy(
                    pred["vx_seg"],
                    vx_gt["labels"],
                    pred["vx_seg"].new_tensor(self.class_weight),
                )
                loss += self.loss_weight[4] * loss_vx_cls
            else:
                loss_vx_cls = torch.tensor(0.0, device=mask.device)
            losses["loss_vx_cls"] = loss_vx_cls
        if "vx_bias" in pred:
            if pred["vx_bias"] is not None:
                loss_vx_bias = self.loss_vx_bias(pred["vx_bias"], vx_gt)
                loss += self.loss_weight[5] * loss_vx_bias
            else:
                loss_vx_bias = torch.tensor(0.0, device=mask.device)
            losses["loss_vx_bias"] = loss_vx_bias
        losses["loss"] = loss
        return losses


@MODELS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost."""
        scores = pred_instances["scores"].softmax(-1)
        cost = -scores[:, gt_instances["labels"]]
        return cost * self.weight


@MODELS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost."""
        with autocast(enabled=False):
            cost = batch_sigmoid_bce_loss(
                pred_instances["masks"].float(), gt_instances["masks"].float()
            )
        return cost * self.weight


@MODELS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost."""
        with autocast(enabled=False):
            cost = batch_dice_loss(
                pred_instances["masks"].float(), gt_instances["masks"].float()
            )
        return cost * self.weight


@MODELS.register_module()
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

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances["labels"]
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,), dtype=torch.int64), labels.new_empty(
                (0,), dtype=torch.int64
            )

        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        cost_value = torch.stack(cost_values).sum(dim=0)

        if torch.isnan(cost_value).any() or torch.isinf(cost_value).any():
            print(
                "Warning: Detected NaN or Inf in cost_value. Replacing with large values."
            )
            cost_value = torch.where(
                torch.isnan(cost_value) | torch.isinf(cost_value),
                torch.full_like(cost_value, LARGE_CONSTANT),
                cost_value,
            )
        query_ids, object_ids = linear_sum_assignment(cost_value.cpu().numpy())

        return labels.new_tensor(query_ids, dtype=torch.int64), labels.new_tensor(
            object_ids, dtype=torch.int64
        )
