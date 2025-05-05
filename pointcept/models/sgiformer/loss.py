import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.amp import autocast
from pointcept.utils.registry import Registry

MATCHER = Registry("MATCHER")
COST = Registry("COST")

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


class SGIFormerLoss(nn.Module):
    def __init__(
        self,
        matcher,
        loss_weight,
        non_object_weight,
        num_classes,
        fix_dice_loss_weight,
        iter_matcher,
        fix_mean_loss=False,
        semantic_ignore_index=-1,
        loss_cls_type="ce_loss",
    ):
        super().__init__()
        self.matcher = MATCHER.build(matcher)
        class_weight = [1] * num_classes + [non_object_weight]
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.loss_cls_type = loss_cls_type
        self.semantic_ignore_index = semantic_ignore_index

    def get_loss(
        self,
        pred_inst_info,
        gt_inst_info,
        matched_idx_list=None,
    ):
        if self.iter_matcher:
            matched_idx_list = []
        pred_cls_list = pred_inst_info["cls_list"]
        pred_mask_list = pred_inst_info["mask_list"]
        pred_score_list = pred_inst_info["score_list"]
        pred_inst_list = []
        for i in range(len(pred_cls_list)):
            pred_inst_list.append(dict(cls=pred_cls_list[i], mask=pred_mask_list[i]))

        device = pred_mask_list[0].device

        cls_loss_list = []
        score_loss_list = []
        mask_bce_loss_list = []
        mask_dice_loss_list = []

        for i in range(len(gt_inst_info)):
            if len(gt_inst_info[i]["cls"]) == 0:
                cls_loss_list.append(
                    torch.tensor(0.0, requires_grad=True, device=device)
                )
                score_loss_list.append(
                    torch.tensor(0.0, requires_grad=True, device=device)
                )
                mask_bce_loss_list.append(
                    torch.tensor(0.0, requires_grad=True, device=device)
                )
                mask_dice_loss_list.append(
                    torch.tensor(0.0, requires_grad=True, device=device)
                )
                continue
            if self.iter_matcher:
                pred_idx, gt_idx = self.matcher(pred_inst_list[i], gt_inst_info[i])
                matched_idx_list.append((pred_idx, gt_idx))
            else:
                pred_idx, gt_idx = matched_idx_list[i]

            pred_cls = pred_cls_list[i]
            pred_mask = pred_mask_list[i]
            pred_score = pred_score_list[i] if pred_score_list is not None else None
            gt_inst = gt_inst_info[i]

            # categorical loss
            num_classes = pred_cls.shape[1] - 1
            # ignored index become the last class to predict
            gt_cls = pred_cls.new_full((len(pred_cls),), num_classes, dtype=torch.long)
            gt_cls[pred_idx] = gt_inst["cls"][gt_idx]
            assert self.loss_cls_type == "ce_loss"
            cls_loss_list.append(
                F.cross_entropy(
                    pred_cls, gt_cls, pred_cls.new_tensor(self.class_weight)
                )
            )

            # 3 other losses
            if gt_inst["mask"].shape[0] == 0:
                continue

            pred_mask = pred_mask[pred_idx]
            gt_mask = gt_inst["mask"][gt_idx]
            mask_bce_loss_list.append(
                F.binary_cross_entropy_with_logits(pred_mask, gt_mask.float())
            )
            mask_dice_loss_list.append(dice_loss(pred_mask, gt_mask.float()))

            # check if skip objectness loss
            if pred_score is None:
                continue

            with torch.no_grad():
                gt_score = get_iou(pred_mask, gt_mask).unsqueeze(1)

            filter_id, _ = torch.where(gt_score > 0.5)
            if filter_id.numel():
                gt_score = gt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_loss_list.append(F.mse_loss(pred_score, gt_score))

        # process loss lists
        cls_loss = torch.mean(torch.stack(cls_loss_list))

        if len(score_loss_list):
            score_loss = torch.stack(score_loss_list).sum() / len(pred_mask_list)
        else:
            score_loss = torch.tensor(0.0, requires_grad=True, device=device)

        if len(mask_bce_loss_list):
            mask_bce_loss = torch.stack(mask_bce_loss_list).sum() / len(pred_mask_list)
            mask_dice_loss = torch.stack(mask_dice_loss_list).sum()

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_mask_list) * 4

            if self.fix_mean_loss:
                mask_bce_loss = (
                    mask_bce_loss * len(pred_mask_list) / len(mask_bce_loss_list)
                )
                mask_dice_loss = (
                    mask_dice_loss * len(pred_mask_list) / len(mask_dice_loss_list)
                )
        else:
            mask_bce_loss = torch.tensor(0.0, requires_grad=True, device=device)
            mask_dice_loss = torch.tensor(0.0, requires_grad=True, device=device)

        loss = (
            self.loss_weight[0] * cls_loss
            + self.loss_weight[1] * mask_bce_loss
            + self.loss_weight[2] * mask_dice_loss
            + self.loss_weight[3] * score_loss
        )
        loss_dict = dict()
        loss_dict["loss_cls"] = cls_loss
        loss_dict["loss_mask"] = mask_bce_loss
        loss_dict["loss_dice"] = mask_dice_loss
        loss_dict["loss_score"] = score_loss
        return loss, loss_dict, matched_idx_list

    @staticmethod
    def loss_bias(pred_bias, gt_bias, gt_mask):
        bias_dist = torch.sum(torch.abs(pred_bias - gt_bias), dim=-1)
        bias_l1_loss = torch.sum(bias_dist * gt_mask) / (torch.sum(gt_mask) + 1e-8)

        loss_bias = bias_l1_loss
        return loss_bias

    def forward(self, pred, target):
        """Loss main function."""
        gt_inst_info = target["inst_info"]
        gt_point_info = target["point_info"]

        pred_inst_info = dict(
            cls_list=pred["cls_list"],
            mask_list=pred["mask_list"],
            score_list=pred["score_list"],
        )
        loss, loss_dict, matched_idx_list = self.get_loss(pred_inst_info, gt_inst_info)
        if "aux_pred_list" in pred:
            if self.iter_matcher:
                matched_idx_list = None
            for pred_inst_info_ in pred["aux_pred_list"]:
                aux_loss, _, _ = self.get_loss(
                    pred_inst_info_, gt_inst_info, matched_idx_list
                )
                loss = loss + aux_loss

        if "seg_logits" in pred:
            if pred["seg_logits"] is not None:
                if gt_point_info["segment"].max() >= 0:
                    loss_seg = F.cross_entropy(
                        pred["seg_logits"],
                        gt_point_info["segment"],
                        pred["seg_logits"].new_tensor(self.class_weight),
                        ignore_index=self.semantic_ignore_index,
                    )
                else:
                    loss_seg = torch.tensor(0.0, requires_grad=True, device=loss.device)

                loss = loss + self.loss_weight[4] * loss_seg
            else:
                loss_seg = torch.tensor(0.0, requires_grad=True, device=loss.device)
            loss_dict["loss_seg"] = loss_seg
        if "bias" in pred:
            if pred["bias"] is not None:
                loss_bias = self.loss_bias(
                    pred["bias"],
                    gt_point_info["bias"],
                    gt_point_info["mask"],
                )
                loss += self.loss_weight[5] * loss_bias
            else:
                loss_bias = torch.tensor(0.0, device=loss.device)
            loss_dict["loss_bias"] = loss_bias
        loss_dict["loss"] = loss
        return loss_dict


@COST.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_inst, gt_inst, **kwargs):
        """Compute match cost."""
        score = pred_inst["cls"].softmax(-1)
        cost = -score[:, gt_inst["cls"]]
        return cost * self.weight


@COST.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for mask.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_inst, gt_inst, **kwargs):
        """Compute match cost."""
        with autocast("cuda", enabled=False):
            cost = batch_sigmoid_bce_loss(
                pred_inst["mask"].float(), gt_inst["mask"].float()
            )
        return cost * self.weight


@COST.register_module()
class MaskDiceCost:
    """Dice cost for mask.

    Args:
        weigth (float): Weight of the cost.
    """

    def __init__(self, weight):
        self.weight = weight

    def __call__(self, pred_inst, gt_inst, **kwargs):
        """Compute match cost."""
        with autocast("cuda", enabled=False):
            cost = batch_dice_loss(pred_inst["mask"].float(), gt_inst["mask"].float())
        return cost * self.weight


@MATCHER.register_module()
class HungarianMatcher:
    """Hungarian matcher.

    Args:
        costs (List[ConfigDict]): Cost functions.
    """

    def __init__(self, costs):
        self.costs = []
        for cost in costs:
            self.costs.append(COST.build(cost))

    @torch.no_grad()
    def __call__(self, pred_inst, gt_inst, **kwargs):
        """Compute match cost.

        Returns:
            Tuple:
                - Tensor: Query ids of shape (n_matched,),
                - Tensor: Object ids of shape (n_matched,).
        """
        inst_cls = gt_inst["cls"]
        inst_num = len(inst_cls)
        if inst_num == 0:
            return inst_cls.new_empty((0,), dtype=torch.int64), inst_cls.new_empty(
                (0,), dtype=torch.int64
            )

        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_inst, gt_inst))
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

        return inst_cls.new_tensor(query_ids, dtype=torch.int64), inst_cls.new_tensor(
            object_ids, dtype=torch.int64
        )
