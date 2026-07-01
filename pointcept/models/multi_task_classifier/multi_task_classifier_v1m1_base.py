import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import segment_csr
from ..builder import MODELS
from ..builder import build_model
from ..losses.misc import CrossEntropyLoss
from pointcept.models.utils.structure import Point

@MODELS.register_module()
class MultiTaskClassifier(nn.Module):
    """
    Backbone + *k* independent classifier heads.
    """

    def __init__(
        self,
        backbone,
        num_classes_list,
        backbone_embed_dim,
        loss_type="ce",
        class_weights=None,
        label_smoothing=0.0,
        stl_task=-1
    ):
        super().__init__()

        self.backbone = build_model(backbone)
        self.num_tasks = len(num_classes_list)
        self.num_classes_list = list(num_classes_list)
        self.stl_task=stl_task

        # one MLP head per task  (Linear-BN-ReLU-Drop-Linear)
        self.heads = nn.ModuleList()
        for n_cls in num_classes_list:
            self.heads.append(
                nn.Sequential(
                    nn.Linear(backbone_embed_dim, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(256, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.5),
                    nn.Linear(128, n_cls),
                )
            )
        # CrossEntropy
        self.criterions = nn.ModuleList()
        for i, n_cls in enumerate(num_classes_list):
            w = None
            if class_weights and class_weights[i] is not None:
                w = torch.tensor(class_weights[i], dtype=torch.float32)
            self.criterions.append(CrossEntropyLoss(weight=w, ignore_index=-1, label_smoothing=label_smoothing))

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)

        # PTv3 or SpuNet
        if isinstance(point, Point):
            point.feat = segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point

        # per-task logits
        logits = [head(feat) for head in self.heads]

        out = {"logits": logits}

        # compute losses
        if any(k.startswith("label_") for k in input_dict):
            losses = []
            for i, logit in enumerate(logits):
                target = input_dict[f"label_{i}"].view(-1)
                loss_val = self.criterions[i](logit, target)
                if loss_val.ndim > 0:
                    loss_val = loss_val.mean()

                # kill loss for STL experiment
                if self.stl_task != -1 and i != self.stl_task:
                    loss_val = loss_val * 0

                losses.append(loss_val)
                out[f"loss_{i}"] = loss_val
            out["loss"] = sum(losses)

        return out