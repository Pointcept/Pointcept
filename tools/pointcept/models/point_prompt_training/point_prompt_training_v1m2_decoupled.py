"""
Point Prompt Training with decoupled segmentation head

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
from pointcept.models.utils.structure import Point
from pointcept.models.builder import MODELS
from pointcept.models.losses import build_criteria


@MODELS.register_module("PPT-v1m2")
class PointPromptTraining(nn.Module):
    """
    PointPromptTraining v1m2 provides Data-driven Context and enables multi-dataset training with
    Decoupled Segmentation Head. PDNorm is supported by SpUNet-v1m3 to adapt the
    backbone to a specific dataset with a given dataset condition and context.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        backbone_out_channels=96,
        context_channels=256,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
        num_classes=(25, 20, 13),
        backbone_mode=False,
    ):
        super().__init__()
        assert len(conditions) == len(num_classes)
        assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.embedding_table = nn.Embedding(len(conditions), context_channels)
        self.backbone_mode = backbone_mode
        self.seg_heads = nn.ModuleList(
            [nn.Linear(backbone_out_channels, num_cls) for num_cls in num_classes]
        )

    def forward(self, data_dict):
        condition = data_dict["condition"][0]
        assert condition in self.conditions
        context = self.embedding_table(
            torch.tensor(
                [self.conditions.index(condition)], device=data_dict["coord"].device
            )
        )
        data_dict["context"] = context
        point = self.backbone(data_dict)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            feat = point.feat
        else:
            feat = point
        if self.backbone_mode:
            # PPT serve as a multi-dataset backbone when enable backbone mode
            return feat
        seg_head = self.seg_heads[self.conditions.index(condition)]
        seg_logits = seg_head(feat)
        # train
        if self.training:
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in data_dict.keys():
            loss = self.criteria(seg_logits, data_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)
