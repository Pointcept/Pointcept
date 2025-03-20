"""
Point Prompt Training specific for Sonata

In Sonata, as we identify the criminal that restricts domain adaptive capacity is BN
and successfully replaces all BN in PTv3 to LN. I think we don't need PDNorm
anymore. Hence, I remove the domain prompting and turns it into a pure multi-dataset
joint training framework.

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


@MODELS.register_module("PPT-v1m3")
class PointPromptTraining(nn.Module):
    """
    PointPromptTraining provides Data-driven Context and enables multi-dataset training with
    Language-driven Categorical Alignment. PDNorm is supported by SpUNet-v1m3 to adapt the
    backbone to a specific dataset with a given dataset condition and context.
    """

    def __init__(
        self,
        backbone=None,
        criteria=None,
        backbone_out_channels=96,
        context_channels=256,
        conditions=("Structured3D", "ScanNet", "S3DIS"),
        template="[x]",
        clip_model="ViT-B/16",
        # fmt: off
        class_name=(
            "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
            "window", "bookshelf", "bookcase", "picture", "counter", "desk", "shelves", "curtain",
            "dresser", "pillow", "mirror", "ceiling", "refrigerator", "television", "shower curtain", "nightstand",
            "toilet", "sink", "lamp", "bathtub", "garbagebin", "board", "beam", "column",
            "clutter", "otherstructure", "otherfurniture", "otherprop",
        ),
        valid_index=(
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 25, 26, 33, 34, 35),
            (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 20, 22, 24, 25, 27, 34),
            (0, 1, 4, 5, 6, 7, 8, 10, 19, 29, 30, 31, 32),
        ),
        # fmt: on
        freeze_backbone=False,
        backbone_mode=False,
    ):
        super().__init__()
        assert len(conditions) == len(valid_index)
        # assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.valid_index = valid_index
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.backbone_mode = backbone_mode
        if not self.backbone_mode:
            import clip

            clip_model, _ = clip.load(
                clip_model, device="cpu", download_root="./.cache/clip"
            )
            clip_model.requires_grad_(False)
            class_prompt = [template.replace("[x]", name) for name in class_name]
            class_token = clip.tokenize(class_prompt)
            class_embedding = clip_model.encode_text(class_token)
            class_embedding = class_embedding / class_embedding.norm(
                dim=-1, keepdim=True
            )
            self.register_buffer("class_embedding", class_embedding)
            self.proj_head = nn.Linear(
                backbone_out_channels, clip_model.text_projection.shape[1]
            )
            self.logit_scale = clip_model.logit_scale

    def forward(self, data_dict):
        condition = data_dict["condition"][0]
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(data_dict)
        else:
            point = self.backbone(data_dict)
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        if self.backbone_mode:
            # PPT serve as a multi-dataset backbone when enable backbone mode
            return point.feat

        feat = self.proj_head(point.feat)

        eps = 1e-6 if feat.dtype == torch.float16 else 1e-12
        feat = nn.functional.normalize(feat, dim=-1, p=2, eps=eps)

        sim = (
            feat
            @ self.class_embedding[
                self.valid_index[self.conditions.index(condition)], :
            ].t()
        )
        logit_scale = self.logit_scale.exp()
        seg_logits = logit_scale * sim
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
