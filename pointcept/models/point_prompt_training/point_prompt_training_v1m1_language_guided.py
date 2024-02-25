"""
Point Prompt Training

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


@MODELS.register_module("PPT-v1m1")
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
        backbone_mode=False,
    ):
        super().__init__()
        assert len(conditions) == len(valid_index)
        assert backbone.type in ["SpUNet-v1m3", "PT-v2m3", "PT-v3m1"]
        self.backbone = MODELS.build(backbone)
        self.criteria = build_criteria(criteria)
        self.conditions = conditions
        self.valid_index = valid_index
        self.embedding_table = nn.Embedding(len(conditions), context_channels)
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
        feat = self.proj_head(feat)
        feat = feat / feat.norm(dim=-1, keepdim=True)
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
