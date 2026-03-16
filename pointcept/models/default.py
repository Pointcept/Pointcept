import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module("BoundaryAwareSegmentor")
class BoundaryAwareSegmentor(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        boundary_k=16,
        boundary_loss_weight=1.0,
        ignore_index=-1,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.boundary_k = boundary_k
        self.boundary_loss_weight = boundary_loss_weight
        self.ignore_index = ignore_index

    def _get_boundary_mask(self, coord, segment, offset):
        if self.boundary_k <= 0:
            return torch.zeros_like(segment, dtype=torch.bool)
        batch = offset2batch(offset)
        edge_index = torch_cluster.knn_graph(
            coord, k=self.boundary_k, batch=batch, loop=False
        )
        src, dst = edge_index[0], edge_index[1]
        valid = (segment[src] != self.ignore_index) & (segment[dst] != self.ignore_index)
        diff = valid & (segment[src] != segment[dst])
        boundary = torch_scatter.scatter(
            diff.to(torch.int8),
            dst,
            dim=0,
            dim_size=segment.shape[0],
            reduce="max",
        ).bool()
        return boundary

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]

        seg_logits = self.backbone(input_dict)

        if "segment" in input_dict.keys():
            target = input_dict["segment"]
            main_loss = self.criteria(seg_logits, target)

            boundary_loss = seg_logits.sum() * 0
            if (
                self.training
                and self.boundary_loss_weight > 0
                and "coord" in input_dict.keys()
                and "offset" in input_dict.keys()
            ):
                boundary_mask = self._get_boundary_mask(
                    input_dict["coord"], target, input_dict["offset"]
                )
                if torch.any(boundary_mask):
                    boundary_loss = F.cross_entropy(
                        seg_logits[boundary_mask],
                        target[boundary_mask],
                        ignore_index=self.ignore_index,
                    )

            loss = main_loss + self.boundary_loss_weight * boundary_loss

            if self.training:
                return dict(loss=loss, main_loss=main_loss, boundary_loss=boundary_loss)
            return dict(loss=loss, seg_logits=seg_logits)

        return dict(seg_logits=seg_logits)


@MODELS.register_module("OrganAwareResidualSegmentor")
class OrganAwareResidualSegmentor(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        organ_class_ids,
        backbone=None,
        criteria=None,
        expert_criteria=None,
        gate_criteria=None,
        ignore_index=-1,
        gate_ignore_class_ids=(),
        main_loss_weight=1.0,
        fused_loss_weight=1.0,
        expert_loss_weight=1.0,
        gate_loss_weight=1.0,
        residual_scale=1.0,
        residual_mode="interpolate",
        center_expert_residual=True,
    ):
        super().__init__()
        if criteria is None:
            raise ValueError("OrganAwareResidualSegmentor requires `criteria`.")
        if len(organ_class_ids) == 0:
            raise ValueError("`organ_class_ids` must contain at least one class id.")
        self.backbone = build_model(backbone)
        self.main_criteria = build_criteria(criteria)
        self.expert_criteria = build_criteria(
            expert_criteria if expert_criteria is not None else criteria
        )
        self.gate_criteria = build_criteria(
            gate_criteria
            if gate_criteria is not None
            else [dict(type="CrossEntropyLoss", ignore_index=ignore_index)]
        )
        self.organ_class_ids = list(organ_class_ids)
        self.gate_ignore_class_ids = set(gate_ignore_class_ids)
        self.ignore_index = ignore_index
        self.main_loss_weight = main_loss_weight
        self.fused_loss_weight = fused_loss_weight
        self.expert_loss_weight = expert_loss_weight
        self.gate_loss_weight = gate_loss_weight
        self.residual_scale = residual_scale
        self.residual_mode = residual_mode
        self.center_expert_residual = center_expert_residual

        self.expert_head = nn.Linear(backbone_out_channels, len(self.organ_class_ids))
        self.gate_head = nn.Linear(backbone_out_channels, 2)

    def _build_expert_target(self, segment):
        target = torch.full_like(segment, self.ignore_index)
        for local_id, class_id in enumerate(self.organ_class_ids):
            target[segment == class_id] = local_id
        return target

    def _build_gate_target(self, segment):
        target = torch.full_like(segment, self.ignore_index)
        valid_mask = segment != self.ignore_index
        target[valid_mask] = 0
        for class_id in self.organ_class_ids:
            target[segment == class_id] = 1
        for class_id in self.gate_ignore_class_ids:
            target[segment == class_id] = self.ignore_index
        return target

    def _compute_loss(self, criteria, logits, target):
        if not torch.any(target != self.ignore_index):
            return logits.sum() * 0
        return criteria(logits, target)

    def _fuse_logits(self, main_logits, expert_logits, gate_logits):
        target_dtype = main_logits.dtype
        gate_prob = F.softmax(gate_logits, dim=1)[:, 1].unsqueeze(1).to(target_dtype)
        organ_main_logits = main_logits[:, self.organ_class_ids]
        expert_logits = expert_logits.to(target_dtype)
        if self.residual_mode == "interpolate":
            expert_residual = expert_logits - organ_main_logits
        elif self.residual_mode == "centered_logit":
            expert_residual = expert_logits
            if self.center_expert_residual:
                expert_residual = expert_residual - expert_residual.mean(
                    dim=1, keepdim=True
                )
        else:
            raise ValueError(f"Unsupported residual_mode: {self.residual_mode}")
        fused_logits = main_logits.clone()
        update = (
            organ_main_logits + gate_prob * expert_residual * self.residual_scale
        ).to(fused_logits.dtype)
        fused_logits[:, self.organ_class_ids] = update
        return fused_logits

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            input_dict["condition"] = input_dict["condition"][0]

        backbone_output = self.backbone(input_dict)
        if not isinstance(backbone_output, dict):
            raise TypeError(
                "OrganAwareResidualSegmentor requires backbone to return a dict "
                "with 'seg_logits' and 'feat'."
            )
        if "seg_logits" not in backbone_output or "feat" not in backbone_output:
            raise KeyError(
                "Backbone output must contain both 'seg_logits' and 'feat' keys."
            )

        main_logits = backbone_output["seg_logits"]
        feat = backbone_output["feat"]
        expert_logits = self.expert_head(feat)
        gate_logits = self.gate_head(feat)
        fused_logits = self._fuse_logits(main_logits, expert_logits, gate_logits)

        if "segment" in input_dict.keys():
            segment = input_dict["segment"]
            expert_target = self._build_expert_target(segment)
            gate_target = self._build_gate_target(segment)

            main_loss = self._compute_loss(self.main_criteria, main_logits, segment)
            expert_loss = self._compute_loss(
                self.expert_criteria, expert_logits, expert_target
            )
            gate_loss = self._compute_loss(self.gate_criteria, gate_logits, gate_target)
            fused_loss = self._compute_loss(self.main_criteria, fused_logits, segment)
            loss = (
                self.main_loss_weight * main_loss
                + self.fused_loss_weight * fused_loss
                + self.expert_loss_weight * expert_loss
                + self.gate_loss_weight * gate_loss
            )

            if self.training:
                return dict(
                    loss=loss,
                    main_loss=main_loss,
                    fused_loss=fused_loss,
                    expert_loss=expert_loss,
                    gate_loss=gate_loss,
                )

            return dict(loss=fused_loss, seg_logits=fused_logits)

        return dict(seg_logits=fused_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_dict):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)
