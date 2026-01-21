"""
SparseUNet V1M3

Enable Prompt-Driven Normalization for Point Prompt Training

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

from timm.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch


class PDBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features,
        context_channels=256,
        eps=1e-3,
        momentum=0.01,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
        affine=True,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        self.affine = affine
        if self.decouple:
            self.bns = nn.ModuleList(
                [
                    nn.BatchNorm1d(
                        num_features=num_features,
                        eps=eps,
                        momentum=momentum,
                        affine=affine,
                    )
                    for _ in conditions
                ]
            )
        else:
            self.bn = nn.BatchNorm1d(
                num_features=num_features, eps=eps, momentum=momentum, affine=affine
            )
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, feat, condition=None, context=None):
        if self.decouple:
            assert condition in self.conditions
            bn = self.bns[self.conditions.index(condition)]
        else:
            bn = self.bn
        feat = bn(feat)
        if self.adaptive:
            assert context is not None
            shift, scale = self.modulation(context).chunk(2, dim=1)
            feat = feat * (1.0 + scale) + shift
        return feat


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        in_channels,
        embed_channels,
        stride=1,
        norm_fn=None,
        indice_key=None,
        bias=False,
    ):
        super().__init__()

        assert norm_fn is not None

        self.in_channels = in_channels
        self.embed_channels = embed_channels
        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            # TODO remove norm after project
            self.proj_conv = spconv.SubMConv3d(
                in_channels, embed_channels, kernel_size=1, bias=False
            )
            self.proj_norm = norm_fn(embed_channels)

        self.conv1 = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels,
            embed_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(embed_channels)
        self.stride = stride

    def forward(self, x):
        x, condition, context = x
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features, condition, context))

        if self.in_channels == self.embed_channels:
            residual = self.proj(residual)
        else:
            residual = residual.replace_feature(
                self.proj_norm(self.proj_conv(residual).features, condition, context)
            )
        out = out.replace_feature(out.features + residual.features)
        out = out.replace_feature(self.relu(out.features))
        return out, condition, context


class SPConvDown(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out


class SPConvUp(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        indice_key,
        kernel_size=2,
        bias=False,
        norm_fn=None,
    ):
        super().__init__()
        self.conv = spconv.SparseInverseConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out


class SPConvPatchEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, norm_fn=None):
        super().__init__()
        self.conv = spconv.SubMConv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
            indice_key="stem",
        )
        self.bn = norm_fn(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, condition, context = x
        out = self.conv(x)
        out = out.replace_feature(self.bn(out.features, condition, context))
        out = out.replace_feature(self.relu(out.features))
        return out


@MODELS.register_module("SpUNet-v1m3")
class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        zero_init=True,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=False,
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2
        self.enc_mode = enc_mode
        self.conditions = conditions
        self.zero_init = zero_init

        norm_fn = partial(
            PDBatchNorm,
            eps=1e-3,
            momentum=0.01,
            conditions=conditions,
            context_channels=context_channels,
            decouple=norm_decouple,
            adaptive=norm_adaptive,
            affine=norm_affine,
        )
        block = BasicBlock

        self.conv_input = SPConvPatchEmbedding(
            in_channels, base_channels, kernel_size=5, norm_fn=norm_fn
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.enc_mode else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                SPConvDown(
                    enc_channels,
                    channels[s],
                    kernel_size=2,
                    bias=False,
                    indice_key=f"spconv{s + 1}",
                    norm_fn=norm_fn,
                )
            )
            self.enc.append(
                spconv.SparseSequential(
                    OrderedDict(
                        [
                            # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                            # if i == 0 else
                            (
                                f"block{i}",
                                block(
                                    channels[s],
                                    channels[s],
                                    norm_fn=norm_fn,
                                    indice_key=f"subm{s + 1}",
                                ),
                            )
                            for i in range(layers[s])
                        ]
                    )
                )
            )
            if not self.enc_mode:
                # decode num_stages
                self.up.append(
                    SPConvUp(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                        norm_fn=norm_fn,
                    )
                )
                self.dec.append(
                    spconv.SparseSequential(
                        OrderedDict(
                            [
                                (
                                    (
                                        f"block{i}",
                                        block(
                                            dec_channels + enc_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                    if i == 0
                                    else (
                                        f"block{i}",
                                        block(
                                            dec_channels,
                                            dec_channels,
                                            norm_fn=norm_fn,
                                            indice_key=f"subm{s}",
                                        ),
                                    )
                                )
                                for i in range(layers[len(channels) - s - 1])
                            ]
                        )
                    )
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        final_in_channels = (
            channels[-1] if not self.enc_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                final_in_channels, num_classes, kernel_size=1, padding=1, bias=True
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            if m.affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, PDBatchNorm):
            if self.zero_init:
                nn.init.constant_(m.modulation[-1].weight, 0)
                nn.init.constant_(m.modulation[-1].bias, 0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]
        condition = input_dict["condition"][0]
        context = input_dict["context"] if "context" in input_dict.keys() else None

        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 96).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input([x, condition, context])
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s]([x, condition, context])
            x, _, _ = self.enc[s]([x, condition, context])
            skips.append(x)
        x = skips.pop(-1)
        if not self.enc_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s]([x, condition, context])
                skip = skips.pop(-1)
                x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
                x, _, _ = self.dec[s]([x, condition, context])

        x = self.final(x)
        if self.enc_mode:
            x = x.replace_feature(
                scatter(x.features, x.indices[:, 0].long(), reduce="mean", dim=0)
            )
        return x.features
