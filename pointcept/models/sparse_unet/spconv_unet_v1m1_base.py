"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

import spconv.pytorch as spconv
from torch_geometric.utils import scatter

try:
    from timm.layers import trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import trunc_normal_

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch


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

        if in_channels == embed_channels:
            self.proj = spconv.SparseSequential(nn.Identity())
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, embed_channels, kernel_size=1, bias=False
                ),
                norm_fn(embed_channels),
            )

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
        residual = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))

        out = out.replace_feature(out.features + self.proj(residual).features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SparseIdentity(spconv.SparseModule):
    def forward(self, x):
        return x


class MultiScaleSubMBlock(spconv.SparseModule):
    def __init__(
        self,
        channels,
        norm_fn,
        indice_key,
        dilation=2,
        with_residual=True,
    ):
        super().__init__()
        self.with_residual = with_residual
        self.branch_k3 = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key=f"{indice_key}_k3",
            ),
            norm_fn(channels),
            nn.ReLU(),
        )
        self.branch_dilated = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                dilation=dilation,
                padding=dilation,
                bias=False,
                indice_key=f"{indice_key}_d{dilation}",
            ),
            norm_fn(channels),
            nn.ReLU(),
        )
        self.fuse = spconv.SparseSequential(
            spconv.SubMConv3d(
                channels * 2,
                channels,
                kernel_size=1,
                bias=False,
                indice_key=f"{indice_key}_fuse",
            ),
            norm_fn(channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out_k3 = self.branch_k3(x)
        out_d = self.branch_dilated(x)
        x_cat = x.replace_feature(torch.cat((out_k3.features, out_d.features), dim=1))
        out = self.fuse(x_cat)
        if self.with_residual:
            out = out.replace_feature(out.features + x.features)
        return out


class SparseSEBlock(spconv.SparseModule):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch = x.indices[:, 0].long()
        pooled = scatter(x.features, batch, reduce="mean", dim=0)
        weight = self.fc2(self.relu(self.fc1(pooled)))
        weight = self.sigmoid(weight)
        x = x.replace_feature(x.features * weight[batch])
        return x


@MODELS.register_module("SpUNet-v1m1")
class SpUNetBase(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
        return_feature=False,
        ms_stages=(),
        ms_dilation=2,
        se_stages=(),
        se_reduction=16,
        skip_gate=False,
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
        self.return_feature = return_feature
        self.ms_stages = set(ms_stages)
        self.ms_dilation = ms_dilation
        self.se_stages = set(se_stages)
        self.se_reduction = se_reduction
        self.skip_gate = skip_gate

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.enc_ms = nn.ModuleList()
        self.enc_se = nn.ModuleList()
        self.dec = nn.ModuleList() if not self.enc_mode else None
        self.skip_gates = nn.ModuleList() if (not self.enc_mode) else None

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
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
            self.enc_ms.append(
                MultiScaleSubMBlock(
                    channels=channels[s],
                    norm_fn=norm_fn,
                    indice_key=f"ms_enc{s + 1}",
                    dilation=self.ms_dilation,
                )
                if s in self.ms_stages
                else SparseIdentity()
            )
            self.enc_se.append(
                SparseSEBlock(channels=channels[s], reduction=self.se_reduction)
                if s in self.se_stages
                else SparseIdentity()
            )
            if not self.enc_mode:
                # decode num_stages
                self.up.append(
                    spconv.SparseSequential(
                        spconv.SparseInverseConv3d(
                            channels[len(channels) - s - 2],
                            dec_channels,
                            kernel_size=2,
                            bias=False,
                            indice_key=f"spconv{s + 1}",
                        ),
                        norm_fn(dec_channels),
                        nn.ReLU(),
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
                self.skip_gates.append(
                    nn.Sequential(
                        nn.Linear(dec_channels + enc_channels, enc_channels, bias=True),
                        nn.Sigmoid(),
                    )
                    if self.skip_gate
                    else nn.Identity()
                )

            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final_in_channels = (
            channels[-1] if not self.enc_mode else channels[self.num_stages - 1]
        )
        self.final = (
            spconv.SubMConv3d(
                self.final_in_channels,
                num_classes,
                kernel_size=1,
                padding=1,
                bias=True,
            )
            if num_classes > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

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
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            x = self.enc_ms[s](x)
            x = self.enc_se[s](x)
            skips.append(x)
        x = skips.pop(-1)
        if not self.enc_mode:
            # dec forward
            for s in reversed(range(self.num_stages)):
                x = self.up[s](x)
                skip = skips.pop(-1)
                skip_feat = skip.features
                if self.skip_gate:
                    gate_in = torch.cat((x.features, skip_feat), dim=1)
                    skip_feat = skip_feat * self.skip_gates[s](gate_in)
                x = x.replace_feature(torch.cat((x.features, skip_feat), dim=1))
                x = self.dec[s](x)

        feat = x.features
        x = self.final(x)
        seg_logits = x.features
        if self.enc_mode:
            feat = scatter(feat, x.indices[:, 0].long(), reduce="mean", dim=0)
            seg_logits = scatter(
                seg_logits, x.indices[:, 0].long(), reduce="mean", dim=0
            )
        if self.return_feature:
            return dict(seg_logits=seg_logits, feat=feat)
        return seg_logits


@MODELS.register_module()
class SpUNetNoSkipBase(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                base_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            ),
            norm_fn(base_channels),
            nn.ReLU(),
        )

        enc_channels = base_channels
        dec_channels = channels[-1]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()

        for s in range(self.num_stages):
            # encode num_stages
            self.down.append(
                spconv.SparseSequential(
                    spconv.SparseConv3d(
                        enc_channels,
                        channels[s],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(channels[s]),
                    nn.ReLU(),
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

            # decode num_stages
            self.up.append(
                spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        channels[len(channels) - s - 2],
                        dec_channels,
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{s + 1}",
                    ),
                    norm_fn(dec_channels),
                    nn.ReLU(),
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
                                        dec_channels,
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

        self.final = (
            spconv.SubMConv3d(
                channels[-1], out_channels, kernel_size=1, padding=1, bias=True
            )
            if out_channels > 0
            else spconv.Identity()
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.conv_input(x)
        skips = [x]
        # enc forward
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            skips.append(x)
        x = skips.pop(-1)
        # dec forward
        for s in reversed(range(self.num_stages)):
            x = self.up[s](x)
            # skip = skips.pop(-1)
            # x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

        x = self.final(x)
        return x.features
