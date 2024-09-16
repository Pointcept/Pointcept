"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import numpy as np


def assign_feats(sp, x):
    return ME.SparseTensor(
        features=x.float(),
        coordinate_map_key=sp.coordinate_map_key,
        coordinate_manager=sp.coordinate_manager,
    )


class MinkConvBN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        dimension=3,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class MinkConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        bias=False,
        dimension=3,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        if x.F.dtype == torch.float16:
            x = assign_feats(x, x.F.float())
        return x


class MinkDeConvBNRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        bias=False,
        dimension=3,
    ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                bias=bias,
                dimension=dimension,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class MinkResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(MinkResBlock, self).__init__()

        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            bias=False,
            dimension=3,
        )
        self.norm1 = ME.MinkowskiBatchNorm(out_channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            dimension=3,
        )

        self.norm2 = ME.MinkowskiBatchNorm(out_channels)
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        out += residual
        out = self.relu(out)

        return out


class SparseTensorLinear(nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, sp):
        x = self.linear(sp.F)
        return assign_feats(sp, x.float())


class SparseTensorLayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, sp):
        x = self.norm(sp.F)
        return assign_feats(sp, x.float())


class MinkResBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        d_2 = out_channels // 4
        self.conv1 = torch.nn.Sequential(
            SparseTensorLinear(in_channels, d_2, bias=False),
            ME.MinkowskiBatchNorm(d_2),
            ME.MinkowskiReLU(),
        )
        self.unary_2 = torch.nn.Sequential(
            SparseTensorLinear(d_2, out_channels, bias=False),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(),
        )
        self.spconv = ME.MinkowskiConvolution(
            in_channels=d_2,
            out_channels=d_2,
            kernel_size=5,
            stride=1,
            dilation=1,
            bias=False,
            dimension=3,
        )
        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                SparseTensorLinear(in_channels, out_channels, bias=False),
                ME.MinkowskiBatchNorm(out_channels),
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, x):
        # feats: [N, C]
        # xyz: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]
        shortcut = x
        x = self.unary_1(x)
        x = self.spconv(x)
        x = self.unary_2(x)
        shortcut = self.shortcut_op(shortcut)
        x += shortcut
        return x


class MinkResBlock_BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MinkResBlock_BottleNeck, self).__init__()
        bottle_neck = out_channels // 4
        self.conv1x1a = MinkConvBNRelu(
            in_channels, bottle_neck, kernel_size=1, stride=1
        )
        self.conv3x3 = MinkConvBNRelu(bottle_neck, bottle_neck, kernel_size=3, stride=1)
        self.conv1x1b = MinkConvBN(bottle_neck, out_channels, kernel_size=1, stride=1)
        if in_channels != out_channels:
            self.conv1x1c = MinkConvBN(
                in_channels, out_channels, kernel_size=1, stride=1
            )
        else:
            self.conv1x1c = None
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1x1a(x)
        out = self.conv3x3(out)
        out = self.conv1x1b(out)
        if self.conv1x1c is not None:
            residual = self.conv1x1c(residual)
        out = self.relu(out + residual)

        return out
