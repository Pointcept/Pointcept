"""
SparseUNet Driven by SpConv (recommend)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    import spconv.pytorch as spconv
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `README.md` to install spconv2.`')

from timm.models.layers import trunc_normal_
from ..builder import MODELS


def offset2batch(offset):
    return torch.cat([torch.tensor([i] * (o - offset[i - 1])) if i > 0 else
                      torch.tensor([i] * o) for i, o in enumerate(offset)],
                     dim=0).long().to(offset.device)


class BasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self,
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
            self.proj = spconv.SparseSequential(
                nn.Identity()
            )
        else:
            self.proj = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, embed_channels, kernel_size=1, bias=False),
                norm_fn(embed_channels, momentum=0.02)
            )

        self.conv1 = spconv.SubMConv3d(
            in_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(embed_channels)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            embed_channels, embed_channels, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
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


@MODELS.register_module("SpUNet-v1m2")
class SpUNetBase(nn.Module):
    def __init__(self,
                 in_channels,
                 num_classes,
                 base_channels=32,
                 channels=(32, 64, 128, 256, 256, 128, 96, 96),
                 layers=(2, 3, 4, 6, 2, 2, 2, 2),
                 bn_momentum=0.1):
        super().__init__()
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        norm_fn = partial(nn.BatchNorm1d, eps=1e-5, momentum=bn_momentum)
        block = BasicBlock

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, base_channels, kernel_size=5, padding=1, bias=False, indice_key='stem'),
            norm_fn(base_channels, momentum=0.02),
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
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(enc_channels, channels[s], kernel_size=2, stride=2, bias=False,
                                    indice_key=f"spconv{s + 1}"),
                norm_fn(channels[s], momentum=0.02),
                nn.ReLU()
            ))
            self.enc.append(spconv.SparseSequential(OrderedDict([
                # (f"block{i}", block(enc_channels, channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                # if i == 0 else
                (f"block{i}", block(channels[s], channels[s], norm_fn=norm_fn, indice_key=f"subm{s + 1}"))
                for i in range(layers[s])
            ])))

            # decode num_stages
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(channels[len(channels) - s - 2], dec_channels,
                                           kernel_size=2, bias=False, indice_key=f"spconv{s + 1}"),
                norm_fn(dec_channels, momentum=0.02),
                nn.ReLU()
            ))
            self.dec.append(spconv.SparseSequential(OrderedDict([
                (f"block{i}", block(dec_channels + enc_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                if i == 0 else
                (f"block{i}", block(dec_channels, dec_channels, norm_fn=norm_fn, indice_key=f"subm{s}"))
                for i in range(layers[len(channels) - s - 1])
            ])))
            enc_channels = channels[s]
            dec_channels = channels[len(channels) - s - 2]

        self.final = spconv.SubMConv3d(channels[-1], num_classes, kernel_size=1, padding=1, bias=True) \
            if num_classes > 0 else spconv.Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, spconv.SubMConv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, data_dict):
        discrete_coord = data_dict["discrete_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        
        batch = offset2batch(offset)
        sparse_shape = torch.add(torch.max(discrete_coord, dim=0).values, 1).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat([batch.unsqueeze(-1).int(), discrete_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1
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
            skip = skips.pop(-1)
            x = x.replace_feature(torch.cat((x.features, skip.features), dim=1))
            x = self.dec[s](x)

        x = self.final(x)
        return x.features
