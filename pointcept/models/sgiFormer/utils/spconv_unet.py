"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import functools
from collections import OrderedDict

import spconv.pytorch as spconv
import torch
from spconv.pytorch.modules import SparseModule
from torch import nn

from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch


class ResidualBlock(SparseModule):
    """Resudual block for SpConv U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int: Number of output channels.
        norm_fn (Callable): Normalization function constructor.
        indice_key (str): SpConv key for conv layer.
        normalize_before (bool): Wheter to call norm before conv.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        indice_key=None,
        normalize_before=True,
    ):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
            )
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=indice_key,
                ),
                norm_fn(out_channels),
                nn.ReLU(),
            )

    def forward(self, input):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.

        Returns:
            SparseConvTensor: Output tensor.
        """
        identity = spconv.SparseConvTensor(
            input.features, input.indices, input.spatial_shape, input.batch_size
        )

        output = self.conv_branch(input)
        output = output.replace_feature(
            output.features + self.i_branch(identity).features
        )

        return output


class SpConvUNet(nn.Module):
    """SpConv U-Net model.

    Args:
        num_planes (List[int]): Number of channels in each level.
        norm_fn (Callable): Normalization function constructor.
        block_reps (int): Times to repeat each block.
        block (Callable): Block base class.
        indice_key_id (int): Id of current level.
        normalize_before (bool): Wheter to call norm before conv.
        return_blocks (bool): Whether to return previous blocks.
    """

    def __init__(
        self,
        num_planes,
        norm_fn=functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        block_reps=2,
        block=ResidualBlock,
        indice_key_id=1,
        normalize_before=True,
        return_blocks=False,
    ):
        super().__init__()
        self.return_blocks = return_blocks
        self.num_planes = num_planes

        # process block and norm_fn caller
        if isinstance(block, str):
            area = ["residual", "vgg", "asym"]
            assert block in area, f"block must be in {area}, but got {block}"
            if block == "residual":
                block = ResidualBlock

        blocks = {
            f"block{i}": block(
                num_planes[0],
                num_planes[0],
                norm_fn,
                normalize_before=normalize_before,
                indice_key=f"subm{indice_key_id}",
            )
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(num_planes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(num_planes[0]),
                    nn.ReLU(),
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                )
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        num_planes[0],
                        num_planes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                    norm_fn(num_planes[1]),
                    nn.ReLU(),
                )

            self.u = SpConvUNet(
                num_planes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks,
            )

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(num_planes[1]),
                    nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                )
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        num_planes[1],
                        num_planes[0],
                        kernel_size=2,
                        bias=False,
                        indice_key=f"spconv{indice_key_id}",
                    ),
                    norm_fn(num_planes[0]),
                    nn.ReLU(),
                )

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f"block{i}"] = block(
                    num_planes[0] * (2 - i),
                    num_planes[0],
                    norm_fn,
                    indice_key=f"subm{indice_key_id}",
                    normalize_before=normalize_before,
                )
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs=None):
        """Forward pass.

        Args:
            input (SparseConvTensor): Input tensor.
            previous_outputs (List[SparseConvTensor]): Previous imput tensors.

        Returns:
            SparseConvTensor: Output tensor.
        """
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(
            output.features, output.indices, output.spatial_shape, output.batch_size
        )

        if len(self.num_planes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(
                    output_decoder, previous_outputs
                )
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(
                torch.cat((identity.features, output_decoder.features), dim=1)
            )
            output = self.blocks_tail(output)

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output


@MODELS.register_module("SpUNet-v2m1")
class SpUNetBackbone(nn.Module):
    def __init__(self, in_channels, num_channels, num_planes, return_blocks=True):
        super().__init__()
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                in_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key="subm1",
            )
        )
        self.output_layer = spconv.SparseSequential(
            torch.nn.BatchNorm1d(num_channels, eps=1e-4, momentum=0.1),
            torch.nn.ReLU(inplace=True),
        )
        self.unet = SpConvUNet(num_planes, return_blocks=return_blocks)

    def forward(self, input_dict):
        grid_coord = input_dict["grid_coord"]
        feat = input_dict["feat"]
        offset = input_dict["offset"]

        batch = offset2batch(offset)
        sparse_shape = torch.clip(
            torch.add(torch.max(grid_coord, dim=0).values, 1), 128
        ).tolist()
        x = spconv.SparseConvTensor(
            features=feat,
            indices=torch.cat(
                [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
            ).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=batch[-1].tolist() + 1,
        )
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        return x.features
