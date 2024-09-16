"""
SPVCNN

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn

try:
    import torchsparse
    import torchsparse.nn as spnn
    import torchsparse.nn.functional as F
    from torchsparse.nn.utils import get_kernel_offsets
    from torchsparse import PointTensor, SparseTensor
except ImportError:
    torchsparse = None


from pointcept.models.utils import offset2batch
from pointcept.models.builder import MODELS


def initial_voxelize(z):
    pc_hash = F.sphash(torch.floor(z.C).int())
    sparse_hash = torch.unique(pc_hash)
    idx_query = F.sphashquery(pc_hash, sparse_hash)
    counts = F.spcount(idx_query.int(), len(sparse_hash))

    inserted_coords = F.spvoxelize(torch.floor(z.C), idx_query, counts)
    inserted_coords = torch.round(inserted_coords).int()
    inserted_feat = F.spvoxelize(z.F, idx_query, counts)

    new_tensor = SparseTensor(inserted_feat, inserted_coords, 1)
    new_tensor.cmaps.setdefault(new_tensor.stride, new_tensor.coords)
    z.additional_features["idx_query"][1] = idx_query
    z.additional_features["counts"][1] = counts
    return new_tensor


# x: SparseTensor, z: PointTensor
# return: SparseTensor
def point_to_voxel(x, z):
    if (
        z.additional_features is None
        or z.additional_features.get("idx_query") is None
        or z.additional_features["idx_query"].get(x.s) is None
    ):
        pc_hash = F.sphash(
            torch.cat(
                [
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1),
                ],
                1,
            )
        )
        sparse_hash = F.sphash(x.C)
        idx_query = F.sphashquery(pc_hash, sparse_hash)
        counts = F.spcount(idx_query.int(), x.C.shape[0])
        z.additional_features["idx_query"][x.s] = idx_query
        z.additional_features["counts"][x.s] = counts
    else:
        idx_query = z.additional_features["idx_query"][x.s]
        counts = z.additional_features["counts"][x.s]

    inserted_feat = F.spvoxelize(z.F, idx_query, counts)
    new_tensor = SparseTensor(inserted_feat, x.C, x.s)
    new_tensor.cmaps = x.cmaps
    new_tensor.kmaps = x.kmaps

    return new_tensor


# x: SparseTensor, z: PointTensor
# return: PointTensor
def voxel_to_point(x, z, nearest=False):
    if (
        z.idx_query is None
        or z.weights is None
        or z.idx_query.get(x.s) is None
        or z.weights.get(x.s) is None
    ):
        off = spnn.utils.get_kernel_offsets(2, x.s, 1, device=z.F.device)
        old_hash = F.sphash(
            torch.cat(
                [
                    torch.floor(z.C[:, :3] / x.s[0]).int() * x.s[0],
                    z.C[:, -1].int().view(-1, 1),
                ],
                1,
            ),
            off,
        )
        pc_hash = F.sphash(x.C.to(z.F.device))
        idx_query = F.sphashquery(old_hash, pc_hash)
        weights = (
            F.calc_ti_weights(z.C, idx_query, scale=x.s[0]).transpose(0, 1).contiguous()
        )
        idx_query = idx_query.transpose(0, 1).contiguous()
        if nearest:
            weights[:, 1:] = 0.0
            idx_query[:, 1:] = -1
        new_feat = F.spdevoxelize(x.F, idx_query, weights)
        new_tensor = PointTensor(
            new_feat, z.C, idx_query=z.idx_query, weights=z.weights
        )
        new_tensor.additional_features = z.additional_features
        new_tensor.idx_query[x.s] = idx_query
        new_tensor.weights[x.s] = weights
        z.idx_query[x.s] = idx_query
        z.weights[x.s] = weights

    else:
        new_feat = F.spdevoxelize(x.F, z.idx_query.get(x.s), z.weights.get(x.s))
        new_tensor = PointTensor(
            new_feat, z.C, idx_query=z.idx_query, weights=z.weights
        )
        new_tensor.additional_features = z.additional_features

    return new_tensor


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, stride=stride, transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


@MODELS.register_module()
class SPVCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 2, 2, 2, 2, 2, 2, 2),
    ):  # not implement
        super().__init__()

        assert (
            torchsparse is not None
        ), "Please follow `README.md` to install torchsparse.`"
        assert len(layers) % 2 == 0
        assert len(layers) == len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.channels = channels
        self.layers = layers
        self.num_stages = len(layers) // 2

        self.stem = nn.Sequential(
            spnn.Conv3d(in_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels),
            spnn.ReLU(True),
            spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1),
            spnn.BatchNorm(base_channels),
            spnn.ReLU(True),
        )

        self.stage1 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    base_channels, base_channels, ks=2, stride=2, dilation=1
                ),
                ResidualBlock(base_channels, channels[0], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[0], channels[0], ks=3, stride=1, dilation=1)
                for _ in range(layers[0] - 1)
            ]
        )

        self.stage2 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[0], channels[0], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[0], channels[1], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[1], channels[1], ks=3, stride=1, dilation=1)
                for _ in range(layers[1] - 1)
            ]
        )

        self.stage3 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[1], channels[1], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[1], channels[2], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[2], channels[2], ks=3, stride=1, dilation=1)
                for _ in range(layers[2] - 1)
            ]
        )

        self.stage4 = nn.Sequential(
            *[
                BasicConvolutionBlock(
                    channels[2], channels[2], ks=2, stride=2, dilation=1
                ),
                ResidualBlock(channels[2], channels[3], ks=3, stride=1, dilation=1),
            ]
            + [
                ResidualBlock(channels[3], channels[3], ks=3, stride=1, dilation=1)
                for _ in range(layers[3] - 1)
            ]
        )

        self.up1 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[3], channels[4], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[4] + channels[2],
                            channels[4],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[4], channels[4], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[4] - 1)
                    ]
                ),
            ]
        )

        self.up2 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[4], channels[5], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[5] + channels[1],
                            channels[5],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[5], channels[5], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[5] - 1)
                    ]
                ),
            ]
        )

        self.up3 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[5], channels[6], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[6] + channels[0],
                            channels[6],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[6], channels[6], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[6] - 1)
                    ]
                ),
            ]
        )

        self.up4 = nn.ModuleList(
            [
                BasicDeconvolutionBlock(channels[6], channels[7], ks=2, stride=2),
                nn.Sequential(
                    *[
                        ResidualBlock(
                            channels[7] + base_channels,
                            channels[7],
                            ks=3,
                            stride=1,
                            dilation=1,
                        )
                    ]
                    + [
                        ResidualBlock(
                            channels[7], channels[7], ks=3, stride=1, dilation=1
                        )
                        for _ in range(layers[7] - 1)
                    ]
                ),
            ]
        )

        self.classifier = nn.Sequential(nn.Linear(channels[7], out_channels))

        self.point_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(base_channels, channels[3]),
                    nn.BatchNorm1d(channels[3]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(channels[3], channels[5]),
                    nn.BatchNorm1d(channels[5]),
                    nn.ReLU(True),
                ),
                nn.Sequential(
                    nn.Linear(channels[5], channels[7]),
                    nn.BatchNorm1d(channels[7]),
                    nn.ReLU(True),
                ),
            ]
        )

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)

        # x: SparseTensor z: PointTensor
        z = PointTensor(
            feat,
            torch.cat(
                [grid_coord.float(), batch.unsqueeze(-1).float()], dim=1
            ).contiguous(),
        )
        x0 = initial_voxelize(z)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z, nearest=False)
        z0.F = z0.F

        x1 = point_to_voxel(x0, z0)
        x1 = self.stage1(x1)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.up1[0](y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.up3[0](y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        out = self.classifier(z3.F)
        return out
