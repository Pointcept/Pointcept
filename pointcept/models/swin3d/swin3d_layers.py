"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from Swin3D.sparse_dl.attn.attn_coff import (
    SelfAttnAIOFunction,
    PosEmb,
    TableDims,
    IndexMode,
    PrecisionMode,
)
import Swin3D.sparse_dl.knn
from Swin3D.sparse_dl.knn import KNN

from .mink_layers import (
    assign_feats,
    SparseTensorLayerNorm,
    SparseTensorLinear,
)


def query_knn_feature(
    K, src_xyz, query_xyz, src_feat, src_offset, query_offset, return_idx=False
):
    """
    gather feature in the KNN neighborhood
    """
    assert (
        src_xyz.is_contiguous()
        and query_xyz.is_contiguous()
        and src_feat.is_contiguous()
    )
    if query_xyz is None:
        query_xyz = src_xyz
        query_offset = src_offset

    idx, _ = KNN.apply(K, src_xyz, query_xyz, src_offset, query_offset)

    n, m, c = src_xyz.shape[0], query_xyz.shape[0], src_feat.shape[1]
    grouped_feat = src_feat[idx.view(-1).long(), :].view(m, K, c)

    if return_idx:
        return grouped_feat, idx
    else:
        return grouped_feat


def knn_linear_interpolation(
    src_xyz, query_xyz, src_feat, src_offset, query_offset, K=3
):
    """
    interpolation feature using distance in KNN neighborhood
    """
    N, C = query_xyz.shape[0], src_feat.shape[1]
    assert (
        src_xyz.is_contiguous()
        and query_xyz.is_contiguous()
        and src_feat.is_contiguous()
    )
    # (N, K)
    idx, dist = KNN.apply(K, src_xyz, query_xyz, src_offset, query_offset)
    weight = 1.0 / (dist + 1e-8)
    norm = torch.sum(weight, dim=1, keepdim=True)
    weight = weight / norm
    query_feat = torch.zeros((N, C), dtype=src_feat.dtype, device=src_feat.device)
    for i in range(K):
        query_feat += src_feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return query_feat


def sparse_self_attention(
    w_w_id: torch.Tensor, w_sizes: torch.Tensor, protocol: str = "v1"
):
    """
    Args:
        indices [torch.Tensor]: sparse window index with shape [N, 2], N is the total
            number of non-empty voxels with indices (window_id, within_window_id). window_id
            is ordered and starts from 0; within_window_id is a sparse index to indicate the
            offset of kernel_size ** 3.
        feats [torch.Tensor]: sprase features of each non-empty voxel with shape [N, C]
    Outputs:
        [M, 3]: sparse indices of cofficient matrix (window_id, att_a_id, att_b_id). att_a_id
            and att_b_id are the within_window_id
        [M, 1]: the sparse coffient matrix

    Spaces:
        W: total number of windows
        N: total number of input voxels
        M: total number of output cofficients
    """
    w_sizes_2 = w_sizes**2

    # w2n_indices - [W], mapping window index to window global offset in input
    #   space
    w_cumsum = torch.cumsum(w_sizes, dim=-1)
    w2n_indices = torch.cat(
        [torch.zeros(1, dtype=w_cumsum.dtype, device=w_cumsum.device), w_cumsum[:-1]]
    )

    # w2m indices - [W], mapping window index to window global offset in output
    #   space
    w2_cumsum = torch.cumsum(w_sizes_2, dim=-1)
    w2m_indices = torch.cat(
        [torch.zeros(1, dtype=w2_cumsum.dtype, device=w2_cumsum.device), w2_cumsum[:-1]]
    )

    # m2w indices - [M], mapping element global offset to the window index
    m2w_indices = torch.zeros(
        [w2_cumsum[-1]], dtype=w_sizes.dtype, device=w_sizes.device
    )
    m2w_offset = torch.zeros(
        [w2_cumsum[-1]], dtype=w_sizes.dtype, device=w_sizes.device
    )
    m2w_indices[w2m_indices[1:]] = 1
    m2w_offset[w2m_indices[1:]] = w_sizes_2[:-1]
    m2w_indices = torch.cumsum(m2w_indices, dim=-1)
    m2w_offset = torch.cumsum(m2w_offset, dim=-1)

    # m_indices = [M], element global offset in output space
    m_indices = torch.arange(
        0, w2_cumsum[-1], dtype=w_sizes.dtype, device=w_sizes.device
    )

    # m2n_indices - [M], mapping element global offset to the window global offset
    #   in input space
    m2n_indices = w2n_indices[m2w_indices]

    m_offset = m_indices - m2w_offset
    m2w_sizes = w_sizes[m2w_indices]

    # print_log_main("m_offset:", m_offset, m_offset.shape)
    # print_log_main("m2n_indices:", m2n_indices, m2n_indices.shape)

    y_offset = m2n_indices + m_offset % m2w_sizes
    x_offset = m2n_indices + torch.div(m_offset, m2w_sizes, rounding_mode="floor")

    # print_log_main("=================================")
    # print_log_main(w_sizes[:5])
    # print_log_main(x_offset[:50])
    # print_log_main(y_offset[:50])
    # coord = torch.stack([m2w_indices, w_w_id[x_offset], w_w_id[y_offset]], axis=-1)
    if protocol == "v1":
        return x_offset, y_offset
    elif protocol == "v2":
        return x_offset, y_offset, m2w_indices, w_sizes, w2n_indices, w2m_indices


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GridCoordsDown(nn.Module):
    """
    downsample the grid coordinates
    keep the nearest point to the average point of the downsampled grid
    """

    def __init__(self, stride):
        super().__init__()
        self.stride = stride
        self.avg_pool = ME.MinkowskiAvgPooling(
            kernel_size=self.stride, stride=self.stride, dimension=3
        )
        self.unpool = ME.MinkowskiPoolingTranspose(
            kernel_size=stride, stride=stride, dimension=3
        )
        self.max_pool = ME.MinkowskiMaxPooling(
            kernel_size=self.stride, stride=self.stride, dimension=3
        )

    def forward(self, coords_sp, sp, return_map=False):
        device = sp.C.device
        # is_pool = True means pooling map
        # is_pool = False means conv map (query as center)

        N = sp.shape[0]
        avg_coords_sp = self.avg_pool(coords_sp)
        dist_sp = self.unpool(avg_coords_sp) - coords_sp
        dist = dist_sp.F
        dist = -torch.sqrt((dist**2).sum(dim=1)).unsqueeze(1)
        dist_sp = assign_feats(dist_sp, dist)
        min_dist_sp = self.max_pool(dist_sp)
        map_pair = sp.coordinate_manager.kernel_map(
            dist_sp.coordinate_map_key,
            min_dist_sp.coordinate_map_key,
            stride=self.stride,
            kernel_size=self.stride,
            is_pool=True,
        )[0]
        in_map, out_map = map_pair
        broad_min_dist_sp = self.unpool(min_dist_sp)
        mask = (broad_min_dist_sp.F == dist_sp.F).squeeze(1)
        in_map = in_map[mask].long()
        out_map = out_map[mask].long()
        downsample_map = torch.zeros(N, dtype=torch.long, device=device) - 1
        downsample_map[out_map] = in_map
        assert (downsample_map >= 0).all()
        assert (dist_sp.F[downsample_map] == min_dist_sp.F).all()
        new_coords = coords_sp.F[downsample_map]
        new_coords_sp = assign_feats(sp, new_coords)
        if return_map:
            return new_coords_sp, downsample_map
        else:
            return new_coords_sp


def get_offset(batch):
    offset = []
    bs = batch.max() + 1
    for i in range(bs):
        offset.append(torch.sum(batch == i))
    offset = torch.cuda.IntTensor(offset)
    offset = offset.cumsum(dim=0).int()
    return offset


class GridDownsample(nn.Module):
    """
    use stride to downsample voxel
    use grid maxpooling with kernel_size
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sp_pool = ME.MinkowskiMaxPooling(
            kernel_size=kernel_size, stride=stride, dimension=3
        )
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = SparseTensorLayerNorm(in_channels)
        self.linear = SparseTensorLinear(in_channels, out_channels)

    def forward(self, sp, coords_sp):
        sp_down = self.sp_pool(self.linear(self.norm(sp)))
        coords_sp_down = self.coords_pool(coords_sp, sp_down)
        return sp_down, coords_sp_down

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"


class GridKNNDownsample(nn.Module):
    """
    use stride to downsample voxel
    use KNN to do maxpooling
    """

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = 16
        self.sp_pool = ME.MinkowskiMaxPooling(
            kernel_size=stride, stride=stride, dimension=3
        )
        self.coords_pool = GridCoordsDown(stride=stride)
        self.norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(self.k)

    def forward(self, sp, coords_sp):
        # calculate the voxel
        sp_down = self.sp_pool(sp)
        # for downsampled cRSE
        coords_sp_down = self.coords_pool(coords_sp, sp_down)
        offset = get_offset(sp.C[:, 0])
        n_offset = get_offset(sp_down.C[:, 0])

        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        n_xyz = coords_sp_down.F[:, 1:4].detach().contiguous()
        feats = query_knn_feature(self.k, xyz, n_xyz, sp.F, offset, n_offset)
        m, k, c = feats.shape
        feats = (
            self.linear(self.norm(feats.view(m * k, c)).view(m, k, c))
            .transpose(1, 2)
            .contiguous()
        )
        feats = self.pool(feats).squeeze(-1)
        sp = assign_feats(sp_down, feats.float())
        coords_sp = coords_sp_down
        return sp, coords_sp

    def extra_repr(self) -> str:
        return f"kernel_size={self.k}, stride={self.stride}, in_channels={self.in_channels}, out_channels={self.out_channels}"


class Upsample(nn.Module):
    """
    upsample using trilinear interpolation
    follower by attn block according to self.attn
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        window_size,
        quant_size,
        attn=True,
        up_k=3,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels)
        )
        self.linear2 = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels)
        )
        self.up_k = up_k
        self.attn = attn and window_size > 0
        if self.attn:
            self.block = BasicLayer(
                dim=out_channels,
                depth=1,
                num_heads=num_heads,
                window_size=window_size,
                quant_size=quant_size,
                drop_path=0.1,
                downsample=None,
                out_channels=None,
                cRSE=cRSE,
                fp16_mode=fp16_mode,
            )

    def forward(self, sp, coords_sp, sp_up, coords_sp_up):
        feats = sp.F
        support_feats = sp_up.F
        xyz = coords_sp.F[:, 1:4].detach().contiguous()
        support_xyz = coords_sp_up.F[:, 1:4].detach().contiguous()
        offset = get_offset(sp.C[:, 0])
        support_offset = get_offset(sp_up.C[:, 0])

        feats = self.linear1(support_feats) + knn_linear_interpolation(
            xyz, support_xyz, self.linear2(feats), offset, support_offset, K=self.up_k
        )
        sp_up = assign_feats(sp_up, feats)
        if self.attn:
            sp_up, _, _ = self.block(sp_up, coords_sp_up)
        return sp_up

    def extra_repr(self) -> str:
        return f"up_k={self.up_k}, in_channels={self.in_channels}, out_channels={self.out_channels}, attn={self.attn}"


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention (W-MSA) module with cRSE.
    Designed for sparse structure
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        quant_size (int): quant_size for for finer cRSE table
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        cRSE (str | 'XYZ', 'XYZ_RGB', 'XYZ_RGB_NORM'): cRSE mode. Default: 'XYZ_RGB'
        fp16_mode (int | 0, 1, 2): fp16 mode for attention module, Default: 0
            0: fp32 forward and fp32 backward
            1: fp16 forward and fp32 backward
            2: fp16 forward and fp16 backward
    """

    def __init__(
        self,
        dim,
        window_size,
        quant_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # color in [-1, 1], color_windowsize = 2
        # normal in [-1, 1], normal_windowsize = 2
        self.color_windowsize = 2
        self.normal_windowsize = 2

        self.fp16_mode = fp16_mode

        table_offsets = []
        self.cRSE = cRSE
        if "XYZ" in cRSE:
            self.xyz_quant_size = quant_size
            quant_grid_length_xyz = window_size * self.xyz_quant_size
            table_shape_xyz = (3, 2 * quant_grid_length_xyz, num_heads, head_dim)
            self.query_xyz_table = nn.Parameter(torch.zeros(table_shape_xyz))
            trunc_normal_(self.query_xyz_table, std=0.02)
            self.key_xyz_table = nn.Parameter(torch.zeros(table_shape_xyz))
            trunc_normal_(self.key_xyz_table, std=0.02)
            self.value_xyz_table = nn.Parameter(torch.zeros(table_shape_xyz))
            trunc_normal_(self.value_xyz_table, std=0.02)
            table_offsets += [np.prod(table_shape_xyz[1:])] * 3

        if "RGB" in cRSE:
            self.color_quant_size = quant_size * 2
            quant_grid_length_rgb = self.color_windowsize * self.color_quant_size
            table_shape_rgb = (3, 2 * quant_grid_length_rgb, num_heads, head_dim)
            self.query_rgb_table = nn.Parameter(torch.zeros(table_shape_rgb))
            trunc_normal_(self.query_rgb_table, std=0.02)
            self.key_rgb_table = nn.Parameter(torch.zeros(table_shape_rgb))
            trunc_normal_(self.key_rgb_table, std=0.02)
            self.value_rgb_table = nn.Parameter(torch.zeros(table_shape_rgb))
            trunc_normal_(self.value_rgb_table, std=0.02)
            table_offsets += [np.prod(table_shape_rgb[1:])] * 3

        if "NORM" in cRSE:
            self.normal_quant_size = quant_size * 2
            quant_grid_length_norm = self.normal_windowsize * self.normal_quant_size
            table_shape_norm = (3, 2 * quant_grid_length_norm, num_heads, head_dim)
            self.query_norm_table = nn.Parameter(torch.zeros(table_shape_norm))
            trunc_normal_(self.query_norm_table, std=0.02)
            self.key_norm_table = nn.Parameter(torch.zeros(table_shape_norm))
            trunc_normal_(self.key_norm_table, std=0.02)
            self.value_norm_table = nn.Parameter(torch.zeros(table_shape_norm))
            trunc_normal_(self.value_norm_table, std=0.02)
            table_offsets += [np.prod(table_shape_norm[1:])] * 3

        self.table_offsets = table_offsets

        self.quant_size = quant_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats: torch.Tensor, attn_args):
        """Forward function.

        Args:
            feats: N, C
            attn_args: arguments for computing attention
        """
        num_v, _ = feats.shape
        num_sc = self.dim // self.num_heads

        (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            n2n_indices,
            w2m_indices,
            n_coords,
        ) = attn_args

        # Query, Key, Value
        qkv = self.qkv(feats)
        qkv = (
            qkv.reshape(num_v, 3, self.num_heads, num_sc)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        query, key, value = qkv[0], qkv[1], qkv[2]  # [N, num_heads, C//num_heads]
        query = query * self.scale

        table_offsets = torch.IntTensor(self.table_offsets).cuda()
        query_table, key_table, value_table = [], [], []
        n_cRSE = []
        if "XYZ" in self.cRSE:
            n_xyz = n_coords[:, 0:3]
            n_xyz = n_xyz * self.quant_size
            n_cRSE.append(n_xyz)
            query_table.append(self.query_xyz_table.view(-1))
            key_table.append(self.key_xyz_table.view(-1))
            value_table.append(self.value_xyz_table.view(-1))
        if "RGB" in self.cRSE:
            n_rgb = n_coords[:, 3:6]
            n_rgb = n_rgb * self.color_quant_size
            n_cRSE.append(n_rgb)
            query_table.append(self.query_rgb_table.view(-1))
            key_table.append(self.key_rgb_table.view(-1))
            value_table.append(self.value_rgb_table.view(-1))
        if "NORM" in self.cRSE:
            n_norm = n_coords[:, 6:9]
            n_norm = n_norm * self.normal_quant_size
            n_cRSE.append(n_norm)
            query_table.append(self.query_norm_table.view(-1))
            key_table.append(self.key_norm_table.view(-1))
            value_table.append(self.value_norm_table.view(-1))

        n_cRSE = torch.cat(n_cRSE, dim=1)

        indices = [m2w_indices, w_sizes, w2m_indices, w2n_indices, n2n_indices, n_cRSE]
        query_table = torch.cat(query_table)
        key_table = torch.cat(key_table)
        value_table = torch.cat(value_table)

        if self.fp16_mode == 0:
            # do not use fp16
            # cast q,k,v to fp32 in forward and backward
            fp16_mode = PrecisionMode.HALF_NONE
        elif self.fp16_mode == 1:
            # use fp16 only in forward
            fp16_mode = PrecisionMode.HALF_FORWARD
        elif self.fp16_mode == 2:
            # use fp16 both in forward and backward
            fp16_mode = PrecisionMode.HALF_ALL

        updated_values = SelfAttnAIOFunction.apply(
            query,
            key,
            value,
            query_table,
            key_table,
            value_table,
            table_offsets,
            indices,
            PosEmb.SEPARATE,
            TableDims.D0,
            IndexMode.INDIRECT,
            fp16_mode,
        )

        updated_values = updated_values.flatten(1)
        updated_feats = updated_values.view(num_v, self.dim)

        updated_feats = self.proj(updated_feats)
        updated_feats = self.proj_drop(updated_feats)  # [N, C]

        return updated_feats


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        quant_size,
        drop_path=0.0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.window_size = window_size

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            quant_size=quant_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            cRSE=cRSE,
            fp16_mode=fp16_mode,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer
        )

    def forward(self, feats, attn_args):
        # feats: [N, c]
        short_cut = feats
        feats = self.norm1(feats)
        feats = self.attn(feats, attn_args)  # [N, c]

        feats = short_cut + self.drop_path(feats)
        feats = feats + self.drop_path(self.mlp(self.norm2(feats)))

        return feats


class BasicLayer(nn.Module):
    """A basic Swin3D layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        quant_size (int): quant_size for for finer cRSE table
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        cRSE (str | 'XYZ', 'XYZ_RGB', 'XYZ_RGB_NORM'): cRSE mode. Default: 'XYZ_RGB'
        fp16_mode (int | 0, 1, 2): fp16 mode for attention module, Default: 0
            0: fp32 forward and fp32 backward
            1: fp16 forward and fp32 backward
            2: fp16 forward and fp16 backward
    """

    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        quant_size,
        out_channels=None,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        down_stride=2,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        self.window_size = window_size
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.quant_size = quant_size
        self.cRSE = cRSE
        self.fp16_mode = fp16_mode

        self.shift_size = window_size // 2
        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads,
                    window_size,
                    quant_size,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    norm_layer=norm_layer,
                    cRSE=cRSE,
                    fp16_mode=fp16_mode,
                )
                for i in range(depth)
            ]
        )

        self.pool = ME.MinkowskiMaxPooling(
            kernel_size=self.window_size, stride=self.window_size, dimension=3
        )

        if downsample is not None:
            if out_channels is None:
                out_channels = dim * 2
            self.downsample = downsample(
                dim, out_channels, kernel_size=down_stride, stride=down_stride
            )
        else:
            self.downsample = None

    def get_map_pair(self, sp):
        """
        use minkowski pool to calculate windows
        get the mapping from voxel to window
        """
        window_size = [self.window_size] * 3
        pool_sp = self.pool(sp)
        windows = pool_sp.C
        window_N = windows.shape[0]

        stride_in = sp.coordinate_map_key.get_tensor_stride()
        x, y, z = [
            torch.arange(window_size[i], device=self.device) * stride_in[i]
            for i in range(3)
        ]
        x, y, z = torch.meshgrid(x, y, z)
        i = torch.zeros_like(x, device=self.device)
        local_window = torch.stack([i, x, y, z], dim=-1).flatten(0, -2)
        all_windows = windows.unsqueeze(1) + local_window.unsqueeze(0)
        all_windows = all_windows.flatten(0, -2).int()
        cm = sp.coordinate_manager
        query_key, (map, inverse_map) = cm.insert_and_map(
            all_windows, tensor_stride=stride_in
        )
        map_pair = cm.kernel_map(query_key, sp.coordinate_map_key, kernel_size=1)[0]
        return map_pair, window_N

    def get_window_mapping(self, sp):
        """
        calculate the relationshape in the window:
        w_w_id: non-empty idx inside the window(sorted by window)
        w_w_xyz: xyz inside the window(sorted by window)
        nempty_num: non-empty voxel number in each window
        sort_idx: sort voxel according to window_id, to gather the point inside the same window
        inv_sort_idx: inverse sort index
        """
        map_pair, window_N = self.get_map_pair(sp)
        window_size = self.window_size
        nW = window_size**3
        in_map, out_map = map_pair
        in_map, sort_idx = torch.sort(in_map)
        # assert out_map == arange(out_map.shape[0])
        out_map = out_map[sort_idx]
        sort_idx = out_map.long()
        inv_sort_idx = torch.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = torch.arange(
            sort_idx.shape[0], dtype=sort_idx.dtype, device=self.device
        )
        N = window_N * nW
        v2w_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        w_id = (
            torch.arange(window_N, dtype=torch.long, device=self.device)
            .unsqueeze(1)
            .repeat(1, nW)
            .view(-1)
        )
        w_w_id = (
            torch.arange(nW, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .repeat(window_N, 1)
            .view(-1)
        )
        v2w_mask[in_map.long()] = True
        nempty_num = v2w_mask.view(-1, nW).sum(dim=-1)
        w_id = w_id[in_map.long()]
        w_w_id = w_w_id[in_map.long()]
        w_w_xyz = torch.stack(
            [
                w_w_id // window_size // window_size,
                w_w_id // window_size % window_size,
                w_w_id % window_size,
            ],
            dim=-1,
        )
        return w_w_id, w_w_xyz, nempty_num, sort_idx, inv_sort_idx

    def get_index01(self, sp, local_xyz, colors):
        """
        calculate the arguments for sparse attention
        """
        (
            w_w_id,
            w_w_xyz,
            nempty_num,
            n2n_indices,
            inv_sort_idx,
        ) = self.get_window_mapping(sp)
        local_xyz = local_xyz[n2n_indices]
        colors = colors[n2n_indices]
        # recover the relative pos in the voxel
        n_coords = w_w_xyz + local_xyz
        n_coords = torch.cat([n_coords, colors], dim=1)
        (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            w2m_indices,
        ) = sparse_self_attention(w_w_id, nempty_num, protocol="v2")
        return (
            x_offset,
            y_offset,
            m2w_indices,
            w_sizes,
            w2n_indices,
            n2n_indices,
            w2m_indices,
            n_coords,
        )

    def get_shifted_sp(self, sp):
        """
        get the shifted sparse tensor for shift-window
        """
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        shift_size = self.shift_size * stride_in[0]
        shifted_C = sp.C.clone()
        shifted_C[:, 1:] += shift_size
        shifted_sp = SparseTensor(
            features=sp.F,
            coordinates=shifted_C,
            device=self.device,
            tensor_stride=stride_in,
        )
        return shifted_sp

    def get_window_pos(self, sp):
        stride_in = sp.coordinate_map_key.get_tensor_stride()
        return (sp.C[:, 1:] / stride_in[0]) % self.window_size

    def forward(self, sp, coords_sp):
        """
        xyz: position of point inside voxel
        colors: other signal for cRSE, include colors and normals
        local_xyz: relative position of point indide voxel(using for finer cRSE table)
        """
        colors = coords_sp.F[:, 4:]
        xyz = coords_sp.F[:, :4]
        local_xyz = (xyz - coords_sp.C)[
            :, 1:
        ] / coords_sp.coordinate_map_key.get_tensor_stride()[0]
        self.device = sp.device
        sp_shift = self.get_shifted_sp(sp)

        attn_args = self.get_index01(sp, local_xyz, colors)
        attn_args_shift = self.get_index01(sp_shift, local_xyz, colors)

        feats = sp.F
        for i, blk in enumerate(self.blocks):
            attn_args_blk = attn_args if i % 2 == 0 else attn_args_shift
            feats = blk(feats, attn_args_blk)  # [N, C]

        sp = assign_feats(sp, feats)
        if self.downsample is not None:
            sp_down, coords_sp = self.downsample(sp, coords_sp)
            return sp, sp_down, coords_sp
        else:
            return sp, sp, coords_sp

    def extra_repr(self) -> str:
        return f"window_size={self.window_size}, depth={self.depth}, channel={self.dim}, num_heads={self.num_heads}, quant_size={self.quant_size}, cRSE={self.cRSE}, fp16_mode={self.fp16_mode}"
