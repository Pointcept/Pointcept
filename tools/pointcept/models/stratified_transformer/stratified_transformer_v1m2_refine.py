"""
Stratified Transformer

Modified from https://github.com/dvlab-research/Stratified-Transformer

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from copy import deepcopy
import torch
import torch.nn as nn

try:
    import torch_points_kernels as tp
except ImportError:
    tp = None

try:
    from torch_points3d.modules.KPConv.kernels import KPConvLayer
    from torch_points3d.core.common_modules import FastBatchNorm1d
except ImportError:
    KPConvLayer = None
    FastBatchNorm1d = None

from torch_scatter import scatter_softmax
from timm.models.layers import DropPath, trunc_normal_
from torch_geometric.nn.pool import voxel_grid

try:
    import pointops2.pointops as pointops
except ImportError:
    pointops = None

from pointcept.models.builder import MODELS


def offset2batch(offset):
    return (
        torch.cat(
            [
                (
                    torch.tensor([i] * (o - offset[i - 1]))
                    if i > 0
                    else torch.tensor([i] * o)
                )
                for i, o in enumerate(offset)
            ],
            dim=0,
        )
        .long()
        .to(offset.device)
    )


def grid_sample(coords, batch, size, start, return_p2v=True):
    cluster = voxel_grid(coords, batch, size, start=start)

    if not return_p2v:
        unique, cluster = torch.unique(cluster, sorted=True, return_inverse=True)
        return cluster
    else:
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )

        # obtain p2v_map
        n = unique.shape[0]
        k = counts.max().item()
        p2v_map = cluster.new_zeros(n, k)
        mask = torch.arange(k).cuda().unsqueeze(0) < counts.unsqueeze(-1)
        p2v_map[mask] = torch.argsort(cluster)
        return cluster, p2v_map, counts


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    """

    def __init__(
        self,
        embed_channels,
        num_heads,
        window_size,
        quant_size,
        attn_drop=0.0,
        proj_drop=0.0,
        scale=None,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        qkv_bias=True,
    ):
        super().__init__()
        self.embed_channels = embed_channels
        self.head_channels = embed_channels // num_heads
        self.num_heads = num_heads
        self.scale = scale or self.head_channels**-0.5

        self.window_size = window_size
        self.quant_size = quant_size

        self.rel_query = rel_query
        self.rel_key = rel_key
        self.rel_value = rel_value

        self.quant_grid_length = int((2 * window_size + 1e-4) // quant_size)

        assert self.rel_query and self.rel_key
        if rel_query:
            self.relative_pos_query_table = nn.Parameter(
                torch.zeros(
                    2 * self.quant_grid_length, self.num_heads, self.head_channels, 3
                )
            )
            trunc_normal_(self.relative_pos_query_table, std=0.02)

        if rel_key:
            self.relative_pos_key_table = nn.Parameter(
                torch.zeros(
                    2 * self.quant_grid_length, self.num_heads, self.head_channels, 3
                )
            )
            trunc_normal_(self.relative_pos_query_table, std=0.02)

        if rel_value:
            self.relative_pos_value_table = nn.Parameter(
                torch.zeros(
                    2 * self.quant_grid_length, self.num_heads, self.head_channels, 3
                )
            )
            trunc_normal_(self.relative_pos_query_table, std=0.02)

        self.qkv = nn.Linear(embed_channels, embed_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(embed_channels, embed_channels)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feats, coords, index_0, index_1, index_0_offsets, n_max):
        n, c = feats.shape
        m = index_0.shape[0]

        assert index_0.shape[0] == index_1.shape[0]

        qkv = (
            self.qkv(feats)
            .reshape(n, 3, self.num_heads, c // self.num_heads)
            .permute(1, 0, 2, 3)
            .contiguous()
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale
        attn_flat = pointops.attention_step1_v2(
            query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max
        )

        # Position embedding
        relative_position = coords[index_0] - coords[index_1]
        relative_position = torch.round(relative_position * 100000) / 100000
        relative_position_index = torch.div(
            relative_position + 2 * self.window_size - 1e-4,
            self.quant_size,
            rounding_mode="trunc",
        )
        # relative_position_index = (relative_position + 2 * self.window_size - 1e-4) // self.quant_size
        assert (relative_position_index >= 0).all()
        assert (relative_position_index <= 2 * self.quant_grid_length - 1).all()

        if self.rel_query and self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx_v3(
                query.float(),
                index_0_offsets.int(),
                n_max,
                key.float(),
                index_1.int(),
                self.relative_pos_query_table.float(),
                self.relative_pos_key_table.float(),
                relative_position_index.int(),
            )
        elif self.rel_query:
            relative_position_bias = pointops.dot_prod_with_idx(
                query.float(),
                index_0.int(),
                self.relative_pos_query_table.float(),
                relative_position_index.int(),
            )  # [M, num_heads]
        elif self.rel_key:
            relative_position_bias = pointops.dot_prod_with_idx(
                key.float(),
                index_1.int(),
                self.relative_pos_key_table.float(),
                relative_position_index.int(),
            )  # [M, num_heads]
        else:
            relative_position_bias = 0.0

        attn_flat += relative_position_bias
        softmax_attn_flat = scatter_softmax(src=attn_flat, index=index_0, dim=0)

        if self.rel_value:
            x = pointops.attention_step2_with_rel_pos_value_v2(
                softmax_attn_flat.float(),
                value.float(),
                index_0_offsets.int(),
                n_max,
                index_1.int(),
                self.relative_pos_value_table.float(),
                relative_position_index.int(),
            )
        else:
            x = pointops.attention_step2(
                softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int()
            )

        x = x.view(n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, drop=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        num_heads,
        window_size,
        quant_size,
        mlp_expend_ratio=4.0,
        drop_path=0.0,
        qk_scale=None,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        qkv_bias=True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_channels)
        self.attn = WindowAttention(
            embed_channels,
            num_heads,
            window_size,
            quant_size,
            scale=qk_scale,
            rel_query=rel_query,
            rel_key=rel_key,
            rel_value=rel_value,
            qkv_bias=qkv_bias,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(embed_channels)
        self.mlp = MLP(
            in_channels=embed_channels,
            hidden_channels=int(embed_channels * mlp_expend_ratio),
        )

    def forward(self, feats, coords, index_0, index_1, index_0_offsets, n_max):
        short_cut = feats
        feats = self.norm1(feats)
        feats = self.attn(feats, coords, index_0, index_1, index_0_offsets, n_max)

        feats = short_cut + self.drop_path(feats)
        feats += self.drop_path(self.mlp(self.norm2(feats)))
        return feats


class BasicLayer(nn.Module):
    def __init__(
        self,
        embed_channels,
        out_channels,
        depth,
        num_heads,
        window_size,
        quant_size,
        mlp_expend_ratio=4.0,
        down_ratio=0.25,
        down_num_sample=16,
        drop_path=None,
        qk_scale=None,
        down=True,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        qkv_bias=True,
    ):
        super().__init__()
        self.depth = depth
        self.window_size = window_size
        self.quant_size = quant_size
        self.down_ratio = down_ratio

        if isinstance(drop_path, list):
            drop_path = drop_path
            assert len(drop_path) == depth
        elif isinstance(drop_path, float):
            drop_path = [deepcopy(drop_path) for _ in range(depth)]
        else:
            drop_path = [0.0 for _ in range(depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels,
                num_heads,
                window_size,
                quant_size,
                mlp_expend_ratio=mlp_expend_ratio,
                drop_path=drop_path[i],
                qk_scale=qk_scale,
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                qkv_bias=qkv_bias,
            )
            self.blocks.append(block)

        self.down = (
            TransitionDown(embed_channels, out_channels, down_ratio, down_num_sample)
            if down
            else None
        )

    def forward(self, feats, coords, offset):
        # window_size -> [window_size, window_size, window_size]
        window_size = torch.tensor(
            [self.window_size] * 3, dtype=coords.dtype, device=coords.device
        )
        new_window_size = 2 * torch.tensor(
            [self.window_size] * 3, dtype=coords.dtype, device=coords.device
        )
        batch = offset2batch(offset)

        # compute new offset
        new_offset = [int(offset[0].item() * self.down_ratio) + 1]
        count = int(offset[0].item() * self.down_ratio) + 1
        for i in range(1, offset.shape[0]):
            count += (
                int((offset[i].item() - offset[i - 1].item()) * self.down_ratio) + 1
            )
            new_offset.append(count)
        new_offset = torch.cuda.IntTensor(new_offset)
        down_idx = pointops.furthestsampling(coords, offset.int(), new_offset.int())

        # compute window mapping
        coords_min = coords.min(0).values
        v2p_map, p2v_map, counts = grid_sample(coords, batch, window_size, start=None)
        shift_size = window_size * 1 / 2
        shift_v2p_map, shift_p2v_map, shift_counts = grid_sample(
            coords + shift_size, batch, window_size, start=coords_min
        )

        new_v2p_map, new_p2v_map, new_counts = grid_sample(
            coords, batch, new_window_size, start=None
        )
        shift_size = new_window_size * 1 / 2
        shift_new_v2p_map, shift_new_p2v_map, shift_new_counts = grid_sample(
            coords + shift_size, batch, new_window_size, start=coords_min
        )

        # stratified attention
        for i, blk in enumerate(self.blocks):
            p2v_map_blk = p2v_map if i % 2 == 0 else shift_p2v_map
            counts_blk = counts if i % 2 == 0 else shift_counts

            new_p2v_map_blk = new_p2v_map if i % 2 == 0 else shift_new_p2v_map
            new_counts_blk = new_counts if i % 2 == 0 else shift_new_counts

            n, k = p2v_map_blk.shape
            mask = torch.arange(k).unsqueeze(0).cuda() < counts_blk.unsqueeze(-1)
            mask_mat = mask.unsqueeze(-1) & mask.unsqueeze(-2)
            index_0 = p2v_map_blk.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
            index_1 = p2v_map_blk.unsqueeze(1).expand(-1, k, -1)[mask_mat]

            down_mask = torch.zeros_like(batch).bool()
            down_mask[down_idx.long()] = True
            down_mask = down_mask[new_p2v_map_blk]  # [n, k], down sample mask
            n, k = new_p2v_map_blk.shape
            mask = torch.arange(k).unsqueeze(0).cuda() < new_counts_blk.unsqueeze(
                -1
            )  # [n, k]
            down_mask = down_mask & mask  # down sample and window mask
            # [n, k, k] query: dense point in large windows; key: sparse point in large windows
            mask_mat = mask.unsqueeze(-1) & down_mask.unsqueeze(-2)

            if i % 2 == 0:
                # [n, k, 3]
                # window_coord = (coords[new_p2v_map_blk] - coords_min) // window_size
                window_coord = torch.div(
                    coords[new_p2v_map_blk] - coords_min,
                    window_size,
                    rounding_mode="trunc",
                )
            else:
                # [n, k, 3]
                # window_coord = (coords[new_p2v_map_blk] - coords_min + 1/2 * window_size) // window_size
                window_coord = torch.div(
                    coords[new_p2v_map_blk] - coords_min + 1 / 2 * window_size,
                    window_size,
                    rounding_mode="trunc",
                )
            # [n, k, k], whether pair points are in same small windows
            mask_mat_prev = (
                window_coord.unsqueeze(2) != window_coord.unsqueeze(1)
            ).any(-1)
            mask_mat = mask_mat & mask_mat_prev

            new_index_0 = new_p2v_map_blk.unsqueeze(-1).expand(-1, -1, k)[mask_mat]
            new_index_1 = new_p2v_map_blk.unsqueeze(1).expand(-1, k, -1)[mask_mat]

            index_0 = torch.cat([index_0, new_index_0], 0)
            index_1 = torch.cat([index_1, new_index_1], 0)

            # rearrange index for acceleration
            index_0, indices = torch.sort(index_0)
            index_1 = index_1[indices]
            index_0_counts = index_0.bincount()
            n_max = index_0_counts.max()
            index_0_offsets = index_0_counts.cumsum(dim=-1)
            index_0_offsets = torch.cat(
                [torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0
            )

            feats = blk(feats, coords, index_0, index_1, index_0_offsets, n_max)

        if self.down:
            feats_down, coords_down, offset_down = self.down(feats, coords, offset)
        else:
            feats_down, coords_down, offset_down = None, None, None

        return feats, coords, offset, feats_down, coords_down, offset_down


class TransitionDown(nn.Module):
    def __init__(self, in_channels, out_channels, ratio, k, norm_layer=nn.LayerNorm):
        super().__init__()
        self.ratio = ratio
        self.k = k
        self.norm = norm_layer(in_channels) if norm_layer else None
        self.linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pool = nn.MaxPool1d(k)

    def forward(self, feats, coords, offset):
        new_offset, count = [int(offset[0].item() * self.ratio) + 1], int(
            offset[0].item() * self.ratio
        ) + 1
        for i in range(1, offset.shape[0]):
            count += ((offset[i].item() - offset[i - 1].item()) * self.ratio) + 1
            new_offset.append(count)
        new_offset = torch.cuda.IntTensor(new_offset)
        idx = pointops.furthestsampling(coords, offset, new_offset)  # (m)
        new_coords = coords[idx.long(), :]  # (m, 3)

        feats = pointops.queryandgroup(
            self.k, coords, new_coords, feats, None, offset, new_offset, use_xyz=False
        )  # (m, nsample, 3+c)
        m, k, c = feats.shape
        feats = (
            self.linear(self.norm(feats.view(m * k, c)).view(m, k, c))
            .transpose(1, 2)
            .contiguous()
        )
        feats = self.pool(feats).squeeze(-1)  # (m, c)
        return feats, new_coords, new_offset


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.linear1 = nn.Sequential(
            nn.LayerNorm(out_channels), nn.Linear(out_channels, out_channels)
        )

        self.linear2 = nn.Sequential(
            nn.LayerNorm(in_channels), nn.Linear(in_channels, out_channels)
        )

    def forward(self, feats, coords, offset, skip_feats, skip_coords, skip_offset):
        feats = self.linear1(skip_feats) + pointops.interpolation(
            coords, skip_coords, self.linear2(feats), offset, skip_offset
        )
        return feats, skip_coords, skip_offset


class KPConvSimpleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_grid_size,
        sigma=1.0,
        negative_slope=0.2,
        bn_momentum=0.02,
    ):
        super().__init__()
        self.kpconv = KPConvLayer(
            in_channels,
            out_channels,
            point_influence=prev_grid_size * sigma,
            add_one=False,
        )
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # coords: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.activation(self.bn(feats))
        return feats


class KPConvResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        prev_grid_size,
        sigma=1.0,
        negative_slope=0.2,
        bn_momentum=0.02,
    ):
        super().__init__()
        d_2 = out_channels // 4
        activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.unary_1 = torch.nn.Sequential(
            nn.Linear(in_channels, d_2, bias=False),
            FastBatchNorm1d(d_2, momentum=bn_momentum),
            activation,
        )
        self.unary_2 = torch.nn.Sequential(
            nn.Linear(d_2, out_channels, bias=False),
            FastBatchNorm1d(out_channels, momentum=bn_momentum),
            activation,
        )
        self.kpconv = KPConvLayer(
            d_2, d_2, point_influence=prev_grid_size * sigma, add_one=False
        )
        self.bn = FastBatchNorm1d(out_channels, momentum=bn_momentum)
        self.activation = activation

        if in_channels != out_channels:
            self.shortcut_op = torch.nn.Sequential(
                nn.Linear(in_channels, out_channels, bias=False),
                FastBatchNorm1d(out_channels, momentum=bn_momentum),
            )
        else:
            self.shortcut_op = nn.Identity()

    def forward(self, feats, xyz, batch, neighbor_idx):
        # feats: [N, C]
        # coords: [N, 3]
        # batch: [N,]
        # neighbor_idx: [N, M]

        shortcut = feats
        feats = self.unary_1(feats)
        feats = self.kpconv(xyz, xyz, neighbor_idx, feats)
        feats = self.unary_2(feats)
        shortcut = self.shortcut_op(shortcut)
        feats += shortcut
        return feats


@MODELS.register_module("ST-v1m2")
class StratifiedTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        channels=(48, 96, 192, 384, 384),
        num_heads=(6, 12, 24, 24),
        depths=(3, 9, 3, 3),
        window_size=(0.2, 0.4, 0.8, 1.6),
        quant_size=(0.01, 0.02, 0.04, 0.08),
        mlp_expend_ratio=4.0,
        down_ratio=0.25,
        down_num_sample=16,
        kp_ball_radius=2.5 * 0.02,
        kp_max_neighbor=34,
        kp_grid_size=0.02,
        kp_sigma=1.0,
        drop_path_rate=0.2,
        rel_query=True,
        rel_key=True,
        rel_value=True,
        qkv_bias=True,
        stem=True,
    ):
        super().__init__()
        assert (
            KPConvLayer is not None and FastBatchNorm1d is not None
        ), "Please make sure torch_points3d is installed"
        assert tp is not None, "Please make sure torch_points_kernels is installed"
        assert pointops is not None, "Please make sure pointops2 is installed"
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.kp_ball_radius = kp_ball_radius
        self.kp_max_neighbor = kp_max_neighbor
        self.stem = stem
        if stem:
            self.point_embed = nn.ModuleList(
                [
                    KPConvSimpleBlock(
                        in_channels, channels[0], kp_grid_size, sigma=kp_sigma
                    ),
                    KPConvResBlock(
                        channels[0], channels[0], kp_grid_size, sigma=kp_sigma
                    ),
                ]
            )
            self.down = TransitionDown(
                channels[0], channels[1], down_ratio, down_num_sample
            )
        else:
            assert channels[0] == channels[1]
            self.point_embed = nn.ModuleList(
                [
                    KPConvSimpleBlock(
                        in_channels, channels[1], kp_grid_size, sigma=kp_sigma
                    ),
                ]
            )

        num_layers = len(depths)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = BasicLayer(
                embed_channels=channels[i + 1],
                out_channels=channels[i + 2] if i < num_layers - 1 else channels[i + 1],
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                quant_size=quant_size[i],
                mlp_expend_ratio=mlp_expend_ratio,
                down_ratio=down_ratio,
                down_num_sample=down_num_sample,
                drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                rel_query=rel_query,
                rel_key=rel_key,
                rel_value=rel_value,
                qkv_bias=qkv_bias,
                down=True if i < num_layers - 1 else False,
            )
            self.layers.append(layer)

        self.up = nn.ModuleList(
            [
                TransitionUp(channels[i + 1], channels[i])
                for i in reversed(range(1, num_layers))
            ]
        )
        if self.stem:
            self.up.append(TransitionUp(channels[1], channels[0]))

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes),
        )

        self.init_weights()

    def forward(self, data_dict):
        feats = data_dict["feat"]
        coords = data_dict["coord"]
        offset = data_dict["offset"].int()
        batch = offset2batch(offset)
        neighbor_idx = tp.ball_query(
            self.kp_ball_radius,
            self.kp_max_neighbor,
            coords,
            coords,
            mode="partial_dense",
            batch_x=batch,
            batch_y=batch,
        )[0]

        feats_stack = []
        coords_stack = []
        offset_stack = []

        for i, layer in enumerate(self.point_embed):
            feats = layer(feats, coords, batch, neighbor_idx)

        feats = feats.contiguous()
        if self.stem:
            feats_stack.append(feats)
            coords_stack.append(coords)
            offset_stack.append(offset)
            feats, coords, offset = self.down(feats, coords, offset)

        for i, layer in enumerate(self.layers):
            feats, coords, offset, feats_down, coords_down, offset_down = layer(
                feats, coords, offset
            )

            feats_stack.append(feats)
            coords_stack.append(coords)
            offset_stack.append(offset)

            feats = feats_down
            coords = coords_down
            offset = offset_down

        feats = feats_stack.pop()
        coords = coords_stack.pop()
        offset = offset_stack.pop()

        for i, up in enumerate(self.up):
            feats, coords, offset = up(
                feats,
                coords,
                offset,
                feats_stack.pop(),
                coords_stack.pop(),
                offset_stack.pop(),
            )

        out = self.classifier(feats)
        return out

    def init_weights(self):
        """Initialize the weights in backbone."""

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
