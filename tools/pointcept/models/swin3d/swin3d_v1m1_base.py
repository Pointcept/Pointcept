import torch
import torch.nn as nn
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from timm.models.layers import trunc_normal_

from .mink_layers import MinkConvBNRelu, MinkResBlock
from .swin3d_layers import GridDownsample, GridKNNDownsample, BasicLayer, Upsample
from pointcept.models.builder import MODELS
from pointcept.models.utils import offset2batch, batch2offset


@MODELS.register_module("Swin3D-v1m1")
class Swin3DUNet(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        base_grid_size,
        depths,
        channels,
        num_heads,
        window_sizes,
        quant_size,
        drop_path_rate=0.2,
        up_k=3,
        num_layers=5,
        stem_transformer=True,
        down_stride=2,
        upsample="linear",
        knn_down=True,
        cRSE="XYZ_RGB",
        fp16_mode=0,
    ):
        super().__init__()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        if knn_down:
            downsample = GridKNNDownsample
        else:
            downsample = GridDownsample

        self.cRSE = cRSE
        if stem_transformer:
            self.stem_layer = MinkConvBNRelu(
                in_channels=in_channels,
                out_channels=channels[0],
                kernel_size=3,
                stride=1,
            )
            self.layer_start = 0
        else:
            self.stem_layer = nn.Sequential(
                MinkConvBNRelu(
                    in_channels=in_channels,
                    out_channels=channels[0],
                    kernel_size=3,
                    stride=1,
                ),
                MinkResBlock(in_channels=channels[0], out_channels=channels[0]),
            )
            self.downsample = downsample(
                channels[0], channels[1], kernel_size=down_stride, stride=down_stride
            )
            self.layer_start = 1
        self.layers = nn.ModuleList(
            [
                BasicLayer(
                    dim=channels[i],
                    depth=depths[i],
                    num_heads=num_heads[i],
                    window_size=window_sizes[i],
                    quant_size=quant_size,
                    drop_path=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    downsample=downsample if i < num_layers - 1 else None,
                    down_stride=down_stride if i == 0 else 2,
                    out_channels=channels[i + 1] if i < num_layers - 1 else None,
                    cRSE=cRSE,
                    fp16_mode=fp16_mode,
                )
                for i in range(self.layer_start, num_layers)
            ]
        )

        if "attn" in upsample:
            up_attn = True
        else:
            up_attn = False

        self.upsamples = nn.ModuleList(
            [
                Upsample(
                    channels[i],
                    channels[i - 1],
                    num_heads[i - 1],
                    window_sizes[i - 1],
                    quant_size,
                    attn=up_attn,
                    up_k=up_k,
                    cRSE=cRSE,
                    fp16_mode=fp16_mode,
                )
                for i in range(num_layers - 1, 0, -1)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels[0], channels[0]),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Linear(channels[0], num_classes),
        )
        self.num_classes = num_classes
        self.base_grid_size = base_grid_size
        self.init_weights()

    def forward(self, data_dict):
        grid_coord = data_dict["grid_coord"]
        feat = data_dict["feat"]
        coord_feat = data_dict["coord_feat"]
        coord = data_dict["coord"]
        offset = data_dict["offset"]
        batch = offset2batch(offset)
        in_field = ME.TensorField(
            features=torch.cat(
                [
                    batch.unsqueeze(-1),
                    coord / self.base_grid_size,
                    coord_feat / 1.001,
                    feat,
                ],
                dim=1,
            ),
            coordinates=torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1),
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
            device=feat.device,
        )

        sp = in_field.sparse()
        coords_sp = SparseTensor(
            features=sp.F[:, : coord_feat.shape[-1] + 4],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        sp = SparseTensor(
            features=sp.F[:, coord_feat.shape[-1] + 4 :],
            coordinate_map_key=sp.coordinate_map_key,
            coordinate_manager=sp.coordinate_manager,
        )
        sp_stack = []
        coords_sp_stack = []
        sp = self.stem_layer(sp)
        if self.layer_start > 0:
            sp_stack.append(sp)
            coords_sp_stack.append(coords_sp)
            sp, coords_sp = self.downsample(sp, coords_sp)

        for i, layer in enumerate(self.layers):
            coords_sp_stack.append(coords_sp)
            sp, sp_down, coords_sp = layer(sp, coords_sp)
            sp_stack.append(sp)
            assert (coords_sp.C == sp_down.C).all()
            sp = sp_down

        sp = sp_stack.pop()
        coords_sp = coords_sp_stack.pop()
        for i, upsample in enumerate(self.upsamples):
            sp_i = sp_stack.pop()
            coords_sp_i = coords_sp_stack.pop()
            sp = upsample(sp, coords_sp, sp_i, coords_sp_i)
            coords_sp = coords_sp_i

        output = self.classifier(sp.slice(in_field).F)
        return output

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
