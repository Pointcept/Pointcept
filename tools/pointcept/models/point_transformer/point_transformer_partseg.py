"""
Point Transformer V1 for Part Segmentation

Might be a bit different from the original paper

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch
import torch.nn as nn
import einops
import pointops

from pointcept.models.builder import MODELS
from .utils import LayerNorm1d


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            LayerNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes),
        )
        self.linear_w = nn.Sequential(
            LayerNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, out_planes // share_planes),
            LayerNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)
        x_k, idx = pointops.knn_query_and_group(
            x_k, p, o, new_xyz=p, new_offset=o, nsample=self.nsample, with_xyz=True
        )
        x_v, _ = pointops.knn_query_and_group(
            x_v,
            p,
            o,
            new_xyz=p,
            new_offset=o,
            idx=idx,
            nsample=self.nsample,
            with_xyz=False,
        )
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]
        p_r = self.linear_p(p_r)
        r_qk = (
            x_k
            - x_q.unsqueeze(1)
            + einops.reduce(
                p_r, "n ns (i j) -> n ns j", reduction="sum", j=self.mid_planes
            )
        )
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum(
            "n t s i, n t i -> n s i",
            einops.rearrange(x_v + p_r, "n ns (s i) -> n ns s i", s=self.share_planes),
            w,
        )
        x = einops.rearrange(x, "n s i -> n (s i)")
        return x


class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.farthest_point_sampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)
            x, _ = pointops.knn_query_and_group(
                x,
                p,
                offset=o,
                new_xyz=n_p,
                new_offset=n_o,
                nsample=self.nsample,
                with_xyz=True,
            )
            x = self.relu(
                self.bn(self.linear(x).transpose(1, 2).contiguous())
            )  # (m, c, nsample)
            x = self.pool(x).squeeze(-1)  # (m, c)
            p, o = n_p, n_o
        else:
            x = self.relu(self.bn(self.linear(x)))  # (n, c)
        return [p, x, o]


class TransitionUp(nn.Module):
    def __init__(self, in_planes, out_planes=None, num_shape_class=None):
        super().__init__()
        if out_planes is None:
            self.num_shape_class = num_shape_class
            if num_shape_class is not None:
                self.linear1 = nn.Sequential(
                    nn.Linear(2 * in_planes + 1024, in_planes),
                    nn.BatchNorm1d(in_planes),
                    nn.ReLU(inplace=True),
                )
            else:
                self.linear1 = nn.Sequential(
                    nn.Linear(2 * in_planes, in_planes),
                    nn.BatchNorm1d(in_planes),
                    nn.ReLU(inplace=True),
                )

            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True)
            )
            if num_shape_class is not None:
                self.linear3 = nn.Sequential(
                    nn.Linear(num_shape_class, 1024), nn.ReLU(inplace=True)
                )
        else:
            self.linear1 = nn.Sequential(
                nn.Linear(out_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True),
            )

    def forward(self, pxo1, pxo2=None, y=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]
                x_b = x[s_i:e_i, :]
                y_b = y[i].unsqueeze(-1).unsqueeze(-1).long()
                y_onehot = torch.zeros(1, self.num_shape_class).cuda()  # (1, l)
                y_onehot.scatter_(1, y_b, 1)  # (1, l)
                x_b = torch.cat(
                    (
                        x_b,
                        self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1),
                        self.linear3(y_onehot).repeat(cnt, 1),
                    ),
                    dim=1,
                )
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            x = self.linear1(x)
        else:
            p1, x1, o1 = pxo1
            p2, x2, o2 = pxo2
            x = self.linear1(x1) + pointops.interpolation(
                p2, p1, self.linear2(x2), o2, o1
            )
        return x


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(Bottleneck, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer = PointTransformerLayer(planes, planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer([p, x, o])))
        x = self.bn3(self.linear3(x))
        x += identity
        x = self.relu(x)
        return [p, x, o]


class PointTransformerSeg(nn.Module):
    def __init__(
        self, block, blocks, in_channels=6, num_classes=50, num_shape_classes=None
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_shape_classes = num_shape_classes
        self.in_planes, planes = in_channels, [32, 64, 128, 256, 512]
        fpn_planes, fpnhead_planes, share_planes = 128, 64, 8
        stride, nsample = [1, 4, 4, 4, 4], [8, 16, 16, 16, 16]
        self.enc1 = self._make_enc(
            block,
            planes[0],
            blocks[0],
            share_planes,
            stride=stride[0],
            nsample=nsample[0],
        )  # N/1
        self.enc2 = self._make_enc(
            block,
            planes[1],
            blocks[1],
            share_planes,
            stride=stride[1],
            nsample=nsample[1],
        )  # N/4
        self.enc3 = self._make_enc(
            block,
            planes[2],
            blocks[2],
            share_planes,
            stride=stride[2],
            nsample=nsample[2],
        )  # N/16
        self.enc4 = self._make_enc(
            block,
            planes[3],
            blocks[3],
            share_planes,
            stride=stride[3],
            nsample=nsample[3],
        )  # N/64
        self.enc5 = self._make_enc(
            block,
            planes[4],
            blocks[4],
            share_planes,
            stride=stride[4],
            nsample=nsample[4],
        )  # N/256
        self.dec5 = self._make_dec(
            block,
            planes[4],
            1,
            share_planes,
            num_shape_classes=num_shape_classes,
            nsample=nsample[4],
            is_head=True,
        )  # transform p5
        self.dec4 = self._make_dec(
            block, planes[3], 1, share_planes, nsample=nsample[3]
        )  # fusion p5 and p4
        self.dec3 = self._make_dec(
            block, planes[2], 1, share_planes, nsample=nsample[2]
        )  # fusion p4 and p3
        self.dec2 = self._make_dec(
            block, planes[1], 1, share_planes, nsample=nsample[1]
        )  # fusion p3 and p2
        self.dec1 = self._make_dec(
            block, planes[0], 1, share_planes, nsample=nsample[0]
        )  # fusion p2 and p1
        self.cls = nn.Sequential(
            nn.Linear(planes[0], planes[0]),
            nn.BatchNorm1d(planes[0]),
            nn.ReLU(inplace=True),
            nn.Linear(planes[0], num_classes),
        )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = [
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _make_dec(
        self,
        block,
        planes,
        blocks,
        share_planes=8,
        num_shape_classes=None,
        nsample=16,
        is_head=False,
    ):
        layers = [
            TransitionUp(
                self.in_planes,
                None if is_head else planes * block.expansion,
                num_shape_classes,
            )
        ]
        self.in_planes = planes * block.expansion
        for _ in range(blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, data_dict):
        p0 = data_dict["coord"]
        x0 = data_dict["feat"]
        o0 = data_dict["offset"].int()
        if self.num_shape_classes is not None:
            y = data_dict["cls_token"]
        p1, x1, o1 = self.enc1([p0, x0, o0])
        p2, x2, o2 = self.enc2([p1, x1, o1])
        p3, x3, o3 = self.enc3([p2, x2, o2])
        p4, x4, o4 = self.enc4([p3, x3, o3])
        p5, x5, o5 = self.enc5([p4, x4, o4])
        if self.num_shape_classes is not None:
            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5], y=y), o5])[1]
        else:
            x5 = self.dec5[1:]([p5, self.dec5[0]([p5, x5, o5]), o5])[1]
        x4 = self.dec4[1:]([p4, self.dec4[0]([p4, x4, o4], [p5, x5, o5]), o4])[1]
        x3 = self.dec3[1:]([p3, self.dec3[0]([p3, x3, o3], [p4, x4, o4]), o3])[1]
        x2 = self.dec2[1:]([p2, self.dec2[0]([p2, x2, o2], [p3, x3, o3]), o2])[1]
        x1 = self.dec1[1:]([p1, self.dec1[0]([p1, x1, o1], [p2, x2, o2]), o1])[1]
        x = self.cls(x1)
        return x


@MODELS.register_module("PointTransformer-PartSeg26")
class PointTransformerSeg26(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg26, self).__init__(
            Bottleneck, [1, 1, 1, 1, 1], **kwargs
        )


@MODELS.register_module("PointTransformer-PartSeg38")
class PointTransformerSeg38(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg38, self).__init__(
            Bottleneck, [1, 2, 2, 2, 2], **kwargs
        )


@MODELS.register_module("PointTransformer-PartSeg50")
class PointTransformerSeg50(PointTransformerSeg):
    def __init__(self, **kwargs):
        super(PointTransformerSeg50, self).__init__(
            Bottleneck, [1, 2, 3, 5, 2], **kwargs
        )
