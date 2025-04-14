"""
PointGroup operations
Written by Li Jiang
"""

import torch
from torch.autograd import Function

# import pointgroup_ops_ext
import pointgroup_ops_sp_cuda as pointgroup_ops_ext


class Voxelization_Idx(Function):

    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        """
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        """
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.IntTensor(N).zero_()
        output_map = input_map.new()

        pointgroup_ops_ext.voxelize_idx(
            coords, output_coords, input_map, output_map, batchsize, mode
        )
        return output_coords, input_map, output_map

    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None


voxelization_idx = Voxelization_Idx.apply


class Voxelization(Function):

    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        """
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        """
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        pointgroup_ops_ext.voxelize_fp(
            feats, output_feats, map_rule, mode, M, maxActive, C
        )
        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        pointgroup_ops_ext.voxelize_bp(
            d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C
        )
        return d_feats, None, None


voxelization = Voxelization.apply


class PointRecover(Function):

    @staticmethod
    def forward(ctx, feats, map_rule, nPoint):
        """
        :param ctx:
        :param feats: cuda float M * C
        :param map_rule: cuda int M * (maxActive + 1)
        :param nPoint: int
        :return: output_feats: cuda float N * C
        """
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        M, C = feats.size()
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(nPoint, C).zero_()

        ctx.for_backwards = (map_rule, maxActive, M)

        pointgroup_ops_ext.point_recover_fp(
            feats, output_feats, map_rule, M, maxActive, C
        )

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, maxActive, M = ctx.for_backwards
        N, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(M, C).zero_()

        pointgroup_ops_ext.point_recover_bp(
            d_output_feats.contiguous(), d_feats, map_rule, M, maxActive, C
        )

        return d_feats, None, None


point_recover = PointRecover.apply


class BallQueryBatchP(Function):

    @staticmethod
    def forward(ctx, coords, batch_idxs, batch_offsets, radius, meanActive):
        """
        :param ctx:
        :param coords: (n, 3) float
        :param batch_idxs: (n) int
        :param batch_offsets: (B+1) int
        :param radius: float
        :param meanActive: int
        :return: idx (nActive), int
        :return: start_len (n, 2), int
        """

        n = coords.size(0)

        assert coords.is_contiguous() and coords.is_cuda
        assert batch_idxs.is_contiguous() and batch_idxs.is_cuda
        assert batch_offsets.is_contiguous() and batch_offsets.is_cuda

        while True:
            idx = torch.cuda.IntTensor(n * meanActive).zero_()
            start_len = torch.cuda.IntTensor(n, 2).zero_()
            nActive = pointgroup_ops_ext.ballquery_batch_p(
                coords, batch_idxs, batch_offsets, idx, start_len, n, meanActive, radius
            )
            if nActive <= n * meanActive:
                break
            meanActive = int(nActive // n + 1)
        idx = idx[:nActive]

        return idx, start_len

    @staticmethod
    def backward(ctx, a=None, b=None):
        return None, None, None


ballquery_batch_p = BallQueryBatchP.apply


class BFSCluster(Function):

    @staticmethod
    def forward(ctx, semantic_label, ball_query_idxs, start_len, threshold):
        """
        :param ctx:
        :param semantic_label: (N), int
        :param ball_query_idxs: (nActive), int
        :param start_len: (N, 2), int
        :return: cluster_idxs:  int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for corresponding point idxs in N
        :return: cluster_offsets: int (nCluster + 1)
        """

        N = start_len.size(0)

        assert semantic_label.is_contiguous()
        assert ball_query_idxs.is_contiguous()
        assert start_len.is_contiguous()

        cluster_idxs = semantic_label.new()
        cluster_offsets = semantic_label.new()

        pointgroup_ops_ext.bfs_cluster(
            semantic_label,
            ball_query_idxs,
            start_len,
            cluster_idxs,
            cluster_offsets,
            N,
            threshold,
        )

        return cluster_idxs, cluster_offsets

    @staticmethod
    def backward(ctx, a=None):
        return None


bfs_cluster = BFSCluster.apply


class RoiPool(Function):

    @staticmethod
    def forward(ctx, feats, proposals_offset):
        """
        :param ctx:
        :param feats: (sumNPoint, C) float
        :param proposals_offset: (nProposal + 1) int
        :return: output_feats (nProposal, C) float
        """
        nProposal = proposals_offset.size(0) - 1
        sumNPoint, C = feats.size()

        assert feats.is_contiguous()
        assert proposals_offset.is_contiguous()

        output_feats = torch.cuda.FloatTensor(nProposal, C).zero_()
        output_maxidx = torch.cuda.IntTensor(nProposal, C).zero_()

        pointgroup_ops_ext.roipool_fp(
            feats, proposals_offset, output_feats, output_maxidx, nProposal, C
        )

        ctx.for_backwards = (output_maxidx, proposals_offset, sumNPoint)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        nProposal, C = d_output_feats.size()

        output_maxidx, proposals_offset, sumNPoint = ctx.for_backwards

        d_feats = torch.cuda.FloatTensor(sumNPoint, C).zero_()

        pointgroup_ops_ext.roipool_bp(
            d_feats,
            proposals_offset,
            output_maxidx,
            d_output_feats.contiguous(),
            nProposal,
            C,
        )

        return d_feats, None


roipool = RoiPool.apply


class GetIoU(Function):

    @staticmethod
    def forward(
        ctx, proposals_idx, proposals_offset, instance_labels, instance_pointnum
    ):
        """
        :param ctx:
        :param proposals_idx: (sumNPoint), int
        :param proposals_offset: (nProposal + 1), int
        :param instance_labels: (N), long, 0~total_nInst-1, -100
        :param instance_pointnum: (total_nInst), int
        :return: proposals_iou: (nProposal, total_nInst), float
        """
        nInstance = instance_pointnum.size(0)
        nProposal = proposals_offset.size(0) - 1

        assert proposals_idx.is_contiguous() and proposals_idx.is_cuda
        assert proposals_offset.is_contiguous() and proposals_offset.is_cuda
        assert instance_labels.is_contiguous() and instance_labels.is_cuda
        assert instance_pointnum.is_contiguous() and instance_pointnum.is_cuda

        proposals_iou = torch.cuda.FloatTensor(nProposal, nInstance).zero_()

        pointgroup_ops_ext.get_iou(
            proposals_idx,
            proposals_offset,
            instance_labels,
            instance_pointnum,
            proposals_iou,
            nInstance,
            nProposal,
        )

        return proposals_iou

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


get_iou = GetIoU.apply


class SecMean(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        """
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        """
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        pointgroup_ops_ext.sec_mean(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_mean = SecMean.apply


class SecMin(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        """
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        """
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        pointgroup_ops_ext.sec_min(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_min = SecMin.apply


class SecMax(Function):

    @staticmethod
    def forward(ctx, inp, offsets):
        """
        :param ctx:
        :param inp: (N, C) float
        :param offsets: (nProposal + 1) int
        :return: out (nProposal, C) float
        """
        nProposal = offsets.size(0) - 1
        C = inp.size(1)

        assert inp.is_contiguous()
        assert offsets.is_contiguous()

        out = torch.cuda.FloatTensor(nProposal, C).zero_()

        pointgroup_ops_ext.sec_max(inp, offsets, out, nProposal, C)

        return out

    @staticmethod
    def backward(ctx, a=None):
        return None, None


sec_max = SecMax.apply
