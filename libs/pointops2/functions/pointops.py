"""
The part of attention operations is written by Xin Lai.
Email: xinlai@cse.cuhk.edu.hk
"""

from typing import Tuple

import torch
from torch.autograd import Function
import torch.nn as nn

import pointops2_cuda as pointops_cuda
import time


class FurthestSampling(Function):
    @staticmethod
    def forward(ctx, xyz, offset, new_offset):
        """
        input: xyz: (n, 3), offset: (b), new_offset: (b)
        output: idx: (m)
        """
        assert xyz.is_contiguous()
        n, b, n_max = xyz.shape[0], offset.shape[0], offset[0]
        for i in range(1, b):
            n_max = max(offset[i] - offset[i - 1], n_max)
        idx = torch.cuda.IntTensor(new_offset[b - 1].item()).zero_()
        tmp = torch.cuda.FloatTensor(n).fill_(1e10)
        pointops_cuda.furthestsampling_cuda(b, n_max, xyz, offset, new_offset, tmp, idx)
        del tmp
        return idx


furthestsampling = FurthestSampling.apply


class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        input: xyz: (n, 3), new_xyz: (m, 3), offset: (b), new_offset: (b)
        output: idx: (m, nsample), dist2: (m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]
        idx = torch.cuda.IntTensor(m, nsample).zero_()
        dist2 = torch.cuda.FloatTensor(m, nsample).zero_()
        pointops_cuda.knnquery_cuda(
            m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2
        )
        return idx, torch.sqrt(dist2)


knnquery = KNNQuery.apply


class Grouping(Function):
    @staticmethod
    def forward(ctx, input, idx):
        """
        input: input: (n, c), idx : (m, nsample)
        output: (m, nsample, c)
        """
        assert input.is_contiguous() and idx.is_contiguous()
        m, nsample, n, c = idx.shape[0], idx.shape[1], input.shape[0], input.shape[1]
        output = torch.cuda.FloatTensor(m, nsample, c)
        pointops_cuda.grouping_forward_cuda(m, nsample, c, input, idx, output)
        ctx.n = n
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (m, c, nsample)
        output: (n, c), None
        """
        n = ctx.n
        (idx,) = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.grouping_backward_cuda(
            m, nsample, c, grad_output, idx, grad_input
        )
        return grad_input, None


grouping = Grouping.apply


class AttentionStep1(Function):
    @staticmethod
    def forward(ctx, q, k, index0, index1):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
        )

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index0.shape[0]
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda(
            N_k, M, h, C, q, k, index0, index1, output
        )
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.save_for_backward(q, k, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """

        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        q, k, index0, index1 = ctx.saved_tensors
        M, h = grad_output.shape

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C // h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C // h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.attention_step1_backward_cuda(
            N_q, M, h, C, grad_output, index0, index1, q, k, grad_q, grad_k
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v7: {}".format(end - start))
        # # input()

        return grad_q, grad_k, None, None


attention_step1 = AttentionStep1.apply


class AttentionStep1_v2(Function):
    @staticmethod
    def forward(ctx, q, k, index1, index0_offsets, n_max):
        """
        input: q: (N, h, C//h), k: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and index0_offsets.is_contiguous()
            and index1.is_contiguous()
        )
        assert n_max <= 1024

        N_q, h, C_div_h = q.shape
        N_k = k.shape[0]
        M = index1.shape[0]
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.attention_step1_forward_cuda_v2(
            N_k, M, h, C, n_max, q, k, index0_offsets, index1, output
        )
        ctx.N_q = N_q
        ctx.N_k = N_k
        ctx.C = C
        ctx.n_max = n_max
        ctx.save_for_backward(q, k, index0_offsets, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """

        N_q = ctx.N_q
        N_k = ctx.N_k
        C = ctx.C
        n_max = ctx.n_max
        q, k, index0_offsets, index1 = ctx.saved_tensors
        M, h = grad_output.shape

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            q.is_contiguous()
            and k.is_contiguous()
            and index0_offsets.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N_q, h, C // h).zero_()
        grad_k = torch.cuda.FloatTensor(N_k, h, C // h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.attention_step1_backward_cuda_v2(
            N_q,
            M,
            h,
            C,
            n_max,
            grad_output,
            index0_offsets,
            index1,
            q,
            k,
            grad_q,
            grad_k,
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v7: {}".format(end - start))
        # # input()

        return grad_q, grad_k, None, None, None


attention_step1_v2 = AttentionStep1_v2.apply


class AttentionStep2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1):
        """
        input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)
        output: output: [N, h, C//h]
        """
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
        )

        M, h = attn.shape
        N_q = index0.max().item() + 1
        N_v, h, C_div_h = v.shape
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(N_q, h, C // h).zero_()
        pointops_cuda.attention_step2_forward_cuda(
            N_q, M, h, C, attn, v, index0, index1, output
        )
        ctx.M = M

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        M = ctx.M
        attn, v, index0, index1 = ctx.saved_tensors
        N_v = v.shape[0]
        N_q, h, C_div_h = grad_output.shape
        C = h * C_div_h

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_v, h, C // h).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.attention_step2_backward_cuda(
            N_q, M, h, C, grad_output, index0, index1, attn, v, grad_attn, grad_v
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v8: {}".format(end - start))
        # # input()

        return grad_attn, grad_v, None, None


attention_step2 = AttentionStep2.apply


class AttentionStep2_v2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1):
        """
        input: attn: (M, h), v: (N, h, C//h), index0: (M), index1: (M)
        output: output: [L, h, C//h]
        """
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
        )

        L = int(index0.max().item()) + 1

        M, h = attn.shape
        N, h, C_div_h = v.shape
        C = int(C_div_h * h)

        output = torch.cuda.FloatTensor(L, h, C // h).zero_()
        pointops_cuda.attention_step2_forward_cuda(
            N, M, h, C, attn, v, index0, index1, output
        )
        ctx.M = M

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (L, h, C//h)
        output: (M, h), (N, h, C//h), None, None
        """
        M = ctx.M
        attn, v, index0, index1 = ctx.saved_tensors
        L, h, C_div_h = grad_output.shape
        N = v.shape[0]
        C = h * C_div_h

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N, h, C // h).zero_()

        pointops_cuda.attention_step2_backward_cuda(
            N, M, h, C, grad_output, index0, index1, attn, v, grad_attn, grad_v
        )
        return grad_attn, grad_v, None, None


attention_step2_v2 = AttentionStep2_v2.apply


class DotProdWithIdx(Function):
    @staticmethod
    def forward(ctx, q, index, table, rel_idx):
        """
        input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert (
            q.is_contiguous()
            and index.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
        )

        N, h, hdim = q.shape
        M = index.shape[0]

        output = torch.cuda.FloatTensor(M, h).zero_()
        pointops_cuda.dot_prod_with_idx_forward_cuda(
            N, M, h, hdim, q, index, table, rel_idx, output
        )
        ctx.save_for_backward(q, index, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index, table, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        assert (
            q.is_contiguous()
            and index.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.dot_prod_with_idx_backward_cuda(
            N, M, h, hdim, grad_output, q, index, table, rel_idx, grad_q, grad_table
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # # input()

        return grad_q, None, grad_table, None


dot_prod_with_idx = DotProdWithIdx.apply


class DotProdWithIdx_v2(Function):
    @staticmethod
    def forward(ctx, q, index_q, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert (
            q.is_contiguous()
            and index_q.is_contiguous()
            and k.is_contiguous()
            and index_k.is_contiguous()
            and table_q.is_contiguous()
            and table_k.is_contiguous()
            and rel_idx.is_contiguous()
        )

        N, h, hdim = q.shape
        M = index_q.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L and index_k.shape[0] == M

        # obtain the mapping from block_idx to m_idx
        rel_idx_merge = (
            rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L**2)
        )  # [M, ]
        sorted_values, sort_indices = torch.sort(rel_idx_merge)
        _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        rel_idx_offsets = torch.cumsum(counts, dim=-1)  # [T,]
        rel_idx_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0
        )  # [T+1,]
        n_max = counts.max()
        T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        output = torch.cuda.FloatTensor(M, h).zero_()
        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v2(
            N,
            M,
            h,
            hdim,
            n_max,
            T,
            q,
            index_q,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            rel_idx_offsets.int(),
            sort_indices.int(),
            output,
        )

        ctx.n_max = n_max
        ctx.T = T
        ctx.save_for_backward(
            q,
            index_q,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            rel_idx_offsets,
            sort_indices,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        (
            q,
            index_q,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            rel_idx_offsets,
            sort_indices,
        ) = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        T, n_max = ctx.T, ctx.n_max

        grad_output = grad_output.contiguous()
        assert (
            q.is_contiguous()
            and index_q.is_contiguous()
            and k.is_contiguous()
            and index_k.is_contiguous()
            and table_q.is_contiguous()
            and table_k.is_contiguous()
            and rel_idx.is_contiguous()
            and rel_idx_offsets.is_contiguous()
            and sort_indices.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.dot_prod_with_idx_backward_cuda_v2(
            N,
            M,
            h,
            hdim,
            n_max,
            T,
            grad_output,
            q,
            index_q,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            rel_idx_offsets.int(),
            sort_indices.int(),
            grad_q,
            grad_k,
            grad_table_q,
            grad_table_k,
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # # input()
        return grad_q, None, grad_k, None, grad_table_q, grad_table_k, None


dot_prod_with_idx_v2 = DotProdWithIdx_v2.apply


class DotProdWithIdx_v3(Function):
    @staticmethod
    def forward(ctx, q, index_q_offsets, n_max, k, index_k, table_q, table_k, rel_idx):
        """
        input: q: (N, h, hdim), index_q: (M), k: (N, h, hdim), index_k: (M), table_q: (L, h, hdim, 3), table_k: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [M, h]
        """
        assert (
            q.is_contiguous()
            and index_q_offsets.is_contiguous()
            and k.is_contiguous()
            and index_k.is_contiguous()
            and table_q.is_contiguous()
            and table_k.is_contiguous()
            and rel_idx.is_contiguous()
        )

        N, h, hdim = q.shape
        M = index_k.shape[0]
        L = table_q.shape[0]
        assert table_k.shape[0] == L

        # # obtain the mapping from block_idx to m_idx
        # rel_idx_merge = rel_idx[:, 0] + rel_idx[:, 1] * L + rel_idx[:, 2] * (L ** 2) #[M, ]
        # sorted_values, sort_indices = torch.sort(rel_idx_merge)
        # _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
        # rel_idx_offsets = torch.cumsum(counts, dim=-1) #[T,]
        # rel_idx_offsets = torch.cat([torch.zeros(1, dtype=torch.long).cuda(), rel_idx_offsets], 0) #[T+1,]
        # n_max = counts.max()
        # T = counts.shape[0]

        # print("M: {}, L: {}, n_max: {}, T: {}".format(M, L, n_max, T))
        # print("rel_idx_merge.shape: {}, sorted_values.shape: {}".format(rel_idx_merge.shape, sorted_values.shape))
        # print("counts.shape: {}".format(counts.shape))

        # print("M: {}, L: {}, n_max: {}".format(M, L, n_max))

        output = torch.cuda.FloatTensor(M, h).zero_()
        # pointops_cuda.dot_prod_with_idx_forward_cuda(N, M, h, hdim, q, index, table, rel_idx, output)
        pointops_cuda.dot_prod_with_idx_forward_cuda_v3(
            N,
            M,
            h,
            hdim,
            n_max,
            q,
            index_q_offsets,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            output,
        )

        ctx.n_max = n_max
        # ctx.T = T
        ctx.save_for_backward(q, index_q_offsets, k, index_k, table_q, table_k, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: [M, h]
        output: (N, h, hdim), None, (L, h, hdim, 3), None
        """
        q, index_q_offsets, k, index_k, table_q, table_k, rel_idx = ctx.saved_tensors
        M, h = grad_output.shape
        N, _, hdim = q.shape
        L = table_q.shape[0]
        n_max = ctx.n_max

        grad_output = grad_output.contiguous()
        assert (
            q.is_contiguous()
            and index_q_offsets.is_contiguous()
            and k.is_contiguous()
            and index_k.is_contiguous()
            and table_q.is_contiguous()
            and table_k.is_contiguous()
            and rel_idx.is_contiguous()
            and grad_output.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_q = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_q = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()
        grad_k = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table_k = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.dot_prod_with_idx_backward_cuda_v3(
            N,
            M,
            h,
            hdim,
            n_max,
            grad_output,
            q,
            index_q_offsets,
            k,
            index_k,
            table_q,
            table_k,
            rel_idx,
            grad_q,
            grad_k,
            grad_table_q,
            grad_table_k,
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v9: {}".format(end - start))
        # # input()
        return grad_q, None, None, grad_k, None, grad_table_q, grad_table_k, None


dot_prod_with_idx_v3 = DotProdWithIdx_v3.apply


class AttentionStep2WithRelPosValue(Function):
    @staticmethod
    def forward(ctx, attn, v, index0, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
        )

        M, h = attn.shape
        N_v, h, hdim = v.shape
        N_q = index0.max().item() + 1

        output = torch.cuda.FloatTensor(N_q, h, hdim).zero_()
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda(
            N_q, M, h, hdim, attn, v, index0, index1, table, rel_idx, output
        )

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.save_for_backward(attn, v, index0, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        attn, v, index0, index1, table, rel_idx = ctx.saved_tensors
        N_q, h, hdim = grad_output.shape
        N_v = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N_v, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda(
            N_q,
            M,
            h,
            hdim,
            grad_output,
            index0,
            index1,
            attn,
            v,
            table,
            rel_idx,
            grad_attn,
            grad_v,
            grad_table,
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}".format(end - start))
        # # input()
        return grad_attn, grad_v, None, None, grad_table, None


attention_step2_with_rel_pos_value = AttentionStep2WithRelPosValue.apply


class AttentionStep2WithRelPosValue_v2(Function):
    @staticmethod
    def forward(ctx, attn, v, index0_offsets, n_max, index1, table, rel_idx):
        """
        input: attn: (M, h), v: (N, h, hdim), index0_offsets: (M), index1: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
        output: output: [N, h, hdim]
        """
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0_offsets.is_contiguous()
            and index1.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
        )

        M, h = attn.shape
        N, h, hdim = v.shape
        # N_q = int(index0_offsets.max().item()) + 1

        output = torch.cuda.FloatTensor(N, h, hdim).zero_()
        pointops_cuda.attention_step2_with_rel_pos_value_forward_cuda_v2(
            N,
            M,
            h,
            hdim,
            n_max,
            attn,
            v,
            index0_offsets,
            index1,
            table,
            rel_idx,
            output,
        )

        # print("attn[:5,:5]: ", attn[:5, :5])

        ctx.n_max = n_max
        ctx.save_for_backward(attn, v, index0_offsets, index1, table, rel_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (N, h, C//h)
        output: (M, h), (N, h, C//h), None, None, (L, h, hdim, 3), None
        """
        n_max = ctx.n_max
        attn, v, index0_offsets, index1, table, rel_idx = ctx.saved_tensors
        N, h, hdim = grad_output.shape
        N = v.shape[0]
        M = attn.shape[0]
        L = table.shape[0]

        # grad_output = grad_output.contiguous()
        # print("grad_output.is_contiguous(): ", grad_output.is_contiguous())
        assert (
            attn.is_contiguous()
            and v.is_contiguous()
            and index0_offsets.is_contiguous()
            and index1.is_contiguous()
            and grad_output.is_contiguous()
            and table.is_contiguous()
            and rel_idx.is_contiguous()
        )

        # print("back: attn[:5,:5]: ", attn[:5, :5])

        # print("attn.shape: {} v.shape: {}, index0_offsets.shape: {}, index1.shape: {}".format(attn.shape, v.shape, index0_offsets.shape, index1.shape))

        grad_attn = torch.cuda.FloatTensor(M, h).zero_()
        grad_v = torch.cuda.FloatTensor(N, h, hdim).zero_()
        grad_table = torch.cuda.FloatTensor(L, h, hdim, 3).zero_()

        # print("attn.shape: {}, grad_attn.shape: {}".format(attn.shape, grad_attn.shape))
        # print("v.shape: {}, grad_v.shape: {}".format(v.shape, grad_v.shape))
        # print("table.shape: {}, grad_table.shape: {}".format(table.shape, grad_table.shape))

        # torch.cuda.synchronize()
        # start = time.time()

        pointops_cuda.attention_step2_with_rel_pos_value_backward_cuda_v2(
            N,
            M,
            h,
            hdim,
            n_max,
            grad_output,
            index0_offsets,
            index1,
            attn,
            v,
            table,
            rel_idx,
            grad_attn,
            grad_v,
            grad_table,
        )

        # torch.cuda.synchronize()
        # end = time.time()
        # print("time v10: {}".format(end - start))

        return grad_attn, grad_v, None, None, None, grad_table, None


attention_step2_with_rel_pos_value_v2 = AttentionStep2WithRelPosValue_v2.apply


def queryandgroup(
    nsample,
    xyz,
    new_xyz,
    feat,
    idx,
    offset,
    new_offset,
    use_xyz=True,
    return_indx=False,
):
    """
    input: xyz: (n, 3), new_xyz: (m, 3), feat: (n, c), idx: (m, nsample), offset: (b), new_offset: (b)
    output: new_feat: (m, c+3, nsample), grouped_idx: (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    if new_xyz is None:
        new_xyz = xyz
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)  # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # (m, nsample, 3)
    # grouped_xyz = grouping(xyz, idx) # (m, nsample, 3)
    # 相对位置
    grouped_xyz -= new_xyz.unsqueeze(1)  # (m, nsample, 3)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, nsample, c)
    # grouped_feat = grouping(feat, idx) # (m, nsample, c)
    if use_xyz:
        if return_indx:
            return torch.cat((grouped_xyz, grouped_feat), -1), idx  # (m, nsample, 3+c)
        else:
            return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        if return_indx:
            return grouped_feat, idx
        else:
            return grouped_feat


def Divide2Patch(nsample, xyz, offset, return_offset=False, anchor_scale=None):
    # nsample: 16  xyz: (n, 3)  offset: (b)
    downsample_scale = anchor_scale or nsample
    new_offset, count = [offset[0].item() // downsample_scale], offset[
        0
    ].item() // downsample_scale
    for i in range(1, offset.shape[0]):
        count += (offset[i].item() - offset[i - 1].item()) // downsample_scale
        new_offset.append(count)
    # print("donw sample scale:", downsample_scale,"offset:", offset, "newoffset:", new_offset)
    new_offset = torch.cuda.IntTensor(new_offset)
    idx = furthestsampling(xyz, offset, new_offset)  # (m)
    new_xyz = xyz[idx.long()]
    p_idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)  # (m, nsample)
    if return_offset:
        return p_idx, new_offset
    else:
        return p_idx


class Subtraction(Function):
    @staticmethod
    def forward(ctx, input1, input2, idx):
        """
        input: input1: (n, c), input2: (n, c), idx: (n, nsample)
        output:  (n, nsample, c)
        """
        assert input1.is_contiguous() and input2.is_contiguous()
        n, c = input1.shape
        nsample = idx.shape[-1]
        output = torch.cuda.FloatTensor(n, nsample, c).zero_()
        pointops_cuda.subtraction_forward_cuda(
            n, nsample, c, input1, input2, idx, output
        )
        ctx.save_for_backward(idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, nsample, c)
        output: grad_input1: (n, c), grad_input2: (n, c)
        """
        (idx,) = ctx.saved_tensors
        n, nsample, c = grad_output.shape
        grad_input1 = torch.cuda.FloatTensor(n, c).zero_()
        grad_input2 = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.subtraction_backward_cuda(
            n, nsample, c, idx, grad_output, grad_input1, grad_input2
        )
        return grad_input1, grad_input2, None


subtraction = Subtraction.apply


class Aggregation(Function):
    @staticmethod
    def forward(ctx, input, position, weight, idx):
        """
        input: input: (n, c), position: (n, nsample, c), weight : (n, nsample, c'), idx: (n, nsample)
        output: (n, c)
        """
        assert (
            input.is_contiguous()
            and position.is_contiguous()
            and weight.is_contiguous()
        )
        n, nsample, c = position.shape
        w_c = weight.shape[-1]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.aggregation_forward_cuda(
            n, nsample, c, w_c, input, position, weight, idx, output
        )
        ctx.save_for_backward(input, position, weight, idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_out: (n, c)
        output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight : (n, nsample, c')
        """
        input, position, weight, idx = ctx.saved_tensors
        n, nsample, c = position.shape
        w_c = weight.shape[-1]
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grad_position = torch.cuda.FloatTensor(n, nsample, c).zero_()
        grad_weight = torch.cuda.FloatTensor(n, nsample, w_c).zero_()
        pointops_cuda.aggregation_backward_cuda(
            n,
            nsample,
            c,
            w_c,
            input,
            position,
            weight,
            idx,
            grad_output,
            grad_input,
            grad_position,
            grad_weight,
        )
        return grad_input, grad_position, grad_weight, None


aggregation = Aggregation.apply


def interpolation(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)  # (n, 3), (n, 3)
    dist_recip = 1.0 / (dist + 1e-8)  # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm  # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


def interpolation_v2(xyz, new_xyz, feat, offset, new_offset, k=3):
    """
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()

    idx, _ = knnquery(k, xyz, new_xyz, offset, new_offset)  # (n, 3), (n, 3)

    # print("e3: idx.shape: {}, idx[:5]: {}".format(idx.shape, idx[:5]))

    dist = torch.sqrt(((new_xyz.unsqueeze(1) - xyz[idx.long()]) ** 2).sum(-1) + 1e-8)

    # print("e4: dist.shape: {}, dist[:5]: {}".format(dist.shape, dist[:5]))
    # print("((_-dist)**2).max(): ", ((_-dist)**2).max())
    # input()

    dist_recip = 1.0 / (dist + 1e-8)  # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm  # (n, 3)

    new_feat = torch.cuda.FloatTensor(new_xyz.shape[0], feat.shape[1]).zero_()
    for i in range(k):
        new_feat += feat[idx[:, i].long(), :] * weight[:, i].unsqueeze(-1)
    return new_feat


class Interpolation(Function):
    @staticmethod
    def forward(ctx, xyz, new_xyz, input, offset, new_offset, k=3):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        assert xyz.is_contiguous() and new_xyz.is_contiguous() and input.is_contiguous()
        idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset)  # (n, k), (n, k)
        dist_recip = 1.0 / (dist + 1e-8)  # (n, k)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm  # (n, k)

        n, c, m = new_xyz.shape[0], input.shape[1], input.shape[0]
        output = torch.cuda.FloatTensor(n, c).zero_()
        pointops_cuda.interpolation_forward_cuda(n, c, k, input, idx, weight, output)
        ctx.m, ctx.k = m, k
        ctx.save_for_backward(idx, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: xyz: (m, 3), new_xyz: (n, 3), input: (m, c), offset: (b), new_offset: (b)
        output: (n, c)
        """
        m, k = ctx.m, ctx.k
        idx, weight = ctx.saved_tensors
        n, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(m, c).zero_()
        pointops_cuda.interpolation_backward_cuda(
            n, c, k, grad_output, idx, weight, grad_input
        )
        return None, None, grad_input, None, None, None


interpolation2 = Interpolation.apply
