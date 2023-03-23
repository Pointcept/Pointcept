import torch
from torch.autograd import Function

from pointops._C import grouping_forward_cuda, grouping_backward_cuda


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
        grouping_forward_cuda(m, nsample, c, input, idx, output)
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
        idx, = ctx.saved_tensors
        m, nsample, c = grad_output.shape
        grad_input = torch.cuda.FloatTensor(n, c).zero_()
        grouping_backward_cuda(m, nsample, c, grad_output, idx, grad_input)
        return grad_input, None


def grouping(idx,
             feat,
             xyz,
             new_xyz=None,
             with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    xyz = torch.cat([xyz, torch.zeros([1, 3]).to(xyz.device)], dim=0)
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, num_sample, c)

    if with_xyz:
        assert new_xyz.is_contiguous()
        mask = torch.sign(idx + 1)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)  # (m, num_sample, 3)
        grouped_xyz = torch.einsum("n s c, n s -> n s c", grouped_xyz, mask)  # (m, num_sample, 3)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat


grouping2 = Grouping.apply
