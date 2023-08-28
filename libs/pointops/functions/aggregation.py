import torch
from torch.autograd import Function

from pointops._C import aggregation_forward_cuda, aggregation_backward_cuda


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
        aggregation_forward_cuda(
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
        aggregation_backward_cuda(
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
