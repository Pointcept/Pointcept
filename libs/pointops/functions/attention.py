import torch
from torch.autograd import Function

from pointops._C import attention_relation_step_forward_cuda, attention_relation_step_backward_cuda, \
    attention_fusion_step_forward_cuda, attention_fusion_step_backward_cuda


class AttentionRelationStep(Function):
    @staticmethod
    def forward(ctx, query, key, weight, index_target, index_refer):
        """
        input - query: (n, g, c), key: (n, g, c), weight: (c)  1_c for scatter attention,
                index_target: (m), index_refer: (m)
        output - relation: (M, g)
        """

        assert query.is_contiguous() \
               and key.is_contiguous() \
               and index_target.is_contiguous() \
               and index_refer.is_contiguous() \
               and weight.is_contiguous()

        assert index_target.shape[0] == index_refer.shape[0]

        _, g, c = query.shape
        m = index_target.shape[0]
        output = torch.cuda.FloatTensor(m, g).zero_()
        attention_relation_step_forward_cuda(m, g, c, query, key, weight,
                                             index_target.int(), index_refer.int(), output)
        ctx.save_for_backward(query, key, weight, index_target, index_refer)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        query, key, weight, index_target, index_refer = ctx.saved_tensors
        n, g, c = query.shape
        m = index_target.shape[0]
        grad_query = torch.cuda.FloatTensor(n, g, c).zero_()
        grad_key = torch.cuda.FloatTensor(n, g, c).zero_()
        grad_weight = torch.cuda.FloatTensor(c).zero_()
        attention_relation_step_backward_cuda(m, g, c,
                                              query, grad_query,
                                              key, grad_key,
                                              weight, grad_weight,
                                              index_target.int(), index_refer.int(),
                                              grad_output)
        return grad_query, grad_key, None, None, None


class AttentionFusionStep(Function):
    @staticmethod
    def forward(ctx, weight, value, index_target, index_refer):
        """
        input - weight: (m, g), value: (n, g, c)
                index_target: (m), index_value: (m)
        output - output: (n, g, c)
        """

        assert weight.is_contiguous() \
               and value.is_contiguous() \
               and index_target.is_contiguous() \
               and index_refer.is_contiguous() \
               and weight.is_contiguous()

        assert index_target.shape[0] == index_refer.shape[0]

        n, g, c = value.shape
        m = index_refer.shape[0]
        output = torch.cuda.FloatTensor(n, g, c).zero_()
        attention_fusion_step_forward_cuda(m, g, c, weight, value, index_target.int(), index_refer.int(), output)
        ctx.save_for_backward(weight, value, index_target, index_refer)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        input: grad_output: (n, g, c)
        output: grad_weight: (m, g), grad_value: (n, g, c), none, none
        """
        weight, value, index_target, index_refer = ctx.saved_tensors
        n, g, c = value.shape
        m = index_target.shape[0]
        grad_weight = torch.cuda.FloatTensor(m, g).zero_()
        grad_value = torch.cuda.FloatTensor(n, g, c).zero_()
        attention_fusion_step_backward_cuda(m, g, c,
                                            weight, grad_weight,
                                            value, grad_value,
                                            index_target.int(), index_refer.int(),
                                            grad_output)
        return grad_weight, grad_value, None, None


attention_relation_step = AttentionRelationStep.apply
attention_fusion_step = AttentionFusionStep.apply
