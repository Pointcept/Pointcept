#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "relative_pos_encoding_cuda_kernel.h"

void dot_prod_with_idx_forward_cuda(int N, int M, int h, int hdim, at::Tensor q_tensor, at::Tensor index_tensor, 
    at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const float *q = q_tensor.data_ptr<float>();
    const float *table = table_tensor.data_ptr<float>();
    const int *index = index_tensor.data_ptr<int>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    dot_prod_with_idx_forward_cuda_launcher(N, M, h, hdim, q, index, table, rel_idx, output);
}

void dot_prod_with_idx_backward_cuda(int N, int M, int h, int hdim, at::Tensor grad_out_tensor, 
    at::Tensor q_tensor, at::Tensor index_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, 
    at::Tensor grad_q_tensor, at::Tensor grad_table_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *q = q_tensor.data_ptr<float>();
    const int *index = index_tensor.data_ptr<int>();
    const float *table = table_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *grad_q = grad_q_tensor.data_ptr<float>();
    float *grad_table = grad_table_tensor.data_ptr<float>();
    dot_prod_with_idx_backward_cuda_launcher(N, M, h, hdim, grad_out, q, index, table, rel_idx, grad_q, grad_table);
}

void attention_step2_with_rel_pos_value_forward_cuda(int N, int M, int h, int hdim, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *table = table_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_step2_with_rel_pos_value_forward_cuda_launcher(N, M, h, hdim, attn, v, index0, index1, table, rel_idx, output);
}

void attention_step2_with_rel_pos_value_backward_cuda(int N, int M, int h, int hdim, at::Tensor grad_out_tensor, 
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const float *table = table_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *grad_attn = grad_attn_tensor.data_ptr<float>();
    float *grad_v = grad_v_tensor.data_ptr<float>();
    float *grad_table = grad_table_tensor.data_ptr<float>();
    attention_step2_with_rel_pos_value_backward_cuda_launcher(N, M, h, hdim, grad_out, index0, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
}
