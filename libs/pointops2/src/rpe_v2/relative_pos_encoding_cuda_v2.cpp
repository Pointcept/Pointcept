#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "relative_pos_encoding_cuda_kernel_v2.h"

void dot_prod_with_idx_forward_cuda_v2(int N, int M, int h, int hdim, int n_max, int T, at::Tensor q_tensor, 
    at::Tensor index_q_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, 
    at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor rel_idx_offsets_tensor, at::Tensor sort_indices_tensor, at::Tensor output_tensor)
{
    const float *q = q_tensor.data_ptr<float>();
    const int *index_q = index_q_tensor.data_ptr<int>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const float *table_q = table_q_tensor.data_ptr<float>();
    const float *table_k = table_k_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    const int *rel_idx_offsets = rel_idx_offsets_tensor.data_ptr<int>();
    const int *sort_indices = sort_indices_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    dot_prod_with_idx_forward_cuda_launcher_v2(N, M, h, hdim, n_max, T, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, output);
}

void dot_prod_with_idx_backward_cuda_v2(int N, int M, int h, int hdim, int n_max, int T, at::Tensor grad_out_tensor, 
    at::Tensor q_tensor, at::Tensor index_q_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, 
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor rel_idx_offsets_tensor, 
    at::Tensor sort_indices_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *q = q_tensor.data_ptr<float>();
    const int *index_q = index_q_tensor.data_ptr<int>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const float *table_q = table_q_tensor.data_ptr<float>();
    const float *table_k = table_k_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    const int *rel_idx_offsets = rel_idx_offsets_tensor.data_ptr<int>();
    const int *sort_indices = sort_indices_tensor.data_ptr<int>();
    float *grad_q = grad_q_tensor.data_ptr<float>();
    float *grad_k = grad_k_tensor.data_ptr<float>();
    float *grad_table_q = grad_table_q_tensor.data_ptr<float>();
    float *grad_table_k = grad_table_k_tensor.data_ptr<float>();
    dot_prod_with_idx_backward_cuda_launcher_v2(N, M, h, hdim, n_max, T, grad_out, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, grad_q, grad_k, grad_table_q, grad_table_k);
}


void dot_prod_with_idx_forward_cuda_v3(int N, int M, int h, int hdim, int n_max, at::Tensor q_tensor, 
    at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, 
    at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const float *q = q_tensor.data_ptr<float>();
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const float *table_q = table_q_tensor.data_ptr<float>();
    const float *table_k = table_k_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    dot_prod_with_idx_forward_cuda_launcher_v3(N, M, h, hdim, n_max, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
}

void dot_prod_with_idx_backward_cuda_v3(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, 
    at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, 
    at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_q_tensor, 
    at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const float *q = q_tensor.data_ptr<float>();
    const int *index_q_offsets = index_q_offsets_tensor.data_ptr<int>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index_k = index_k_tensor.data_ptr<int>();
    const float *table_q = table_q_tensor.data_ptr<float>();
    const float *table_k = table_k_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *grad_q = grad_q_tensor.data_ptr<float>();
    float *grad_k = grad_k_tensor.data_ptr<float>();
    float *grad_table_q = grad_table_q_tensor.data_ptr<float>();
    float *grad_table_k = grad_table_k_tensor.data_ptr<float>();
    dot_prod_with_idx_backward_cuda_launcher_v3(N, M, h, hdim, n_max, grad_out, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
}


void attention_step2_with_rel_pos_value_forward_cuda_v2(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor)
{
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *table = table_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_step2_with_rel_pos_value_forward_cuda_launcher_v2(N, M, h, hdim, n_max, attn, v, index0_offsets, index1, table, rel_idx, output);
}

void attention_step2_with_rel_pos_value_backward_cuda_v2(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, 
    at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor,
    at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0_offsets = index0_offsets_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const float *table = table_tensor.data_ptr<float>();
    const int *rel_idx = rel_idx_tensor.data_ptr<int>();
    float *grad_attn = grad_attn_tensor.data_ptr<float>();
    float *grad_v = grad_v_tensor.data_ptr<float>();
    float *grad_table = grad_table_tensor.data_ptr<float>();
    attention_step2_with_rel_pos_value_backward_cuda_launcher_v2(N, M, h, hdim, n_max, grad_out, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
}
