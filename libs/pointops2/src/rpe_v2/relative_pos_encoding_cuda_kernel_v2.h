#ifndef _RPE_V2_CUDA_KERNEL
#define _RPE_V2_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void dot_prod_with_idx_forward_cuda_v2(int N, int M, int h, int hdim, int n_max, int T, at::Tensor q_tensor, at::Tensor index_q_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor rel_idx_offsets_tensor, at::Tensor sort_indices_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_backward_cuda_v2(int N, int M, int h, int hdim, int n_max, int T, at::Tensor grad_out_tensor, at::Tensor q_tensor, at::Tensor index_q_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor rel_idx_offsets_tensor, at::Tensor sort_indices_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor);

void dot_prod_with_idx_forward_cuda_v3(int N, int M, int h, int hdim, int n_max, at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_backward_cuda_v3(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor q_tensor, at::Tensor index_q_offsets_tensor, at::Tensor k_tensor, at::Tensor index_k_tensor, at::Tensor table_q_tensor, at::Tensor table_k_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_q_tensor, at::Tensor grad_k_tensor, at::Tensor grad_table_q_tensor, at::Tensor grad_table_k_tensor);

void attention_step2_with_rel_pos_value_forward_cuda_v2(int N, int M, int h, int hdim, int n_max, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void attention_step2_with_rel_pos_value_backward_cuda_v2(int N, int M, int h, int hdim, int n_max, at::Tensor grad_out_tensor, at::Tensor index0_offsets_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void dot_prod_with_idx_forward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, int T, const float *q, const int *index_q, const float *k, const int *index_k, const float *table_q, const float *table_k, const int *rel_idx, const int *rel_idx_offsets, const int *sort_indices, float *output);
void dot_prod_with_idx_backward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, int T, const float *grad_out, const float *q, const int *index_q, const float *k, const int *index_k, const float *table_q, const float *table_k, const int *rel_idx, const int *rel_idx_offsets, const int *sort_indices, float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k);

void dot_prod_with_idx_forward_cuda_launcher_v3(int N, int M, int h, int hdim, int n_max, const float *q, const int *index_q_offsets, const float *k, const int *index_k, const float *table_q, const float *table_k, const int *rel_idx, float *output);
void dot_prod_with_idx_backward_cuda_launcher_v3(int N, int M, int h, int hdim, int n_max, const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k, const float *table_q, const float *table_k, const int *rel_idx, float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k);

void attention_step2_with_rel_pos_value_forward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, const float *attn, const float *v, const int *index0_offsets, const int *index1, const float *table, const int *rel_idx, float *output);
void attention_step2_with_rel_pos_value_backward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, const float *grad_out, const int *index0_offsets, const int *index1, const float *attn, const float *v, const float *table, const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table);

#ifdef __cplusplus
}
#endif
#endif
