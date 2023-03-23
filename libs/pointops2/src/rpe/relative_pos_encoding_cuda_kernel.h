#ifndef _RPE_CUDA_KERNEL
#define _RPE_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void dot_prod_with_idx_forward_cuda(int N, int M, int h, int hdim, at::Tensor q_tensor, at::Tensor index_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void dot_prod_with_idx_backward_cuda(int N, int M, int h, int hdim, at::Tensor grad_out_tensor, at::Tensor q_tensor, at::Tensor index_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_q_tensor, at::Tensor grad_table_tensor);

void attention_step2_with_rel_pos_value_forward_cuda(int N, int M, int h, int hdim, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor output_tensor);
void attention_step2_with_rel_pos_value_backward_cuda(int N, int M, int h, int hdim, at::Tensor grad_out_tensor, at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, at::Tensor table_tensor, at::Tensor rel_idx_tensor, at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor, at::Tensor grad_table_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void dot_prod_with_idx_forward_cuda_launcher(int N, int M, int h, int hdim, const float *q, const int *index, const float *table, const int *rel_idx, float *output);
void dot_prod_with_idx_backward_cuda_launcher(int N, int M, int h, int hdim, const float *grad_out, const float *q, const int *index, const float *table, const int *rel_idx, float *grad_q, float *grad_table);

void attention_step2_with_rel_pos_value_forward_cuda_launcher(int N, int M, int h, int hdim, const float *attn, const float *v, const int *index0, const int *index1, const float *table, const int *rel_idx, float *output);
void attention_step2_with_rel_pos_value_backward_cuda_launcher(int N, int M, int h, int hdim, const float *grad_out, const int *index0, const int *index1, const float *attn, const float *v, const float *table, const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table);

#ifdef __cplusplus
}
#endif
#endif
