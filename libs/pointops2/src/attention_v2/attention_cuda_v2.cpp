#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_cuda_kernel_v2.h"

void attention_step1_forward_cuda_v2(int N, int M, int h, int C, const unsigned int n_max, at::Tensor q_tensor, at::Tensor k_tensor, 
    at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor attn_tensor)
{
    const float *q = q_tensor.data_ptr<float>();
    const float *k = k_tensor.data_ptr<float>();
    const int *index0_offsets = index0_tensor_offsets.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    float *attn = attn_tensor.data_ptr<float>();
    attention_step1_forward_cuda_launcher_v2(N, M, h, C, n_max, q, k, index0_offsets, index1, attn);
}

void attention_step1_backward_cuda_v2(int N, int M, int h, int C, const unsigned int n_max, at::Tensor grad_out_tensor, 
    at::Tensor index0_tensor_offsets, at::Tensor index1_tensor, at::Tensor q_tensor, at::Tensor k_tensor, 
    at::Tensor grad_q_tensor, at::Tensor grad_k_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0_offsets = index0_tensor_offsets.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *q = q_tensor.data_ptr<float>();
    const float *k = k_tensor.data_ptr<float>();
    float *grad_q = grad_q_tensor.data_ptr<float>();
    float *grad_k = grad_k_tensor.data_ptr<float>();
    attention_step1_backward_cuda_launcher_v2(N, M, h, C, n_max, grad_out, index0_offsets, index1, q, k, grad_q, grad_k);
}

void attention_step2_forward_cuda_v2(int N, int M, int h, int C, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor output_tensor)
{
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_step2_forward_cuda_launcher_v2(N, M, h, C, attn, v, index0, index1, output);
}


void attention_step2_backward_cuda_v2(int N, int M, int h, int C, at::Tensor grad_out_tensor, 
    at::Tensor index0_tensor, at::Tensor index1_tensor, at::Tensor attn_tensor, at::Tensor v_tensor, 
    at::Tensor grad_attn_tensor, at::Tensor grad_v_tensor)
{
    const float *grad_out = grad_out_tensor.data_ptr<float>();
    const int *index0 = index0_tensor.data_ptr<int>();
    const int *index1 = index1_tensor.data_ptr<int>();
    const float *attn = attn_tensor.data_ptr<float>();
    const float *v = v_tensor.data_ptr<float>();
    float *grad_attn = grad_attn_tensor.data_ptr<float>();
    float *grad_v = grad_v_tensor.data_ptr<float>();
    attention_step2_backward_cuda_launcher_v2(N, M, h, C, grad_out, index0, index1, attn, v, grad_attn, grad_v);
}
