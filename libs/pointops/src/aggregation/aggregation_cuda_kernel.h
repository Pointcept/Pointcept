#ifndef _AGGREGATION_CUDA_KERNEL
#define _AGGREGATION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void aggregation_forward_cuda(int n, int nsample, int c, int w_c, at::Tensor input_tensor, at::Tensor position_tensor, at::Tensor weight_tensor, at::Tensor idx_tensor, at::Tensor output_tensor);
void aggregation_backward_cuda(int n, int nsample, int c, int w_c, at::Tensor input_tensor, at::Tensor position_tensor, at::Tensor weight_tensor, at::Tensor idx_tensor, at::Tensor grad_output_tensor, at::Tensor grad_input_tensor, at::Tensor grad_position_tensor, at::Tensor grad_weight_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void aggregation_forward_cuda_launcher(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, float *output);
void aggregation_backward_cuda_launcher(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, const float *grad_output, float *grad_input, float *grad_position, float *grad_weight);

#ifdef __cplusplus
}
#endif
#endif
