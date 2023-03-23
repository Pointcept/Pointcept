#ifndef _INTERPOLATION_CUDA_KERNEL
#define _INTERPOLATION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void interpolation_forward_cuda(int n, int c, int k, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor output_tensor);
void interpolation_backward_cuda(int n, int c, int k, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_input_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void interpolation_forward_cuda_launcher(int n, int c, int k, const float *input, const int *idx, const float *weight, float *output);
void interpolation_backward_cuda_launcher(int n, int c, int k, const float *grad_output, const int *idx, const float *weight, float *grad_input);

#ifdef __cplusplus
}
#endif
#endif
