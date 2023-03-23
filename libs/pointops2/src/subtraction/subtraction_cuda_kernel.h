#ifndef _SUBTRACTION_CUDA_KERNEL
#define _SUBTRACTION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void subtraction_forward_cuda(int n, int nsample, int c, at::Tensor input1_tensor, at::Tensor input2_tensor, at::Tensor idx_tensor, at::Tensor output_tensor);
void subtraction_backward_cuda(int n, int nsample, int c, at::Tensor idx_tensor, at::Tensor grad_output_tensor, at::Tensor grad_input1_tensor, at::Tensor grad_input2_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void subtraction_forward_cuda_launcher(int n, int nsample, int c, const float *input1, const float *input2, const int *idx, float *output);
void subtraction_backward_cuda_launcher(int n, int nsample, int c, const int *idx, const float *grad_output, float *grad_input1, float *grad_input2);

#ifdef __cplusplus
}
#endif
#endif
