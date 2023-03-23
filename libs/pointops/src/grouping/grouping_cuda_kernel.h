#ifndef _GROUPING_CUDA_KERNEL
#define _GROUPING_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void grouping_forward_cuda(int m, int nsample, int c, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor output_tensor);
void grouping_backward_cuda(int m, int nsample, int c, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor grad_input_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void grouping_forward_cuda_launcher(int m, int nsample, int c, const float *input, const int *idx, float *output);
void grouping_backward_cuda_launcher(int m, int nsample, int c, const float *grad_output, const int *idx, float *grad_input);

#ifdef __cplusplus
}
#endif
#endif
