#ifndef _SAMPLING_CUDA_KERNEL
#define _SAMPLING_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void farthest_point_sampling_cuda(int b, int n, at::Tensor xyz_tensor, at::Tensor offset_tensor, at::Tensor new_offset_tensor, at::Tensor tmp_tensor, at::Tensor idx_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void farthest_point_sampling_cuda_launcher(int b, int n, const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx);

#ifdef __cplusplus
}
#endif
#endif
