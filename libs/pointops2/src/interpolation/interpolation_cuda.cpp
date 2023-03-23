#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "interpolation_cuda_kernel.h"


void interpolation_forward_cuda(int n, int c, int k, at::Tensor input_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor output_tensor)
{
    const float *input = input_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *output = output_tensor.data_ptr<float>();
    interpolation_forward_cuda_launcher(n, c, k, input, idx, weight, output);
}

void interpolation_backward_cuda(int n, int c, int k, at::Tensor grad_output_tensor, at::Tensor idx_tensor, at::Tensor weight_tensor, at::Tensor grad_input_tensor)
{
    const float *grad_output = grad_output_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_input = grad_input_tensor.data_ptr<float>();
    interpolation_backward_cuda_launcher(n, c, k, grad_output, idx, weight, grad_input);
}
