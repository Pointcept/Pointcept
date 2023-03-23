#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "aggregation_cuda_kernel.h"


void aggregation_forward_cuda(int n, int nsample, int c, int w_c, at::Tensor input_tensor, at::Tensor position_tensor, at::Tensor weight_tensor, at::Tensor idx_tensor, at::Tensor output_tensor)
{
    const float *input = input_tensor.data_ptr<float>();
    const float *position = position_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    aggregation_forward_cuda_launcher(n, nsample, c, w_c, input, position, weight, idx, output);
}

void aggregation_backward_cuda(int n, int nsample, int c, int w_c, at::Tensor input_tensor, at::Tensor position_tensor, at::Tensor weight_tensor, at::Tensor idx_tensor, at::Tensor grad_output_tensor, at::Tensor grad_input_tensor, at::Tensor grad_position_tensor, at::Tensor grad_weight_tensor)
{
	const float *input = input_tensor.data_ptr<float>();
    const float *position = position_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    const float *grad_output = grad_output_tensor.data_ptr<float>();
    float *grad_input = grad_input_tensor.data_ptr<float>();
    float *grad_position = grad_position_tensor.data_ptr<float>();
    float *grad_weight = grad_weight_tensor.data_ptr<float>();
    aggregation_backward_cuda_launcher(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight);
}
