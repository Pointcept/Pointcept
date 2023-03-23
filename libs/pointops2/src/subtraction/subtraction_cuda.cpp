#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "subtraction_cuda_kernel.h"


void subtraction_forward_cuda(int n, int nsample, int c, at::Tensor input1_tensor, at::Tensor input2_tensor, at::Tensor idx_tensor, at::Tensor output_tensor)
{
    const float *input1 = input1_tensor.data_ptr<float>();
    const float *input2 = input2_tensor.data_ptr<float>();
    const int *idx = idx_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    subtraction_forward_cuda_launcher(n, nsample, c, input1, input2, idx, output);
}

void subtraction_backward_cuda(int n, int nsample, int c, at::Tensor idx_tensor, at::Tensor grad_output_tensor, at::Tensor grad_input1_tensor, at::Tensor grad_input2_tensor)
{
    const int *idx = idx_tensor.data_ptr<int>();
    const float *grad_output = grad_output_tensor.data_ptr<float>();
    float *grad_input1 = grad_input1_tensor.data_ptr<float>();
    float *grad_input2 = grad_input2_tensor.data_ptr<float>();
    subtraction_backward_cuda_launcher(n, nsample, c, idx, grad_output, grad_input1, grad_input2);
}
