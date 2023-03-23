#include "../cuda_utils.h"
#include "interpolation_cuda_kernel.h"


__global__ void interpolation_forward_cuda_kernel(int n, int c, int k, const float *input, const int *idx, const float *weight, float *output)
{
    // input: input: (m, c), idx: (n, k), weight: (n, k), output: output (n, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    int c_idx = index % c;
    int n_idx = index / c;
    for (int i = 0; i < k; i++)
    {
        int idx_idx = n_idx * k + i;
        int input_idx = idx[idx_idx] * c + c_idx;
        output[index] += input[input_idx] * weight[idx_idx];
    }
}

__global__ void interpolation_backward_cuda_kernel(int n, int c, int k, const float *grad_output, const int *idx, const float *weight, float *grad_input)
{
    // input: grad_output: (n, c), idx: (n, k), weight: (n, k), output: grad_input (m, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    int c_idx = index % c;
    int n_idx = index / c;
    for (int i = 0; i < k; i++)
    {
        int idx_idx = n_idx * k + i;
        int input_idx = idx[idx_idx] * c + c_idx;
        atomicAdd(grad_input + input_idx, grad_output[index] * weight[idx_idx]);
    }
}

void interpolation_forward_cuda_launcher(int n, int c, int k, const float *input, const int *idx, const float *weight, float *output) {
    // input: input: (m, c), idx: (n, k), weight: (n, k), output: output (n, c)
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    interpolation_forward_cuda_kernel<<<blocks, threads, 0>>>(n, c, k, input, idx, weight, output);
}

void interpolation_backward_cuda_launcher(int n, int c, int k, const float *grad_output, const int *idx, const float *weight, float *grad_input) {
    // input: grad_output: (n, c), idx: (n, k), weight: (n, k), output: grad_input (m, c)
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    interpolation_backward_cuda_kernel<<<blocks, threads, 0>>>(n, c, k, grad_output, idx, weight, grad_input);
}
