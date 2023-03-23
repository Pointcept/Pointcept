#include "../cuda_utils.h"
#include "aggregation_cuda_kernel.h"


__global__ void aggregation_forward_cuda_kernel(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, float *output) {
    // input: input: (n, c), position: (n, nsample, c), weight: (n, nsample, w_c), idx: (n, nsample), output: (n, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    const int c_idx = index % c;
    const int n_idx = index / c;
    const int w_c_idx = c_idx % w_c;
    for (int nsample_idx = 0; nsample_idx < nsample; nsample_idx++)
    {   
        int idx_idx = n_idx * nsample + nsample_idx;
        int input_idx = idx[idx_idx] * c + c_idx;
        int position_idx = n_idx * nsample * c + nsample_idx * c + c_idx;
        int weight_idx = n_idx * nsample * w_c + nsample_idx * w_c + w_c_idx;
        output[index] += (input[input_idx] + position[position_idx]) * weight[weight_idx];
    }
}

__global__ void aggregation_backward_cuda_kernel(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, const float *grad_output, float *grad_input, float *grad_position, float *grad_weight) {
    // input: grad_output: (n, c), output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight: (n, nsample, w_c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * c) return;
    const int c_idx = index % c;
    const int n_idx = index / c;
    const int w_c_idx = c_idx % w_c;
    for (int nsample_idx = 0; nsample_idx < nsample; nsample_idx++)
    {   
        int idx_idx = n_idx * nsample + nsample_idx;
        int input_idx = idx[idx_idx] * c + c_idx;
        int position_idx = n_idx * nsample * c + nsample_idx * c + c_idx;
        int weight_idx = n_idx * nsample * w_c + nsample_idx * w_c + w_c_idx;
        atomicAdd(grad_input + input_idx, grad_output[index] * weight[weight_idx]);
        grad_position[position_idx] = grad_output[index] * weight[weight_idx];
        atomicAdd(grad_weight + weight_idx, grad_output[index] * (input[input_idx] + position[position_idx]));
    }
}

void aggregation_forward_cuda_launcher(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, float *output) {
    // input: input: (n, c), position: (n, nsample, c), weight: (n, nsample, w_c), idx: (n, nsample), output: (n, c)
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    aggregation_forward_cuda_kernel<<<blocks, threads, 0>>>(n, nsample, c, w_c, input, position, weight, idx, output);
}

void aggregation_backward_cuda_launcher(int n, int nsample, int c, int w_c, const float *input, const float *position, const float *weight, const int *idx, const float *grad_output, float *grad_input, float *grad_position, float *grad_weight) {  
    // input: grad_output: (n, c), output: grad_input: (n, c), grad_position: (n, nsample, c), grad_weight: (n, nsample, w_c)
    dim3 blocks(DIVUP(n * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    aggregation_backward_cuda_kernel<<<blocks, threads, 0>>>(n, nsample, c, w_c, input, position, weight, idx, grad_output, grad_input, grad_position, grad_weight);
}
