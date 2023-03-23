#include "../cuda_utils.h"
#include "subtraction_cuda_kernel.h"


__global__ void subtraction_forward_cuda_kernel(int n, int nsample, int c, const float *input1, const float *input2, const int *idx, float *output) {
    // input: input1: (n, c), input2: (n, c), idx: (n, nsample), output: (n, nsample, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * nsample * c) return;
    const int c_idx = index % c;
    const int nsample_idx = (index / c) % nsample;
    const int n_idx = index / nsample / c;
    const int idx_idx = n_idx * nsample + nsample_idx;
    const int input1_idx = n_idx * c + c_idx;
    const int input2_idx = idx[idx_idx] * c + c_idx;
    output[index] = input1[input1_idx] - input2[input2_idx];
}

__global__ void subtraction_backward_cuda_kernel(int n, int nsample, int c, const int *idx, const float *grad_output, float *grad_input1, float *grad_input2) {
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * nsample * c) return;
    const int c_idx = index % c;
    const int nsample_idx = (index / c) % nsample;
    const int n_idx = index / nsample / c;
    const int idx_idx = n_idx * nsample + nsample_idx;
    const int input1_idx = n_idx * c + c_idx;
    const int input2_idx = idx[idx_idx] * c + c_idx;
    atomicAdd(grad_input1 + input1_idx, grad_output[index]);
    atomicAdd(grad_input2 + input2_idx, -grad_output[index]);
}

void subtraction_forward_cuda_launcher(int n, int nsample, int c, const float *input1, const float *input2, const int *idx, float *output) {
    // input: input1: (n, c), input2: (n, c), idx: (n, nsample), output: (n, nsample, c)
    dim3 blocks(DIVUP(n * nsample * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    subtraction_forward_cuda_kernel<<<blocks, threads, 0>>>(n, nsample, c, input1, input2, idx, output);
}

void subtraction_backward_cuda_launcher(int n, int nsample, int c, const int *idx, const float *grad_output, float *grad_input1, float *grad_input2) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    dim3 blocks(DIVUP(n * nsample * c, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    subtraction_backward_cuda_kernel<<<blocks, threads, 0>>>(n, nsample, c, idx, grad_output, grad_input1, grad_input2);
}
