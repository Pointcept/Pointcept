/* written by Xin Lai. Email: xinlai@cse.cuhk.edu.hk */

#include "../cuda_utils.h"
#include "attention_cuda_kernel.h"


__global__ void attention_step1_forward_cuda_kernel( // M, h, C//h
    int N, int M, int h, int C, const float *q, const float *k,
    const int *index0, const int *index1, float *attn) {

    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx >= M || h_idx >= h || c_idx >= C / h) return;

    int idx0 = index0[m_idx];
    int idx1 = index1[m_idx];
    float val = q[idx0*C+h_idx*C/h+c_idx] * k[idx1*C+h_idx*C/h+c_idx];
    atomicAdd(attn+m_idx*h+h_idx, val);
}

__global__ void attention_step1_backward_cuda_kernel( // M, h, C//h
    int N, int M, int h, int C, const float *grad_out, const int *index0, const int *index1, const float *q, const float *k,
    float *grad_q, float *grad_k) {
    
    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx >= M || h_idx >= h || c_idx >= C / h) return;

    int idx0 = index0[m_idx];
    int idx1 = index1[m_idx];
    int grad_out_idx = m_idx*h+h_idx;
    int q_idx = idx0*C+h_idx*C/h+c_idx;
    int k_idx = idx1*C+h_idx*C/h+c_idx;
    atomicAdd(grad_q+q_idx, grad_out[grad_out_idx] * k[k_idx]);
    atomicAdd(grad_k+k_idx, grad_out[grad_out_idx] * q[q_idx]);
}

void attention_step1_forward_cuda_launcher(int N, int M, int h, int C, const float *q, const float *k,
    const int *index0, const int *index1, float *attn) {
    // input: attn: (M, h), v: (N, h, C/h), index0: (M, ), index1: (M, )
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step1_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, C, q, k, index0, index1, attn);
}

void attention_step1_backward_cuda_launcher(int N, int M, int h, int C, const float *grad_out, const int *index0, const int *index1, 
    const float *q, const float *k, float *grad_q, float *grad_k) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step1_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, C, grad_out, index0, index1, q, k, grad_q, grad_k);
}

__global__ void attention_step2_forward_cuda_kernel( // M, h, C//h
    int N, int M, int h, int C, const float *attn, const float *v,
    const int *index0, const int *index1, float *output) {

    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx >= M || h_idx >= h || c_idx >= C / h) return;

    int idx1 = index1[m_idx];
    float val = attn[m_idx*h+h_idx] * v[idx1*C+h_idx*C/h+c_idx];
    int idx0 = index0[m_idx];
    atomicAdd(output+idx0*C+h_idx*C/h+c_idx, val);
}

__global__ void attention_step2_backward_cuda_kernel( // M, h, C//h
    int N, int M, int h, int C, const float *grad_out, const int *index0, const int *index1, const float *attn, const float *v,
    float *grad_attn, float *grad_v) {
    
    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int m_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (m_idx >= M || h_idx >= h || c_idx >= C / h) return;

    int idx0 = index0[m_idx];
    int idx1 = index1[m_idx];
    int grad_out_idx = idx0*C+h_idx*C/h+c_idx;
    atomicAdd(grad_attn+m_idx*h+h_idx, grad_out[grad_out_idx] * v[idx1*C+h_idx*C/h+c_idx]);
    atomicAdd(grad_v+idx1*C+h_idx*C/h+c_idx, grad_out[grad_out_idx] * attn[m_idx*h+h_idx]);
}

void attention_step2_forward_cuda_launcher(int N, int M, int h, int C, const float *attn, const float *v,
    const int *index0, const int *index1, float *output) {
    // input: attn: (M, h), v: (N, h, C/h), index0: (M, ), index1: (M, )
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, C, attn, v, index0, index1, output);
}

void attention_step2_backward_cuda_launcher(int N, int M, int h, int C, const float *grad_out, const int *index0, const int *index1, 
    const float *attn, const float *v, float *grad_attn, float *grad_v) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, C, grad_out, index0, index1, attn, v, grad_attn, grad_v);
}
