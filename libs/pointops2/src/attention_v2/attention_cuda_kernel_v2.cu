/* written by Xin Lai. Email: xinlai@cse.cuhk.edu.hk */

#include "../cuda_utils.h"
#include "attention_cuda_kernel_v2.h"


template <unsigned int d>
__global__ void attention_step1_forward_cuda_kernel_v2( // M, h, C//h
    int N, int M, int h, const float *q, const float *k,
    const int *index0_offsets, const int *index1, float *attn) {

    int h_idx = blockIdx.y;
    int q_idx = blockIdx.x;
    int n_idx = threadIdx.x;
    int C = h * d;
    // if (m_idx >= M || h_idx >= h || c_idx >= C / h) return;

    __shared__ float query_vec[d];
    __shared__ int start, end;

    // if(n_idx == 0){
    //     printf("blockDim.x: %d\n", blockDim.x);
    // }

    if (n_idx == 0){
        start = index0_offsets[q_idx];
        end = index0_offsets[q_idx+1];
        // printf("start: %d, end: %d, blockDim.x: %d\n", start, end, blockDim.x);
    }
    for(int i = n_idx; i < d; i += blockDim.x)
        query_vec[i] = q[q_idx*C + h_idx*d + i];
    
    __syncthreads();

    int m_idx = start + n_idx;
    if(m_idx >= end)
        return;

    float sum = 0;
    for(int i = 0; i < d; i++){
        int k_idx = index1[m_idx];
        float key = k[k_idx * C + h_idx * d + i];
        sum += query_vec[i] * key;
    }
    attn[m_idx*h + h_idx] = sum;
    // int idx0 = index0[m_idx];
    // int idx1 = index1[m_idx];
    // float val = q[idx0*C+h_idx*C/h+c_idx] * k[idx1*C+h_idx*C/h+c_idx];
    // atomicAdd(attn+m_idx*h+h_idx, val);
}

template <unsigned int d>
__global__ void attention_step1_backward_cuda_kernel_v2( // M, h, C//h
    int N, int M, int h, const float *grad_out, const int *index0_offsets, const int *index1, const float *q, const float *k,
    float *grad_q, float *grad_k) {
    
    int h_idx = blockIdx.y;
    int q_idx = blockIdx.x;
    int n_idx = threadIdx.x;
    int C = d * h;
    
    __shared__ float query_vec[d];
    __shared__ int start, end;

    if (n_idx == 0){
        start = index0_offsets[q_idx];
        end = index0_offsets[q_idx+1];
    }
    for(int i = n_idx; i < d; i += blockDim.x)
        query_vec[i] = q[q_idx*C + h_idx*d + i];
    
    __shared__ float gradient_new[d];
    for(int i = n_idx; i < d; i += blockDim.x)
        gradient_new[i] = 0;

    __syncthreads();

    int m_idx = start + n_idx;
    if(m_idx < end){
        float gradient = grad_out[m_idx*h + h_idx];
        for(int i = 0; i < d; i++){
            int k_idx = index1[m_idx];
            atomicAdd(&gradient_new[i], gradient * k[k_idx*C + h_idx*d + i]);
            atomicAdd(grad_k + k_idx*C + h_idx*d + i, gradient * query_vec[i]);
        }
    }
    __syncthreads();

    for(int i = n_idx; i < d; i += blockDim.x)
        grad_q[q_idx*C + h_idx*d + i] = gradient_new[i];
}

void attention_step1_forward_cuda_launcher_v2(int N, int M, int h, int C, const unsigned int n_max, 
    const float *q, const float *k, const int *index0_offsets, const int *index1, float *attn) {
    // input: attn: (M, h), v: (N, h, C/h), index0: (M, ), index1: (M, )
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(N, h);
	unsigned int n_threads = opt_n_threads(n_max);

    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    // n_threads = n_threads > 1024 ? 512 : n_threads;

    // printf("n_max: %d, n_threads: %d\n", n_max, n_threads);

    // dim3 threads(THREADS_PER_BLOCK);
    // attention_step1_forward_cuda_kernel_v2<<<blocks, threads, 0>>>(N, M, h, C, q, k, index0, index1, attn);
    
	switch (C / h) {
        case 16:
            attention_step1_forward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, q, k, index0_offsets, index1, attn);
            break;
        case 32:
            attention_step1_forward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, q, k, index0_offsets, index1, attn);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}

void attention_step1_backward_cuda_launcher_v2(int N, int M, int h, int C, const unsigned int n_max, 
    const float *grad_out, const int *index0_offsets, const int *index1, const float *q, const float *k, float *grad_q, float *grad_k) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    // dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    // dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    // attention_step1_backward_cuda_kernel_v2<<<blocks, n_threads, 0>>>(N, M, h, C/h, grad_out, index0_offsets, index1, q, k, grad_q, grad_k);

    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    // n_threads = n_threads > 1024 ? 512 : n_threads;

    // printf("n_max: %d, n_threads: %d\n", n_max, n_threads);

	switch (C / h) {
        case 16:
            attention_step1_backward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, grad_out, index0_offsets, index1, q, k, grad_q, grad_k);
        break;
        case 32:
            attention_step1_backward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, grad_out, index0_offsets, index1, q, k, grad_q, grad_k);
            break;
        default:
            throw "d != 16 and d != 32";
    }

}

__global__ void attention_step2_forward_cuda_kernel_v2( // M, h, C//h
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

__global__ void attention_step2_backward_cuda_kernel_v2( // M, h, C//h
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

void attention_step2_forward_cuda_launcher_v2(int N, int M, int h, int C, const float *attn, const float *v,
    const int *index0, const int *index1, float *output) {
    // input: attn: (M, h), v: (N, h, C/h), index0: (M, ), index1: (M, )
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_forward_cuda_kernel_v2<<<blocks, threads, 0>>>(N, M, h, C, attn, v, index0, index1, output);
}

void attention_step2_backward_cuda_launcher_v2(int N, int M, int h, int C, const float *grad_out, const int *index0, const int *index1, 
    const float *attn, const float *v, float *grad_attn, float *grad_v) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(C/h, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M, THREADS_PER_BLOCK), h, C/h);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_backward_cuda_kernel_v2<<<blocks, threads, 0>>>(N, M, h, C, grad_out, index0, index1, attn, v, grad_attn, grad_v);
}
