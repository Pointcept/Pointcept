/* written by Xin Lai. Email: xinlai@cse.cuhk.edu.hk */

#include "../cuda_utils.h"
#include "relative_pos_encoding_cuda_kernel.h"


__global__ void dot_prod_with_idx_forward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, const float *q, const int *index,
    const float *table, const int *rel_idx, float *output) {
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3), output: (M, h)

    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= M*3 || h_idx >= h || c_idx >= hdim) return;

    int dim = thread_idx % 3;
    int m_idx = thread_idx / 3;

    int q_idx = index[m_idx];
    int rel_idx_dim = rel_idx[thread_idx];
    float rel_table_val = table[rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim];
    float val = q[q_idx*h*hdim+h_idx*hdim+c_idx] * rel_table_val;
    atomicAdd(output+m_idx*h+h_idx, val);
}

__global__ void dot_prod_with_idx_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, const float *grad_out, const float *q, const int *index, 
    const float *table, const int *rel_idx, float *grad_q, float *grad_table) {
    
    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= M*3 || h_idx >= h || c_idx >= hdim) return;

    int dim = thread_idx % 3;
    int m_idx = thread_idx / 3;

    int q_idx = index[m_idx];
    int rel_idx_dim = rel_idx[thread_idx];
    int grad_out_idx = m_idx*h+h_idx;
    float grad_out_value = grad_out[grad_out_idx];

    float rel_table_val = table[rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim];
    atomicAdd(grad_q+q_idx*h*hdim+h_idx*hdim+c_idx, grad_out_value * rel_table_val);

    float q_value = q[q_idx*h*hdim+h_idx*hdim+c_idx];
    atomicAdd(grad_table+rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim, grad_out_value * q_value);
}

void dot_prod_with_idx_forward_cuda_launcher(int N, int M, int h, int hdim, const float *q, const int *index,
    const float *table, const int *rel_idx, float *output) {
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    dot_prod_with_idx_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, q, index, table, rel_idx, output);
}

void dot_prod_with_idx_backward_cuda_launcher(int N, int M, int h, int hdim, const float *grad_out, 
    const float *q, const int *index, const float *table, const int *rel_idx, float *grad_q, float *grad_table) {  
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    dot_prod_with_idx_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, grad_out, q, index, table, rel_idx, grad_q, grad_table);
}

__global__ void attention_step2_with_rel_pos_value_forward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, const float *attn, const float *v,
    const int *index0, const int *index1, const float *table, const int *rel_idx, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)

    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= M*3 || h_idx >= h || c_idx >= hdim) return;

    int dim = thread_idx % 3;
    int m_idx = thread_idx / 3;

    int idx1 = index1[m_idx];

    int rel_idx_dim = rel_idx[thread_idx];
    float table_val = table[rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim];

    float val = attn[m_idx*h+h_idx] * (v[idx1*h*hdim+h_idx*hdim+c_idx] / 3.0 + table_val);

    int idx0 = index0[m_idx];
    atomicAdd(output+idx0*h*hdim+h_idx*hdim+c_idx, val);
}


__global__ void attention_step2_with_rel_pos_value_backward_cuda_kernel( // M, h, hdim
    int N, int M, int h, int hdim, const float *grad_out, const int *index0, const int *index1, const float *attn, const float *v, const float *table,
    const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)

    int c_idx = blockIdx.z;
    int h_idx = blockIdx.y;
    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_idx >= M*3 || h_idx >= h || c_idx >= hdim) return;

    int dim = thread_idx % 3;
    int m_idx = thread_idx / 3;

    int idx0 = index0[m_idx];
    int idx1 = index1[m_idx];
    int grad_out_idx = idx0*h*hdim+h_idx*hdim+c_idx;

    int rel_idx_dim = rel_idx[thread_idx];
    float table_val = table[rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim];
    float grad_out_value = grad_out[grad_out_idx];

    atomicAdd(grad_attn+m_idx*h+h_idx, grad_out_value * (v[idx1*h*hdim+h_idx*hdim+c_idx]/3 + table_val));
    atomicAdd(grad_v+idx1*h*hdim+h_idx*hdim+c_idx, grad_out_value * attn[m_idx*h+h_idx]/3);
    atomicAdd(grad_table+rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim, grad_out_value * attn[m_idx*h+h_idx]);
}

void attention_step2_with_rel_pos_value_forward_cuda_launcher(int N, int M, int h, int hdim, const float *attn, const float *v, const int *index0,
    const int *index1, const float *table, const int *rel_idx, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_with_rel_pos_value_forward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, attn, v, index0, index1, table, rel_idx, output);
}

void attention_step2_with_rel_pos_value_backward_cuda_launcher(int N, int M, int h, int hdim, const float *grad_out, const int *index0, 
    const int *index1, const float *attn, const float *v, const float *table, const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    dim3 threads(THREADS_PER_BLOCK);
    attention_step2_with_rel_pos_value_backward_cuda_kernel<<<blocks, threads, 0>>>(N, M, h, hdim, grad_out, index0, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
}
