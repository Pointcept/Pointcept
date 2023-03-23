/* written by Xin Lai. Email: xinlai@cse.cuhk.edu.hk */

#include "../cuda_utils.h"
#include "relative_pos_encoding_cuda_kernel_v2.h"


// N, M, h, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, output

template <unsigned int d>
__global__ void dot_prod_with_idx_forward_cuda_kernel_v2( // M, h, hdim
    int N, int M, int h, const float *q, const int *index_q, const float *k, const int *index_k,
    const float *table_q, const float *table_k, const int *rel_idx, const int *rel_idx_offsets, 
    const int *sort_indices, float *output) {
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3), output: (M, h)

    int h_idx = blockIdx.y;
    int t_idx = blockIdx.x;
    int n_idx = threadIdx.x;
    int C = h*d;

    __shared__ int start, end;
    if(n_idx == 0){
        start = rel_idx_offsets[t_idx];
        end = rel_idx_offsets[t_idx+1];
        // printf("e2: start: %d, end: %d\n", start, end);
    }

    __syncthreads();
    
    int m_idx_prev = start + n_idx;
    // if(m_idx_prev >= end)
    //     return;

    __shared__ int m_idx;
    if(n_idx == 0)
        m_idx = sort_indices[m_idx_prev];

    __syncthreads();
    
    __shared__ int rel_idx_vec[3];
    if(n_idx < 3)
        rel_idx_vec[n_idx] = rel_idx[m_idx*3 + n_idx];
    
    __syncthreads();
    
    __shared__ float table_q_vec[d];
    __shared__ float table_k_vec[d];

    for(int i = n_idx; i < 2*d; i += blockDim.x){
        if (i < d){
            int ind0 = rel_idx_vec[0] * C * 3 + h_idx * d * 3 + i * 3 + 0;
            int ind1 = rel_idx_vec[1] * C * 3 + h_idx * d * 3 + i * 3 + 1;
            int ind2 = rel_idx_vec[2] * C * 3 + h_idx * d * 3 + i * 3 + 2;
            table_q_vec[i] = table_q[ind0] + table_q[ind1] + table_q[ind2];
        } else{
            int ind0 = rel_idx_vec[0] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 0;
            int ind1 = rel_idx_vec[1] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 1;
            int ind2 = rel_idx_vec[2] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 2;
            table_k_vec[i-d] = table_k[ind0] + table_k[ind1] + table_k[ind2];
        }
    }

    __syncthreads();

    for(int i = m_idx_prev; i < end; i += blockDim.x){
        float sum = 0;
        int m_idx_i = sort_indices[i];
        int q_idx = index_q[m_idx_i];
        int k_idx = index_k[m_idx_i];
        for(int j = 0; j < d; j++){
            sum += q[q_idx*C + h_idx*d + j] * table_q_vec[j];
            sum += k[k_idx*C + h_idx*d + j] * table_k_vec[j];
        }
        output[m_idx_i*h + h_idx] = sum;
    }
}

// N, M, h, hdim, grad_out, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, grad_q, grad_k, grad_table_q, grad_table_k

template <unsigned int d>
__global__ void dot_prod_with_idx_backward_cuda_kernel_v2( // M, h, hdim
    int N, int M, int h, const float *grad_out, const float *q, const int *index_q, 
    const float *k, const int *index_k, const float *table_q, const float *table_k, 
    const int *rel_idx, const int *rel_idx_offsets, const int *sort_indices, float *grad_q, 
    float *grad_k, float *grad_table_q, float *grad_table_k) {
    
    int h_idx = blockIdx.y;
    int t_idx = blockIdx.x;
    int n_idx = threadIdx.x;
    int C = h*d;

    __shared__ int start, end;
    if(n_idx == 0){
        start = rel_idx_offsets[t_idx];
        end = rel_idx_offsets[t_idx+1];
    }

    __syncthreads();
    
    int m_idx_prev = start + n_idx;
    // if(m_idx_prev >= end)
    //     return;

    __shared__ int m_idx;
    if(n_idx == 0)
        m_idx = sort_indices[m_idx_prev];

    __syncthreads();
    
    __shared__ int rel_idx_vec[3];
    if(n_idx < 3)
        rel_idx_vec[n_idx] = rel_idx[m_idx*3 + n_idx];
    
    __syncthreads();
    
    __shared__ float table_q_vec[d];
    __shared__ float table_k_vec[d];

    for(int i = n_idx; i < 2*d; i += blockDim.x){
        if (i < d){
            int ind0 = rel_idx_vec[0] * C * 3 + h_idx * d * 3 + i * 3 + 0;
            int ind1 = rel_idx_vec[1] * C * 3 + h_idx * d * 3 + i * 3 + 1;
            int ind2 = rel_idx_vec[2] * C * 3 + h_idx * d * 3 + i * 3 + 2;
            table_q_vec[i] = table_q[ind0] + table_q[ind1] + table_q[ind2];
        } else{
            int ind0 = rel_idx_vec[0] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 0;
            int ind1 = rel_idx_vec[1] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 1;
            int ind2 = rel_idx_vec[2] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 2;
            table_k_vec[i-d] = table_k[ind0] + table_k[ind1] + table_k[ind2];
        }
    }

    __shared__ float gradient_q[d];
    __shared__ float gradient_k[d];
    for(int i = n_idx; i < d; i += blockDim.x){
        gradient_q[i] = 0;
        gradient_k[i] = 0;
    }

    __syncthreads();

    for(int i = m_idx_prev; i < end; i += blockDim.x){
        int m_idx_i = sort_indices[i];
        int q_idx = index_q[m_idx_i];
        int k_idx = index_k[m_idx_i];
        float grad_out_i = grad_out[m_idx_i*h+h_idx];
        for(int j = 0; j < d; j++){
            atomicAdd(&gradient_q[j], q[q_idx*C + h_idx*d + j] * grad_out_i);
            atomicAdd(&gradient_k[j], k[k_idx*C + h_idx*d + j] * grad_out_i);
            atomicAdd(grad_q + q_idx*C + h_idx*d + j, table_q_vec[j] * grad_out_i);
            atomicAdd(grad_k + k_idx*C + h_idx*d + j, table_k_vec[j] * grad_out_i);
        }
    }

    __syncthreads();

    for(int i = n_idx; i < d*2; i += blockDim.x){
        if(i < d){
            atomicAdd(grad_table_q + rel_idx_vec[0] * C * 3 + h_idx * d * 3 + i * 3, gradient_q[i]);
            atomicAdd(grad_table_q + rel_idx_vec[1] * C * 3 + h_idx * d * 3 + i * 3 + 1, gradient_q[i]);
            atomicAdd(grad_table_q + rel_idx_vec[2] * C * 3 + h_idx * d * 3 + i * 3 + 2, gradient_q[i]);
        }else{
            atomicAdd(grad_table_k + rel_idx_vec[0] * C * 3 + h_idx * d * 3 + (i-d) * 3, gradient_k[i-d]);
            atomicAdd(grad_table_k + rel_idx_vec[1] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 1, gradient_k[i-d]);
            atomicAdd(grad_table_k + rel_idx_vec[2] * C * 3 + h_idx * d * 3 + (i-d) * 3 + 2, gradient_k[i-d]);
        }
    }

    // int c_idx = blockIdx.z;
    // int h_idx = blockIdx.y;
    // int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if (thread_idx >= M*3 || h_idx >= h || c_idx >= hdim) return;

    // int dim = thread_idx % 3;
    // int m_idx = thread_idx / 3;

    // int q_idx = index[m_idx];
    // int rel_idx_dim = rel_idx[thread_idx];
    // int grad_out_idx = m_idx*h+h_idx;
    // float grad_out_value = grad_out[grad_out_idx];

    // float rel_table_val = table[rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim];
    // atomicAdd(grad_q+q_idx*h*hdim+h_idx*hdim+c_idx, grad_out_value * rel_table_val);

    // float q_value = q[q_idx*h*hdim+h_idx*hdim+c_idx];
    // atomicAdd(grad_table+rel_idx_dim*h*hdim*3+h_idx*hdim*3+c_idx*3+dim, grad_out_value * q_value);
}

void dot_prod_with_idx_forward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, int T, const float *q,
    const int *index_q, const float *k, const int *index_k, const float *table_q, const float *table_k, 
    const int *rel_idx, const int *rel_idx_offsets, const int *sort_indices, float *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(T, h);
    // dim3 threads(THREADS_PER_BLOCK);
    
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 512 : n_threads;

    // printf("e1: T: %d, h: %d, n_threads: %d\n", T, h, n_threads);

	switch (hdim) {
        case 16:
            dot_prod_with_idx_forward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, output);
            break;
        case 32:
            dot_prod_with_idx_forward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, output);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}

void dot_prod_with_idx_backward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, int T, 
    const float *grad_out, const float *q, const int *index_q, const float *k, const int *index_k, 
    const float *table_q, const float *table_k, const int *rel_idx, const int *rel_idx_offsets, const int *sort_indices, 
    float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k)
{  
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    // dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    // dim3 threads(THREADS_PER_BLOCK);

    dim3 blocks(T, h);
    // dim3 threads(THREADS_PER_BLOCK);
    
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;
    n_threads = n_threads > 1024 ? 512 : n_threads;

	switch (hdim) {
        case 16:
            dot_prod_with_idx_backward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, grad_out, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, grad_q, grad_k, grad_table_q, grad_table_k);
            break;
        case 32:
            dot_prod_with_idx_backward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, grad_out, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, grad_q, grad_k, grad_table_q, grad_table_k);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}



template <unsigned int d>
__global__ void dot_prod_with_idx_forward_cuda_kernel_v3( // M, h, hdim
    int N, int M, int h, const float *q, const int *index_q_offsets, const float *k, const int *index_k,
    const float *table_q, const float *table_k, const int *rel_idx, float *output) {
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3), output: (M, h)
    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int n_idx = threadIdx.x;
    int C = h*d;

    __shared__ float query_vec[d];
    __shared__ int start, end;
    if (n_idx == 0){
        start = index_q_offsets[q_idx];
        end = index_q_offsets[q_idx+1];
    }
    for(int i = n_idx; i < d; i += blockDim.x)
        query_vec[i] = q[q_idx*C + h_idx*d + i];

    __syncthreads();

    int m_idx = start + n_idx;
    if(m_idx >= end)
        return;

    int k_idx = index_k[m_idx];
    int r_idx1 = rel_idx[m_idx*3], r_idx2 = rel_idx[m_idx*3+1], r_idx3 = rel_idx[m_idx*3+2];
    float sum = 0;
    for(int i = 0; i < d; i++){
        float table_q_scalar_i = table_q[r_idx1*C*3+h_idx*d*3+i*3] + table_q[r_idx2*C*3+h_idx*d*3+i*3+1] + table_q[r_idx3*C*3+h_idx*d*3+i*3+2];
        sum += query_vec[i] * table_q_scalar_i;
        float table_k_scalar_i = table_k[r_idx1*C*3+h_idx*d*3+i*3] + table_k[r_idx2*C*3+h_idx*d*3+i*3+1] + table_k[r_idx3*C*3+h_idx*d*3+i*3+2];
        sum += k[k_idx*C+h_idx*d+i] * table_k_scalar_i;
    }
    output[m_idx*h + h_idx] = sum;

}

// N, M, h, hdim, grad_out, q, index_q, k, index_k, table_q, table_k, rel_idx, rel_idx_offsets, sort_indices, grad_q, grad_k, grad_table_q, grad_table_k

template <unsigned int d>
__global__ void dot_prod_with_idx_backward_cuda_kernel_v3( // M, h, hdim
    int N, int M, int h, const float *grad_out, const float *q, const int *index_q_offsets, 
    const float *k, const int *index_k, const float *table_q, const float *table_k, 
    const int *rel_idx, float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k) {

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int n_idx = threadIdx.x;
    int C = h*d;

    __shared__ float query_vec[d];
    __shared__ int start, end;
    if (n_idx == 0){
        start = index_q_offsets[q_idx];
        end = index_q_offsets[q_idx+1];
    }
    for(int i = n_idx; i < d; i += blockDim.x)
        query_vec[i] = q[q_idx*C + h_idx*d + i];

    __shared__ float gradients_q[d];
    for(int i = n_idx; i < d; i += blockDim.x){
        gradients_q[i] = 0;
    }

    __syncthreads();

    int m_idx = start + n_idx;

    if(m_idx < end){
        int k_idx = index_k[m_idx];
        int r_idx1 = rel_idx[m_idx*3], r_idx2 = rel_idx[m_idx*3+1], r_idx3 = rel_idx[m_idx*3+2];
        float gradient = grad_out[m_idx*h + h_idx];
        for(int i = 0; i < d; i++){
            float table_q_scalar_i = table_q[r_idx1*C*3+h_idx*d*3+i*3] + table_q[r_idx2*C*3+h_idx*d*3+i*3+1] + table_q[r_idx3*C*3+h_idx*d*3+i*3+2];
            float table_k_scalar_i = table_k[r_idx1*C*3+h_idx*d*3+i*3] + table_k[r_idx2*C*3+h_idx*d*3+i*3+1] + table_k[r_idx3*C*3+h_idx*d*3+i*3+2];
            float q_scalar_i = query_vec[i];
            float k_scalar_i = k[k_idx*C+h_idx*d+i];
            atomicAdd(&gradients_q[i], table_q_scalar_i * gradient);
            atomicAdd(grad_k+k_idx*C+h_idx*d+i, table_k_scalar_i * gradient);
            atomicAdd(grad_table_q+r_idx1*C*3+h_idx*d*3+i*3, q_scalar_i * gradient);
            atomicAdd(grad_table_q+r_idx2*C*3+h_idx*d*3+i*3+1, q_scalar_i * gradient);
            atomicAdd(grad_table_q+r_idx3*C*3+h_idx*d*3+i*3+2, q_scalar_i * gradient);
            atomicAdd(grad_table_k+r_idx1*C*3+h_idx*d*3+i*3, k_scalar_i * gradient);
            atomicAdd(grad_table_k+r_idx2*C*3+h_idx*d*3+i*3+1, k_scalar_i * gradient);
            atomicAdd(grad_table_k+r_idx3*C*3+h_idx*d*3+i*3+2, k_scalar_i * gradient);
        }
    }
    __syncthreads();

    for(int i = n_idx; i < d; i += blockDim.x){
        grad_q[q_idx*C+h_idx*d+i] = gradients_q[i];
    }
}

void dot_prod_with_idx_forward_cuda_launcher_v3(int N, int M, int h, int hdim, int n_max, const float *q,
    const int *index_q_offsets, const float *k, const int *index_k, const float *table_q, const float *table_k, 
    const int *rel_idx, float *output)
{
    // input: q: (N, h, hdim), index: (M), table: (L, h, hdim, 3), rel_idx: (M, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    dim3 blocks(N, h);
    // dim3 threads(THREADS_PER_BLOCK);
    
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;

    // printf("e1: h: %d, n_max: %d, n_threads: %d\n", h, n_max, n_threads);

	switch (hdim) {
        case 16:
            dot_prod_with_idx_forward_cuda_kernel_v3<16><<<blocks, n_threads, 0>>>(N, M, h, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
            break;
        case 32:
            dot_prod_with_idx_forward_cuda_kernel_v3<32><<<blocks, n_threads, 0>>>(N, M, h, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, output);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}

void dot_prod_with_idx_backward_cuda_launcher_v3(int N, int M, int h, int hdim, int n_max, 
    const float *grad_out, const float *q, const int *index_q_offsets, const float *k, const int *index_k, 
    const float *table_q, const float *table_k, const int *rel_idx, 
    float *grad_q, float *grad_k, float *grad_table_q, float *grad_table_k)
{  
    // input: grad_out: (M, h), output: grad_q: (N, h, hdim), grad_table: (L, h, hdim, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    // dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    // dim3 threads(THREADS_PER_BLOCK);

    dim3 blocks(N, h);
    // dim3 threads(THREADS_PER_BLOCK);
    
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;

	switch (hdim) {
        case 16:
            dot_prod_with_idx_backward_cuda_kernel_v3<16><<<blocks, n_threads, 0>>>(N, M, h, grad_out, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
            break;
        case 32:
            dot_prod_with_idx_backward_cuda_kernel_v3<32><<<blocks, n_threads, 0>>>(N, M, h, grad_out, q, index_q_offsets, k, index_k, table_q, table_k, rel_idx, grad_q, grad_k, grad_table_q, grad_table_k);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}


template <unsigned int d>
__global__ void attention_step2_with_rel_pos_value_forward_cuda_kernel_v2( // M, h, hdim
    int N, int M, int h, const float *attn, const float *v,
    const int *index0_offsets, const int *index1, const float *table, const int *rel_idx, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int n_idx = threadIdx.x;

    int C = h*d;

    __shared__ int start, end;
    __shared__ float result[d];

    if (n_idx == 0){
        start = index0_offsets[q_idx];
        end = index0_offsets[q_idx+1];
    }
    for (int i = n_idx; i < d; i += blockDim.x){
        result[i] = 0;
    }

    __syncthreads();

    int m_idx = start + n_idx;
    if (m_idx < end){
        float attn_scalar = attn[m_idx*h + h_idx];
        int r_idx1 = rel_idx[m_idx*3], r_idx2 = rel_idx[m_idx*3+1], r_idx3 = rel_idx[m_idx*3+2];
        for(int i = 0; i < d; i ++){
            int v_idx = index1[m_idx];
            float table_scaler_i = table[r_idx1*C*3+h_idx*d*3+i*3] + table[r_idx2*C*3+h_idx*d*3+i*3+1] + table[r_idx3*C*3+h_idx*d*3+i*3+2];
            float value_scaler_i = v[v_idx*C + h_idx*d + i];
            atomicAdd(&result[i], (table_scaler_i + value_scaler_i) * attn_scalar);
        }
    }

    __syncthreads();

    for (int i = n_idx; i < d; i += blockDim.x)
        output[q_idx*C + h_idx*d + i] = result[i];
}


template <unsigned int d>
__global__ void attention_step2_with_rel_pos_value_backward_cuda_kernel_v2( // M, h, hdim
    int N, int M, int h, const float *grad_out, const int *index0_offsets, const int *index1, const float *attn, const float *v, const float *table,
    const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)

    int q_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int n_idx = threadIdx.x;

    int C = h*d;

    __shared__ int start, end;
    __shared__ float gradients[d];

    if (n_idx == 0){
        start = index0_offsets[q_idx];
        end = index0_offsets[q_idx+1];
    }
    for (int i = n_idx; i < d; i += blockDim.x){
        gradients[i] = grad_out[q_idx*C + h_idx*d + i];
    }

    __syncthreads();

    int m_idx = start + n_idx;
    if (m_idx < end){
        int v_idx = index1[m_idx];
        int r_idx1 = rel_idx[m_idx*3], r_idx2 = rel_idx[m_idx*3+1], r_idx3 = rel_idx[m_idx*3+2];
        float attn_scalar = attn[m_idx*h + h_idx];
        float grad_attn_sum = 0;
        for (int i = 0; i < d; i++){
            float grad_out_scaler_i = gradients[i];
            float table_scaler_i = table[r_idx1*C*3+h_idx*d*3+i*3] + table[r_idx2*C*3+h_idx*d*3+i*3+1] + table[r_idx3*C*3+h_idx*d*3+i*3+2];
            float value_scaler_i = v[v_idx*C + h_idx*d + i];
            grad_attn_sum += (table_scaler_i + value_scaler_i) * grad_out_scaler_i;
            atomicAdd(grad_v + v_idx*C + h_idx*d + i, attn_scalar * grad_out_scaler_i);
            atomicAdd(grad_table + r_idx1*C*3 + h_idx*d*3 + i*3, attn_scalar * grad_out_scaler_i);
            atomicAdd(grad_table + r_idx2*C*3 + h_idx*d*3 + i*3 + 1, attn_scalar * grad_out_scaler_i);
            atomicAdd(grad_table + r_idx3*C*3 + h_idx*d*3 + i*3 + 2, attn_scalar * grad_out_scaler_i);
        }
        grad_attn[m_idx*h + h_idx] = grad_attn_sum;
    }
}

void attention_step2_with_rel_pos_value_forward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, const float *attn, const float *v, const int *index0_offsets,
    const int *index1, const float *table, const int *rel_idx, float *output) {
    // input: attn: (M, h), v: (N, h, hdim), index0: (M, ), index1: (M, ), table: (L, h, hdim, 3), rel_idx: (M, 3)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    // dim3 blocks(DIVUP(M*3, THREADS_PER_BLOCK), h, hdim);
    // dim3 threads(THREADS_PER_BLOCK);
    dim3 blocks(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;

	switch (hdim) {
        case 16:
            attention_step2_with_rel_pos_value_forward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, attn, v, index0_offsets, index1, table, rel_idx, output);
            break;
        case 32:
            attention_step2_with_rel_pos_value_forward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, attn, v, index0_offsets, index1, table, rel_idx, output);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}

void attention_step2_with_rel_pos_value_backward_cuda_launcher_v2(int N, int M, int h, int hdim, int n_max, const float *grad_out, const int *index0_offsets, 
    const int *index1, const float *attn, const float *v, const float *table, const int *rel_idx, float *grad_attn, float *grad_v, float *grad_table) {  
    // input: grad_output: (n, nsample, c), output: grad_input1: (n, c), grad_input2: (n, c)
    //dim3 blocks(DIVUP(hdim, THREADS_PER_BLOCK), h, M);
    
    dim3 blocks(N, h);
	unsigned int n_threads = opt_n_threads(n_max);
    n_threads = n_threads == n_max ? n_threads : n_threads * 2;

	switch (hdim) {
        case 16:
            attention_step2_with_rel_pos_value_backward_cuda_kernel_v2<16><<<blocks, n_threads, 0>>>(N, M, h, grad_out, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
            break;
        case 32:
            attention_step2_with_rel_pos_value_backward_cuda_kernel_v2<32><<<blocks, n_threads, 0>>>(N, M, h, grad_out, index0_offsets, index1, attn, v, table, rel_idx, grad_attn, grad_v, grad_table);
            break;
        default:
            throw "d != 16 and d != 32";
    }
}
