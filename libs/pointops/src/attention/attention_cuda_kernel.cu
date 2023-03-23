#include "../cuda_utils.h"
#include "attention_cuda_kernel.h"


/*
Kernels
*/

__global__ void attention_relation_step_forward_cuda_kernel(int m, int g, int c,
                                                            const float *query, const float *key, const float *weight,
                                                            const int *index_target, const int *index_refer,
                                                            float *output)
{
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_idx = blockIdx.y;
    int c_idx = blockIdx.z;

    if (r_idx >= m || g_idx >= g || c_idx >= c) return;
    int q_idx = index_target[r_idx] * g * c + g_idx * c + c_idx;
    int k_idx = index_refer[r_idx] * g * c + g_idx * c + c_idx;

    float r = query[q_idx] * key[k_idx] * weight[c_idx];
    atomicAdd(output + r_idx * g + g_idx, r);
}

__global__ void attention_relation_step_backward_cuda_kernel(int m, int g, int c,
                                                             const float *query, float *grad_query,
                                                             const float *key, float *grad_key,
                                                             const float *weight, float *grad_weight,
                                                             const int *index_target, const int *index_refer,
                                                             const float *grad_output)
{
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_idx = blockIdx.y;
    int c_idx = blockIdx.z;

    if (r_idx >= m || g_idx >= g || c_idx >= c) return;

    int q_idx = index_target[r_idx] * g * c + g_idx * c + c_idx;
    int k_idx = index_refer[r_idx] * g * c + g_idx * c + c_idx;
    int o_idx = r_idx * g + g_idx;
    float grad_r = grad_output[o_idx];
    atomicAdd(grad_query + q_idx, grad_r * key[k_idx] * weight[c_idx]);
    atomicAdd(grad_key + k_idx, grad_r * query[q_idx] * weight[c_idx]);
    atomicAdd(grad_weight + c_idx, grad_r * key[k_idx] * query[q_idx]);
}


__global__ void attention_fusion_step_forward_cuda_kernel(int m, int g, int c,
                                                          const float *weight, const float *value,
                                                          const int *index_target, const int *index_refer,
                                                          float *output)
{
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_idx = blockIdx.y;
    int c_idx = blockIdx.z;

    if (r_idx >= m || g_idx >= g || c_idx >= c) return;

    int o_idx = index_target[r_idx] * g * c + g_idx * c + c_idx;
    int v_idx = index_refer[r_idx] * g * c + g_idx * c + c_idx;

    float f = weight[r_idx * g + g_idx] * value[v_idx];
    atomicAdd(output + o_idx, f);
}


__global__ void attention_fusion_step_backward_cuda_kernel(int m, int g, int c,
                                                           const float *weight, float *grad_weight,
                                                           const float *value, float *grad_value,
                                                           const int *index_target, const int *index_refer,
                                                           const float *grad_output)
{
    int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int g_idx = blockIdx.y;
    int c_idx = blockIdx.z;

    if (r_idx >= m || g_idx >= g || c_idx >= c) return;

    int o_idx = index_target[r_idx] * g * c + g_idx * c + c_idx;
    int v_idx = index_refer[r_idx] * g * c + g_idx * c + c_idx;
    int w_idx = r_idx * g + g_idx;
    float grad = grad_output[o_idx];
    atomicAdd(grad_weight + w_idx, grad * value[v_idx]);
    atomicAdd(grad_value + v_idx, grad * weight[w_idx]);
}

/*
Launchers
*/


void attention_relation_step_forward_cuda_launcher(int m, int g, int c,
                                                   const float *query, const float *key, const float *weight,
                                                   const int *index_target, const int *index_refer,
                                                   float *output)
{
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), g, c);
    dim3 threads(THREADS_PER_BLOCK);
    attention_relation_step_forward_cuda_kernel<<<blocks, threads, 0>>>(m, g, c, query, key, weight,
                                                                        index_target, index_refer, output);
}

void attention_relation_step_backward_cuda_launcher(int m, int g, int c,
                                                    const float *query, float *grad_query,
                                                    const float *key, float *grad_key,
                                                    const float *weight, float *grad_weight,
                                                    const int *index_target, const int *index_refer,
                                                    const float *grad_output)
{
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), g, c);
    dim3 threads(THREADS_PER_BLOCK);
    attention_relation_step_backward_cuda_kernel<<<blocks, threads, 0>>>(m, g, c,
                                                                         query, grad_query,
                                                                         key, grad_key,
                                                                         weight, grad_weight,
                                                                         index_target, index_refer,
                                                                         grad_output);
}


void attention_fusion_step_forward_cuda_launcher(int m, int g, int c,
                                                 const float *weight, const float *value,
                                                 const int *index_target, const int *index_refer,
                                                 float *output)
{
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), g, c);
    dim3 threads(THREADS_PER_BLOCK);
    attention_fusion_step_forward_cuda_kernel<<<blocks, threads, 0>>>(m, g, c, weight, value,
                                                                      index_target, index_refer, output);
}


void attention_fusion_step_backward_cuda_launcher(int m, int g, int c,
                                                  const float *weight, float *grad_weight,
                                                  const float *value, float *grad_value,
                                                  const int *index_target, const int *index_refer,
                                                  const float *grad_output)
{
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK), g, c);
    dim3 threads(THREADS_PER_BLOCK);
    attention_fusion_step_backward_cuda_kernel<<<blocks, threads, 0>>>(m, g, c,
                                                                       weight, grad_weight,
                                                                       value, grad_value,
                                                                       index_target, index_refer,
                                                                       grad_output);
}


