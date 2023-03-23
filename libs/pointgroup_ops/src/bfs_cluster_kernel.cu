/*
Ball Query with BatchIdx
Written by Li Jiang
All Rights Reserved 2020.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TOTAL_THREADS 1024
#define THREADS_PER_BLOCK 512
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))


/* ================================== ballquery_batch_p ================================== */
__global__ void ballquery_batch_p_cuda_(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, int *cumsum) {
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= n) return;

    start_len += (pt_idx * 2);
    int idx_temp[1000];

    float radius2 = radius * radius;
    float o_x = xyz[pt_idx * 3 + 0];
    float o_y = xyz[pt_idx * 3 + 1];
    float o_z = xyz[pt_idx * 3 + 2];

    int batch_idx = batch_idxs[pt_idx];
    int start = batch_offsets[batch_idx];
    int end = batch_offsets[batch_idx + 1];

    int cnt = 0;
    for(int k = start; k < end; k++){
        float x = xyz[k * 3 + 0];
        float y = xyz[k * 3 + 1];
        float z = xyz[k * 3 + 2];
        float d2 = (o_x - x) * (o_x - x) + (o_y - y) * (o_y - y) + (o_z - z) * (o_z - z);
        if(d2 < radius2){
            if(cnt < 1000){
                idx_temp[cnt] = k;
            }
            else{
                break;
            }
            ++cnt;
        }
    }

    start_len[0] = atomicAdd(cumsum, cnt);
    start_len[1] = cnt;

    int thre = n * meanActive;
    if(start_len[0] >= thre) return;

    idx += start_len[0];
    if(start_len[0] + cnt >= thre) cnt = thre - start_len[0];

    for(int k = 0; k < cnt; k++){
        idx[k] = idx_temp[k];
    }
}


int ballquery_batch_p_cuda(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, cudaStream_t stream) {
    // param xyz: (n, 3)
    // param batch_idxs: (n)
    // param batch_offsets: (B + 1)
    // output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
    // output start_len: (n, 2), int

    cudaError_t err;

    dim3 blocks(DIVUP(n, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);

    int cumsum = 0;
    int* p_cumsum;
    cudaMalloc((void**)&p_cumsum, sizeof(int));
    cudaMemcpy(p_cumsum, &cumsum, sizeof(int), cudaMemcpyHostToDevice);

    ballquery_batch_p_cuda_<<<blocks, threads, 0, stream>>>(n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len, p_cumsum);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    cudaMemcpy(&cumsum, p_cumsum, sizeof(int), cudaMemcpyDeviceToHost);
    return cumsum;
}
