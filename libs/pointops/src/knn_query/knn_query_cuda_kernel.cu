#include "../cuda_utils.h"
#include "knn_query_cuda_kernel.h"


namespace knn_query_utils{

template <typename DType>
__device__ void swap(DType *x, DType *y)
{
    DType tmp = *x;
    *x = *y;
    *y = tmp;
}

__device__ void reheap(float *dist, int *idx, int k)
{
    int root = 0;
    int child = root * 2 + 1;
    while (child < k)
    {
        if(child + 1 < k && dist[child+1] > dist[child])
            child++;
        if(dist[root] > dist[child])
            return;
        swap<float>(&dist[root], &dist[child]);
        swap<int>(&idx[root], &idx[child]);
        root = child;
        child = root * 2 + 1;
    }
}


__device__ void heap_sort(float *dist, int *idx, int k)
{
    int i;
    for (i = k - 1; i > 0; i--)
    {
        swap<float>(&dist[0], &dist[i]);
        swap<int>(&idx[0], &idx[i]);
        reheap(dist, idx, i);
    }
}


__device__ int get_bt_idx(int idx, const int *offset)
{
    int i = 0;
    while (1)
    {
        if (idx < offset[i])
            break;
        else
            i++;
    }
    return i;
}
}  // namespace knn_query_utils


__global__ void knn_query_cuda_kernel(int m, int nsample, const float *__restrict__ xyz, const float *__restrict__ new_xyz, const int *__restrict__ offset, const int *__restrict__ new_offset, int *__restrict__ idx, float *__restrict__ dist2) {
    // input: xyz (n, 3) new_xyz (m, 3)
    // output: idx (m, nsample) dist2 (m, nsample)
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt_idx >= m) return;

    new_xyz += pt_idx * 3;
    idx += pt_idx * nsample;
    dist2 += pt_idx * nsample;

    int bt_idx = knn_query_utils::get_bt_idx(pt_idx, new_offset);
    int start;
    if (bt_idx == 0)
        start = 0;
    else
        start = offset[bt_idx - 1];
    int end = offset[bt_idx];

    float new_x = new_xyz[0];
    float new_y = new_xyz[1];
    float new_z = new_xyz[2];

    float best_dist[128];
    int best_idx[128];
    for(int i = 0; i < nsample; i++){
        best_dist[i] = 1e10;
        best_idx[i] = -1;
    }
    for(int i = start; i < end; i++){
        float x = xyz[i * 3 + 0];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) + (new_z - z) * (new_z - z);
        if (d2 < best_dist[0]){
            best_dist[0] = d2;
            best_idx[0] = i;
            knn_query_utils::reheap(best_dist, best_idx, nsample);
        }
    }
    knn_query_utils::heap_sort(best_dist, best_idx, nsample);
    for(int i = 0; i < nsample; i++){
        idx[i] = best_idx[i];
        dist2[i] = best_dist[i];
    }
}


void knn_query_cuda_launcher(int m, int nsample, const float *xyz, const float *new_xyz, const int *offset, const int *new_offset, int *idx, float *dist2) {
    // input: new_xyz: (m, 3), xyz: (n, 3), idx: (m, nsample)
    dim3 blocks(DIVUP(m, THREADS_PER_BLOCK));
    dim3 threads(THREADS_PER_BLOCK);
    knn_query_cuda_kernel<<<blocks, threads, 0>>>(m, nsample, xyz, new_xyz, offset, new_offset, idx, dist2);
}
