#include "../cuda_utils.h"
#include "sampling_cuda_kernel.h"


__device__ void __update(float *dists, int *dists_i, int idx1, int idx2) {
    const float v1 = dists[idx1], v2 = dists[idx2];
    const int i1 = dists_i[idx1], i2 = dists_i[idx2];
    dists[idx1] = max(v1, v2);
    dists_i[idx1] = v2 > v1 ? i2 : i1;
}

// input xyz: (n, 3), tmp: (b, n_max)
// ouput idx (m)
template <unsigned int block_size>
__global__ void furthestsampling_cuda_kernel(const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx)
{
    __shared__ float dists[block_size];
    __shared__ int dists_i[block_size];

    int bid = blockIdx.x;
    int start_n, end_n, start_m, end_m, old;
    if (bid == 0) {
        start_n = 0;
        end_n = offset[0];
        start_m = 0;
        end_m = new_offset[0];
        old = 0;
    }
    else {
        start_n = offset[bid - 1];
        end_n = offset[bid];
        start_m = new_offset[bid - 1];
        end_m = new_offset[bid];
        old = offset[bid - 1];
    }

    const int stride = block_size;
    int tid = threadIdx.x;
    if (tid == 0) idx[start_m] = start_n;

    __syncthreads();
    for (int j = start_m + 1; j < end_m; j++)
    {
        int besti = start_n;
        float best = -1;
        float x1 = xyz[old * 3 + 0];
        float y1 = xyz[old * 3 + 1];
        float z1 = xyz[old * 3 + 2];
        for (int k = start_n + tid; k < end_n; k += stride)
        {
            float x2 = xyz[k * 3 + 0];
            float y2 = xyz[k * 3 + 1];
            float z2 = xyz[k * 3 + 2];
            float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);
            float d2 = min(d, tmp[k]);
            tmp[k] = d2;
            besti = d2 > best ? k : besti;
            best = d2 > best ? d2 : best;
        }
        dists[tid] = best;
        dists_i[tid] = besti;
        __syncthreads();

        if (block_size >= 1024) {
            if (tid < 512) {
            __update(dists, dists_i, tid, tid + 512);
            }
            __syncthreads();
        }
        if (block_size >= 512) {
            if (tid < 256) {
            __update(dists, dists_i, tid, tid + 256);
            }
            __syncthreads();
        }
        if (block_size >= 256) {
            if (tid < 128) {
            __update(dists, dists_i, tid, tid + 128);
            }
            __syncthreads();
        }
        if (block_size >= 128) {
            if (tid < 64) {
            __update(dists, dists_i, tid, tid + 64);
            }
            __syncthreads();
        }
        if (block_size >= 64) {
            if (tid < 32) {
            __update(dists, dists_i, tid, tid + 32);
            }
            __syncthreads();
        }
        if (block_size >= 32) {
            if (tid < 16) {
            __update(dists, dists_i, tid, tid + 16);
            }
            __syncthreads();
        }
        if (block_size >= 16) {
            if (tid < 8) {
            __update(dists, dists_i, tid, tid + 8);
            }
            __syncthreads();
        }
        if (block_size >= 8) {
            if (tid < 4) {
            __update(dists, dists_i, tid, tid + 4);
            }
            __syncthreads();
        }
        if (block_size >= 4) {
            if (tid < 2) {
            __update(dists, dists_i, tid, tid + 2);
            }
            __syncthreads();
        }
        if (block_size >= 2) {
            if (tid < 1) {
            __update(dists, dists_i, tid, tid + 1);
            }
            __syncthreads();
        }

        old = dists_i[0];
        if (tid == 0)
            idx[j] = old;
    }
}

void furthestsampling_cuda_launcher(int b, int n, const float *xyz, const int *offset, const int *new_offset, float *tmp, int *idx)
{   
	unsigned int n_threads = opt_n_threads(n);
	switch (n_threads) {
        case 1024:
            furthestsampling_cuda_kernel<1024><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 512:
            furthestsampling_cuda_kernel<512><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 256:
            furthestsampling_cuda_kernel<256><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 128:
            furthestsampling_cuda_kernel<128><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 64:
            furthestsampling_cuda_kernel<64><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 32:
            furthestsampling_cuda_kernel<32><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 16:
            furthestsampling_cuda_kernel<16><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 8:
            furthestsampling_cuda_kernel<8><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 4:
            furthestsampling_cuda_kernel<4><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 2:
            furthestsampling_cuda_kernel<2><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        case 1:
            furthestsampling_cuda_kernel<1><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
            break;
        default:
            furthestsampling_cuda_kernel<512><<<b, n_threads, 0>>>(xyz, offset, new_offset, tmp, idx);
    }
}
