/*
Points to Voxels & Voxels to Points (Modified from SparseConv)
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "voxelize.h"

template <typename T>
__global__ void voxelize_fp_cuda_(Int nOutputRows, Int maxActive, Int nPlanes, T *feats, T *output_feats, Int *rules, bool average){
    for(int row = blockIdx.x; row < nOutputRows; row += gridDim.x){
        T *out = output_feats + row * nPlanes;
        Int *r = rules + row * (maxActive + 1);
        Int nActive = r[0];
        T multiplier = (average and nActive > 0) ? (T) 1 / nActive : (T) 1;
        for(int i = 1; i <= nActive; i++){
            T *inp = feats + r[i] * nPlanes;
            for(int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x){
                atomicAdd(&out[plane], multiplier * inp[plane]);
            }
        }
     }
}

// input: feats N * C
// input: rules M * (1 + maxActive)
// output: output_feats M * C
template <typename T>
void voxelize_fp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *feats, T *output_feats, Int *rules, bool average){
    voxelize_fp_cuda_<T><<<std::min(nOutputRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(nOutputRows, maxActive, nPlanes, feats, output_feats, rules, average);
}


template <typename T>
__global__ void voxelize_bp_cuda_(Int nOutputRows, Int maxActive, Int nPlanes, T *d_output_feats, T *d_feats, Int *rules, bool average){
    for(int row = blockIdx.x; row < nOutputRows; row += gridDim.x){
        T *out = d_output_feats + row * nPlanes;
        Int *r = rules + row * (maxActive + 1);
        Int nActive = r[0];
        T multiplier = (average and nActive > 0) ? (T) 1 / nActive : (T) 1;
        for(int i = 1; i <= nActive; i++){
            T *inp = d_feats + r[i] * nPlanes;
            for(int plane = threadIdx.x; plane < nPlanes; plane += blockDim.x){
                atomicAdd(&inp[plane], multiplier * out[plane]);
            }
        }
    }
}

template <typename T>
void voxelize_bp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *d_output_feats, T *d_feats, Int *rules, bool average){
    voxelize_bp_cuda_<T><<<std::min(nOutputRows, (Int)32768), std::min(nPlanes, (Int)32)>>>(nOutputRows, maxActive, nPlanes, d_output_feats, d_feats, rules, average);
}
