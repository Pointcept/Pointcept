/*
Points to Voxels & Voxels to Points (Modified from SparseConv)
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef VOXELIZE_H
#define VOXELIZE_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"

/* ================================== voxelize_idx ================================== */
template <Int dimension>
void voxelize_idx(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode);

template <Int dimension>
void voxelize_outputmap(long *coords, long *output_coords, Int *output_map, Int *rule, Int nOutputRows, Int maxActive);

template <Int dimension>
Int voxelize_inputmap(SparseGrids<dimension> &SGs, Int *input_map, RuleBook &rules, Int &nActive, long *coords, Int nInputRows, Int nInputColumns, Int batchSize, Int mode);

/* ================================== voxelize ================================== */
template <typename T>
void voxelize_fp(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane);

template <typename T>
void voxelize_fp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *feats, T *output_feats, Int *rules, bool average);


//
template <typename T>
void voxelize_bp(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane);

template <typename T>
void voxelize_bp_cuda(Int nOutputRows, Int maxActive, Int nPlanes, T *d_output_feats, T *d_feats, Int *rules, bool average);


/* ================================== point_recover ================================== */
template <typename T>
void point_recover_fp(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);

//
template <typename T>
void point_recover_bp(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);


#endif //VOXELIZE_H
