/*
ROI Max Pool
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef ROIPOOL_H
#define ROIPOOL_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

#include "../datatype/datatype.h"

//
void roipool_fp(at::Tensor feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_feats_tensor, at::Tensor output_maxidx_tensor, int nProposal, int C);

void roipool_fp_cuda(int nProposal, int C, float *feats, int *proposals_offset, float *output_feats, int *output_maxidx);


//
void roipool_bp(at::Tensor d_feats_tensor, at::Tensor proposals_offset_tensor, at::Tensor output_maxidx_tensor, at::Tensor d_output_feats_tensor, int nProposal, int C);

void roipool_bp_cuda(int nProposal, int C, float *d_feats, int *proposals_offset, int *output_maxidx, float *d_output_feats);

#endif //ROIPOOL_H
