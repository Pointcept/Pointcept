/*
Segment Operations (mean, max, min)
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "sec_mean.h"

void sec_mean(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C){
    int *offsets = offsets_tensor.data<int>();
    float *inp = inp_tensor.data<float>();
    float *out = out_tensor.data<float>();

    sec_mean_cuda(nProposal, C, inp, offsets, out);
}

void sec_min(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C){
    int *offsets = offsets_tensor.data<int>();
    float *inp = inp_tensor.data<float>();
    float *out = out_tensor.data<float>();

    sec_min_cuda(nProposal, C, inp, offsets, out);
}

void sec_max(at::Tensor inp_tensor, at::Tensor offsets_tensor, at::Tensor out_tensor, int nProposal, int C){
    int *offsets = offsets_tensor.data<int>();
    float *inp = inp_tensor.data<float>();
    float *out = out_tensor.data<float>();

    sec_max_cuda(nProposal, C, inp, offsets, out);
}
