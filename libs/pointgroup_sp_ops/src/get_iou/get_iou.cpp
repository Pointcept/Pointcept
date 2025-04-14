/*
Get the IoU between predictions and gt masks
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "get_iou.h"

void get_iou(at::Tensor proposals_idx_tensor, at::Tensor proposals_offset_tensor, at::Tensor instance_labels_tensor, at::Tensor instance_pointnum_tensor, at::Tensor proposals_iou_tensor, int nInstance, int nProposal){
    int *proposals_idx = proposals_idx_tensor.data<int>();
    int *proposals_offset = proposals_offset_tensor.data<int>();
    long *instance_labels = instance_labels_tensor.data<long>();
    int *instance_pointnum = instance_pointnum_tensor.data<int>();

    float *proposals_iou = proposals_iou_tensor.data<float>();

    get_iou_cuda(nInstance, nProposal, proposals_idx, proposals_offset, instance_labels, instance_pointnum, proposals_iou);
}
