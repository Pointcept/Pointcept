/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#ifndef BFS_CLUSTER_H
#define BFS_CLUSTER_H
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
//#include <THC/THC.h>

#include "../datatype/datatype.h"

int ballquery_batch_p(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor batch_offsets_tensor, at::Tensor idx_tensor, at::Tensor start_len_tensor, int n, int meanActive, float radius);
int ballquery_batch_p_cuda(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, cudaStream_t stream);

void bfs_cluster(at::Tensor semantic_label_tensor, at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor, at::Tensor cluster_idxs_tensor, at::Tensor cluster_offsets_tensor, const int N, int threshold);

#endif //BFS_CLUSTER_H
