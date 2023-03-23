#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "random_ball_query_cuda_kernel.h"


void random_ball_query_cuda(int m, int nsample,
                            float min_radius, float max_radius, at::Tensor order_tensor,
                            at::Tensor xyz_tensor, at::Tensor new_xyz_tensor,
                            at::Tensor offset_tensor, at::Tensor new_offset_tensor,
                            at::Tensor idx_tensor, at::Tensor dist2_tensor)
{
    const int *order = order_tensor.data_ptr<int>();
    const float *xyz = xyz_tensor.data_ptr<float>();
    const float *new_xyz = new_xyz_tensor.data_ptr<float>();
    const int *offset = offset_tensor.data_ptr<int>();
    const int *new_offset = new_offset_tensor.data_ptr<int>();
    int *idx = idx_tensor.data_ptr<int>();
    float *dist2 = dist2_tensor.data_ptr<float>();
    random_ball_query_cuda_launcher(m, nsample, min_radius, max_radius, order, xyz, new_xyz, offset, new_offset, idx, dist2);
}
