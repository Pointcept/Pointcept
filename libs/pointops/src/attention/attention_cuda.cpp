#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include "attention_cuda_kernel.h"


void attention_relation_step_forward_cuda(int m, int g, int c,
                                          at::Tensor query_tensor, at::Tensor key_tensor, at::Tensor weight_tensor,
                                          at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                          at::Tensor output_tensor)
{
    const float *query = query_tensor.data_ptr<float>();
    const float *key = key_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    const int *index_target = index_target_tensor.data_ptr<int>();
    const int *index_refer = index_refer_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_relation_step_forward_cuda_launcher(m, g, c, query, key, weight, index_target, index_refer, output);
}

void attention_relation_step_backward_cuda(int m, int g, int c,
                                           at::Tensor query_tensor, at::Tensor grad_query_tensor,
                                           at::Tensor key_tensor, at::Tensor grad_key_tensor,
                                           at::Tensor weight_tensor, at::Tensor grad_weight_tensor,
                                           at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                           at::Tensor grad_output_tensor)
{
    const float *query = query_tensor.data_ptr<float>();
    float *grad_query = grad_query_tensor.data_ptr<float>();
    const float *key = key_tensor.data_ptr<float>();
    float *grad_key = grad_key_tensor.data_ptr<float>();
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_weight = grad_weight_tensor.data_ptr<float>();
    const int *index_target = index_target_tensor.data_ptr<int>();
    const int *index_refer = index_refer_tensor.data_ptr<int>();
    const float *grad_output = grad_output_tensor.data_ptr<float>();
    attention_relation_step_backward_cuda_launcher(m, g, c,
                                                   query, grad_query,
                                                   key, grad_key,
                                                   weight, grad_weight,
                                                   index_target, index_refer, grad_output);
}


void attention_fusion_step_forward_cuda(int m, int g, int c,
                                        at::Tensor weight_tensor, at::Tensor value_tensor,
                                        at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                        at::Tensor output_tensor)
{
    const float *weight = weight_tensor.data_ptr<float>();
    const float *value = value_tensor.data_ptr<float>();
    const int *index_target = index_target_tensor.data_ptr<int>();
    const int *index_refer = index_refer_tensor.data_ptr<int>();
    float *output = output_tensor.data_ptr<float>();
    attention_fusion_step_forward_cuda_launcher(m, g, c, weight, value, index_target, index_refer, output);
}


void attention_fusion_step_backward_cuda(int m, int g, int c,
                                         at::Tensor weight_tensor, at::Tensor grad_weight_tensor,
                                         at::Tensor value_tensor, at::Tensor grad_value_tensor,
                                         at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                         at::Tensor grad_output_tensor)
{
    const float *weight = weight_tensor.data_ptr<float>();
    float *grad_weight = grad_weight_tensor.data_ptr<float>();
    const float *value = value_tensor.data_ptr<float>();
    float *grad_value = grad_value_tensor.data_ptr<float>();
    const int *index_target = index_target_tensor.data_ptr<int>();
    const int *index_refer = index_refer_tensor.data_ptr<int>();
    const float *grad_output = grad_output_tensor.data_ptr<float>();
    attention_fusion_step_backward_cuda_launcher(m, g, c,
                                                 weight, grad_weight,
                                                 value, grad_value,
                                                 index_target, index_refer, grad_output);
}
