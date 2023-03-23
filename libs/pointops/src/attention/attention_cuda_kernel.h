#ifndef _ATTENTION_CUDA_KERNEL
#define _ATTENTION_CUDA_KERNEL
#include <vector>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>

void attention_relation_step_forward_cuda(int m, int g, int c,
                                          at::Tensor query_tensor, at::Tensor key_tensor, at::Tensor weight_tensor,
                                          at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                          at::Tensor output_tensor);
void attention_relation_step_backward_cuda(int m, int g, int c,
                                           at::Tensor query_tensor, at::Tensor grad_query_tensor,
                                           at::Tensor key_tensor, at::Tensor grad_key_tensor,
                                           at::Tensor weight_tensor, at::Tensor grad_weight_tensor,
                                           at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                           at::Tensor grad_output_tensor);
void attention_fusion_step_forward_cuda(int m, int g, int c,
                                        at::Tensor weight_tensor, at::Tensor value_tensor,
                                        at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                        at::Tensor output_tensor);
void attention_fusion_step_backward_cuda(int m, int g, int c,
                                         at::Tensor weight_tensor, at::Tensor grad_weight_tensor,
                                         at::Tensor value_tensor, at::Tensor grad_value_tensor,
                                         at::Tensor index_target_tensor, at::Tensor index_refer_tensor,
                                         at::Tensor grad_output_tensor);

#ifdef __cplusplus
extern "C" {
#endif

void attention_relation_step_forward_cuda_launcher(int m, int g, int c,
                                                   const float *query, const float *key, const float *weight,
                                                   const int *index_target, const int *index_refer,
                                                   float *output);
void attention_relation_step_backward_cuda_launcher(int m, int g, int c,
                                                    const float *query, float *grad_query,
                                                    const float *key, float *grad_key,
                                                    const float *weight, float *grad_weight,
                                                    const int *index_target, const int *index_refer,
                                                    const float *grad_output);
void attention_fusion_step_forward_cuda_launcher(int m, int g, int c,
                                                 const float *weight, const float *value,
                                                 const int *index_target, const int *index_refer,
                                                 float *output);
void attention_fusion_step_backward_cuda_launcher(int m, int g, int c,
                                                  const float *weight, float *grad_weight,
                                                  const float *value, float *grad_value,
                                                  const int *index_target, const int *index_refer,
                                                  const float *grad_output);

#ifdef __cplusplus
}
#endif
#endif
