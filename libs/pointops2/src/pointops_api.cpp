#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knnquery/knnquery_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "grouping/grouping_cuda_kernel.h"
#include "interpolation/interpolation_cuda_kernel.h"
#include "aggregation/aggregation_cuda_kernel.h"
#include "subtraction/subtraction_cuda_kernel.h"
#include "attention/attention_cuda_kernel.h"
#include "rpe/relative_pos_encoding_cuda_kernel.h"
#include "attention_v2/attention_cuda_kernel_v2.h"
#include "rpe_v2/relative_pos_encoding_cuda_kernel_v2.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knnquery_cuda", &knnquery_cuda, "knnquery_cuda");
    m.def("furthestsampling_cuda", &furthestsampling_cuda, "furthestsampling_cuda");
    m.def("grouping_forward_cuda", &grouping_forward_cuda, "grouping_forward_cuda");
    m.def("grouping_backward_cuda", &grouping_backward_cuda, "grouping_backward_cuda");
    m.def("interpolation_forward_cuda", &interpolation_forward_cuda, "interpolation_forward_cuda");
    m.def("interpolation_backward_cuda", &interpolation_backward_cuda, "interpolation_backward_cuda");
    m.def("subtraction_forward_cuda", &subtraction_forward_cuda, "subtraction_forward_cuda");
    m.def("subtraction_backward_cuda", &subtraction_backward_cuda, "subtraction_backward_cuda");
    m.def("aggregation_forward_cuda", &aggregation_forward_cuda, "aggregation_forward_cuda");
    m.def("aggregation_backward_cuda", &aggregation_backward_cuda, "aggregation_backward_cuda");
    m.def("attention_step1_forward_cuda", &attention_step1_forward_cuda, "attention_step1_forward_cuda");
    m.def("attention_step1_backward_cuda", &attention_step1_backward_cuda, "attention_step1_backward_cuda");
    m.def("attention_step2_forward_cuda", &attention_step2_forward_cuda, "attention_step2_forward_cuda");
    m.def("attention_step2_backward_cuda", &attention_step2_backward_cuda, "attention_step2_backward_cuda");
    m.def("dot_prod_with_idx_forward_cuda", &dot_prod_with_idx_forward_cuda, "dot_prod_with_idx_forward_cuda");
    m.def("dot_prod_with_idx_backward_cuda", &dot_prod_with_idx_backward_cuda, "dot_prod_with_idx_backward_cuda");
    m.def("attention_step2_with_rel_pos_value_forward_cuda", &attention_step2_with_rel_pos_value_forward_cuda, "attention_step2_with_rel_pos_value_forward_cuda");
    m.def("attention_step2_with_rel_pos_value_backward_cuda", &attention_step2_with_rel_pos_value_backward_cuda, "attention_step2_with_rel_pos_value_backward_cuda");
    m.def("attention_step1_forward_cuda_v2", &attention_step1_forward_cuda_v2, "attention_step1_forward_cuda_v2");
    m.def("attention_step1_backward_cuda_v2", &attention_step1_backward_cuda_v2, "attention_step1_backward_cuda_v2");
    m.def("attention_step2_forward_cuda_v2", &attention_step2_forward_cuda_v2, "attention_step2_forward_cuda_v2");
    m.def("attention_step2_backward_cuda_v2", &attention_step2_backward_cuda_v2, "attention_step2_backward_cuda_v2");
    m.def("dot_prod_with_idx_forward_cuda_v2", &dot_prod_with_idx_forward_cuda_v2, "dot_prod_with_idx_forward_cuda_v2");
    m.def("dot_prod_with_idx_backward_cuda_v2", &dot_prod_with_idx_backward_cuda_v2, "dot_prod_with_idx_backward_cuda_v2");
    m.def("attention_step2_with_rel_pos_value_forward_cuda_v2", &attention_step2_with_rel_pos_value_forward_cuda_v2, "attention_step2_with_rel_pos_value_forward_cuda_v2");
    m.def("attention_step2_with_rel_pos_value_backward_cuda_v2", &attention_step2_with_rel_pos_value_backward_cuda_v2, "attention_step2_with_rel_pos_value_backward_cuda_v2");
    m.def("dot_prod_with_idx_forward_cuda_v3", &dot_prod_with_idx_forward_cuda_v3, "dot_prod_with_idx_forward_cuda_v3");
    m.def("dot_prod_with_idx_backward_cuda_v3", &dot_prod_with_idx_backward_cuda_v3, "dot_prod_with_idx_backward_cuda_v3");
    }
