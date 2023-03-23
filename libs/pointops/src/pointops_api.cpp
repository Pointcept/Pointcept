#include <torch/serialize/tensor.h>
#include <torch/extension.h>

#include "knn_query/knn_query_cuda_kernel.h"
#include "ball_query/ball_query_cuda_kernel.h"
#include "random_ball_query/random_ball_query_cuda_kernel.h"
#include "sampling/sampling_cuda_kernel.h"
#include "grouping/grouping_cuda_kernel.h"
#include "interpolation/interpolation_cuda_kernel.h"
#include "aggregation/aggregation_cuda_kernel.h"
#include "subtraction/subtraction_cuda_kernel.h"
#include "attention/attention_cuda_kernel.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("knn_query_cuda", &knn_query_cuda, "knn_query_cuda");
    m.def("ball_query_cuda", &ball_query_cuda, "ball_query_cuda");
    m.def("random_ball_query_cuda", &random_ball_query_cuda, "random_ball_query_cuda");
    m.def("farthest_point_sampling_cuda", &farthest_point_sampling_cuda, "farthest_point_sampling_cuda");
    m.def("grouping_forward_cuda", &grouping_forward_cuda, "grouping_forward_cuda");
    m.def("grouping_backward_cuda", &grouping_backward_cuda, "grouping_backward_cuda");
    m.def("interpolation_forward_cuda", &interpolation_forward_cuda, "interpolation_forward_cuda");
    m.def("interpolation_backward_cuda", &interpolation_backward_cuda, "interpolation_backward_cuda");
    m.def("subtraction_forward_cuda", &subtraction_forward_cuda, "subtraction_forward_cuda");
    m.def("subtraction_backward_cuda", &subtraction_backward_cuda, "subtraction_backward_cuda");
    m.def("aggregation_forward_cuda", &aggregation_forward_cuda, "aggregation_forward_cuda");
    m.def("aggregation_backward_cuda", &aggregation_backward_cuda, "aggregation_backward_cuda");
    m.def("attention_relation_step_forward_cuda", &attention_relation_step_forward_cuda, "attention_relation_step_forward_cuda");
    m.def("attention_relation_step_backward_cuda", &attention_relation_step_backward_cuda, "attention_relation_step_backward_cuda");
    m.def("attention_fusion_step_forward_cuda", &attention_fusion_step_forward_cuda, "attention_fusion_step_forward_cuda");
    m.def("attention_fusion_step_backward_cuda", &attention_fusion_step_backward_cuda, "attention_fusion_step_backward_cuda");
}
