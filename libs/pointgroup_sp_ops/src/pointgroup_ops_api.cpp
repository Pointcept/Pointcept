#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include "pointgroup_ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("voxelize_idx", &voxelize_idx_3d, "voxelize_idx");
    m.def("voxelize_fp", &voxelize_fp_feat, "voxelize_fp");
    m.def("voxelize_bp", &voxelize_bp_feat, "voxelize_bp");
    m.def("point_recover_fp", &point_recover_fp_feat, "point_recover_fp");
    m.def("point_recover_bp", &point_recover_bp_feat, "point_recover_bp");

    m.def("ballquery_batch_p", &ballquery_batch_p, "ballquery_batch_p");
    m.def("bfs_cluster", &bfs_cluster, "bfs_cluster");

    m.def("roipool_fp", &roipool_fp, "roipool_fp");
    m.def("roipool_bp", &roipool_bp, "roipool_bp");

    m.def("get_iou", &get_iou, "get_iou");

    m.def("sec_mean", &sec_mean, "sec_mean");
    m.def("sec_min", &sec_min, "sec_min");
    m.def("sec_max", &sec_max, "sec_max");
}
