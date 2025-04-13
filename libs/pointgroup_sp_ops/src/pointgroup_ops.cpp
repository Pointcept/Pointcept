#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "datatype/datatype.cpp"

#include "voxelize/voxelize.cpp"
#include "bfs_cluster/bfs_cluster.cpp"
#include "roipool/roipool.cpp"
#include "get_iou/get_iou.cpp"
#include "sec_mean/sec_mean.cpp"

void voxelize_idx_3d(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode){
    voxelize_idx<3>(coords, output_coords, input_map, output_map, batchSize, mode);
}

void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane){
    voxelize_fp<float>(feats, output_feats, output_map, mode, nActive, maxActive, nPlane);
}


void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane){
    voxelize_bp<float>(d_output_feats, d_feats, output_map, mode, nActive, maxActive, nPlane);
}

void point_recover_fp_feat(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    point_recover_fp<float>(feats, output_feats, idx_map, nActive, maxActive, nPlane);
}

void point_recover_bp_feat(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    point_recover_bp<float>(d_output_feats, d_feats, idx_map, nActive, maxActive, nPlane);
}
