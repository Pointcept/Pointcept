#ifndef POINTGROUP_H
#define POINTGROUP_H
#include "datatype/datatype.h"

#include "bfs_cluster/bfs_cluster.h"
#include "roipool/roipool.h"
#include "get_iou/get_iou.h"
#include "sec_mean/sec_mean.h"

void voxelize_idx_3d(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode);

void voxelize_fp_feat(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane);

void voxelize_bp_feat(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane);

void point_recover_fp_feat(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);

void point_recover_bp_feat(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane);


#endif // POINTGROUP_H
