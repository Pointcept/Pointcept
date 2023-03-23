/*
Ball Query with BatchIdx & Clustering Algorithm
Written by Li Jiang
All Rights Reserved 2020.
*/

#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cstdint>
#include <array>
#include <vector>
#include <queue>
#include <google/dense_hash_map>

int ballquery_batch_p_cuda(int n, int meanActive, float radius, const float *xyz, const int *batch_idxs, const int *batch_offsets, int *idx, int *start_len, cudaStream_t stream);


using Int = int32_t;
class ConnectedComponent{
public:
    std::vector<Int> pt_idxs {};

    ConnectedComponent(){};
    void addPoint(Int pt_idx)
    {
	pt_idxs.push_back(pt_idx);

    }
};
using ConnectedComponents = std::vector<ConnectedComponent>;

/* ================================== ballquery_batch_p ================================== */
// input xyz: (n, 3) float
// input batch_idxs: (n) int
// input batch_offsets: (B+1) int, batch_offsets[-1]
// output idx: (n * meanActive) dim 0 for number of points in the ball, idx in n
// output start_len: (n, 2), int
int ballquery_batch_p(at::Tensor xyz_tensor, at::Tensor batch_idxs_tensor, at::Tensor batch_offsets_tensor, at::Tensor idx_tensor, at::Tensor start_len_tensor, int n, int meanActive, float radius){
    const float *xyz = xyz_tensor.data<float>();
    const int *batch_idxs = batch_idxs_tensor.data<int>();
    const int *batch_offsets = batch_offsets_tensor.data<int>();
    int *idx = idx_tensor.data<int>();
    int *start_len = start_len_tensor.data<int>();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    int cumsum = ballquery_batch_p_cuda(n, meanActive, radius, xyz, batch_idxs, batch_offsets, idx, start_len, stream);
    return cumsum;
}

/* ================================== bfs_cluster ================================== */
ConnectedComponent find_cc(Int idx, int *semantic_label, Int *ball_query_idxs, int *start_len, int *visited){
    ConnectedComponent cc;
    cc.addPoint(idx);
    visited[idx] = 1;

    std::queue<Int> Q;
    assert(Q.empty());
    Q.push(idx);

    while(!Q.empty()){
        Int cur = Q.front(); Q.pop();
        int start = start_len[cur * 2];
        int len = start_len[cur * 2 + 1];
        int label_cur = semantic_label[cur];
        for(Int i = start; i < start + len; i++){
            Int idx_i = ball_query_idxs[i];
            if(semantic_label[idx_i] != label_cur) continue;
            if(visited[idx_i] == 1) continue;

            cc.addPoint(idx_i);
            visited[idx_i] = 1;

            Q.push(idx_i);
        }
    }
    return cc;
}

//input: semantic_label, int, N
//input: ball_query_idxs, Int, (nActive)
//input: start_len, int, (N, 2)
//output: clusters, CCs
int get_clusters(int *semantic_label, Int *ball_query_idxs, int *start_len, const Int nPoint, int threshold, ConnectedComponents &clusters){
    int visited[nPoint] = {0};

    int sumNPoint = 0;
    for(Int i = 0; i < nPoint; i++){
        if(visited[i] == 0){
            ConnectedComponent CC = find_cc(i, semantic_label, ball_query_idxs, start_len, visited);
            if((int)CC.pt_idxs.size() >= threshold){
                clusters.push_back(CC);
                sumNPoint += (int)CC.pt_idxs.size();
            }
        }
    }

    return sumNPoint;
}

void fill_cluster_idxs_(ConnectedComponents &CCs, int *cluster_idxs, int *cluster_offsets){
    for(int i = 0; i < (int)CCs.size(); i++){
        cluster_offsets[i + 1] = cluster_offsets[i] + (int)CCs[i].pt_idxs.size();
        for(int j = 0; j < (int)CCs[i].pt_idxs.size(); j++){
            int idx = CCs[i].pt_idxs[j];
            cluster_idxs[(cluster_offsets[i] + j) * 2 + 0] = i;
            cluster_idxs[(cluster_offsets[i] + j) * 2 + 1] = idx;
        }
    }
}

//input: semantic_label, int, N
//input: ball_query_idxs, int, (nActive)
//input: start_len, int, (N, 2)
//output: cluster_idxs, int (sumNPoint, 2), dim 0 for cluster_id, dim 1 for corresponding point idxs in N
//output: cluster_offsets, int (nCluster + 1)
void bfs_cluster(at::Tensor semantic_label_tensor, at::Tensor ball_query_idxs_tensor, at::Tensor start_len_tensor,
at::Tensor cluster_idxs_tensor, at::Tensor cluster_offsets_tensor, const int N, int threshold){
    int *semantic_label = semantic_label_tensor.data<int>();
    Int *ball_query_idxs = ball_query_idxs_tensor.data<Int>();
    int *start_len = start_len_tensor.data<int>();

    ConnectedComponents CCs;
    int sumNPoint = get_clusters(semantic_label, ball_query_idxs, start_len, N, threshold, CCs);

    int nCluster = (int)CCs.size();
    cluster_idxs_tensor.resize_({sumNPoint, 2});
    cluster_offsets_tensor.resize_({nCluster + 1});
    cluster_idxs_tensor.zero_();
    cluster_offsets_tensor.zero_();

    int *cluster_idxs = cluster_idxs_tensor.data<int>();
    int *cluster_offsets = cluster_offsets_tensor.data<int>();

    fill_cluster_idxs_(CCs, cluster_idxs, cluster_offsets);
}

//------------------------------------API------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){

    m.def("ballquery_batch_p", &ballquery_batch_p, "ballquery_batch_p");
    m.def("bfs_cluster", &bfs_cluster, "bfs_cluster");

}
