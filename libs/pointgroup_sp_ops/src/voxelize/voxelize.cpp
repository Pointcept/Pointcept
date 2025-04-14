/*
Points to Voxels & Voxels to Points (Modified from SparseConv)
Written by Li Jiang
All Rights Reserved 2020.
*/

#include "voxelize.h"

/* ================================== voxelize_idx ================================== */
template <Int dimension>
void voxelize_idx(/* long N*4 */ at::Tensor coords, /* long M*4 */ at::Tensor output_coords,
                  /* Int N */ at::Tensor input_map, /* Int M*(maxActive+1) */ at::Tensor output_map, Int batchSize, Int mode){
    assert(coords.ndimension() == 2);
    assert(coords.size(1) >= dimension and coords.size(1) <= dimension + 1);

    RuleBook voxelizeRuleBook;  // rule[1]: M voxels -> N points  output_map
    SparseGrids<dimension> inputSGs; // voxel_coords -> voxel_idx in M voxels      input_map: N points -> M voxels
    Int nActive = 0;

    Int maxActive = voxelize_inputmap<dimension>(inputSGs, input_map.data<Int>(), voxelizeRuleBook, nActive, coords.data<long>(), coords.size(0), coords.size(1), batchSize, mode);

    output_map.resize_({nActive, maxActive + 1});
    output_map.zero_();

    output_coords.resize_({nActive, coords.size(1)});
    output_coords.zero_();

    Int *oM = output_map.data<Int>();
    long *oC = output_coords.data<long>();
    voxelize_outputmap<dimension>(coords.data<long>(), oC, oM, &voxelizeRuleBook[1][0], nActive, maxActive);
}


template <Int dimension>
void voxelize_outputmap(long *coords, long *output_coords, Int *output_map, Int *rule, Int nOutputRows, Int maxActive){
    for(Int i = 0; i < nOutputRows; i++){
        for(Int j = 0; j <= maxActive; j++)
            output_map[j] = rule[j];
        Int inputIdx = rule[1];
        rule += (1 + maxActive);
        output_map += (1 + maxActive);

        long *coord = coords + inputIdx * (dimension + 1);
        long *output_coord  = output_coords + i * (dimension + 1);
        for(Int j = 0; j <= dimension; j++){
            output_coord[j] = coord[j];
        }
    }
}

//mode 0=guaranteed unique 1=last item(overwrite) 2=first item(keep) 3=sum, 4=mean
//input: coords
//output: SGs: one map for each batch: map from voxel_coord to voxel_idx(in M voxels)
//output: input_map: N, N points -> M voxels
//output: rules
//output: nActive
//output: maxActive
template <Int dimension>
Int voxelize_inputmap(SparseGrids<dimension> &SGs, Int *input_map, RuleBook &rules, Int &nActive, long *coords, Int nInputRows, Int nInputColumns, Int batchSize, Int mode){
    assert(nActive == 0);
    assert(rules.size() == 0);
    assert(SGs.size() == 0);

    SGs.resize(batchSize);
    Point<dimension> p;

    std::vector<std::vector<Int>> outputRows;
    if(nInputColumns == dimension){
        SGs.resize(1);
        auto &sg = SGs[0];
        for(Int i = 0; i < nInputRows; i++){
            for(Int j = 0; j < dimension; j++)
                p[j] = coords[j];
            coords += dimension;
            auto iter = sg.mp.find(p);
            if (iter == sg.mp.end()){
                sg.mp[p] = nActive++;
                outputRows.resize(nActive);
            }
            outputRows[sg.mp[p]].push_back(i);

            input_map[i] = sg.mp[p];
        }
    }
    else{  // nInputColumns == dimension + 1 (1 in index 0 for batchidx)
        Int batchIdx;
        for(Int i = 0; i < nInputRows; i++){
            batchIdx = coords[0];
            for(Int j = 0; j < dimension; j++)
                p[j] = coords[j + 1];
            coords += (dimension + 1);
            if(batchIdx + 1 >= (Int)SGs.size()){
                SGs.resize(batchIdx + 1);
            }
            auto &sg = SGs[batchIdx];
            auto iter = sg.mp.find(p);
            if(iter == sg.mp.end()){
                sg.mp[p] = nActive++;
                outputRows.resize(nActive);
            }
            outputRows[sg.mp[p]].push_back(i);

            input_map[i] = sg.mp[p];
        }
    }

    // Rulebook Format
    // rules[0][0] == mode
    // rules[0][1] == maxActive per spatial location (==1 for modes 0,1,2)
    // rules[0][2] == nInputRows
    // rules[0][3] == nOutputRows
    // rules[1]   nOutputRows x (1+maxActive)
    rules.resize(2);
    rules[0].push_back(mode);
    rules[0].push_back(1);
    rules[0].push_back(nInputRows);
    rules[0].push_back(outputRows.size());
    auto &rule = rules[1];
    if(mode == 0){
        assert(nInputRows == (Int)outputRows.size());
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            assert((Int)outputRows[i].size() == 1);
            rule.push_back(outputRows[i][0]);
        }
    }
    if(mode == 1){
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            rule.push_back(outputRows[i].front());
        }
    }
    if(mode == 2){
        for(Int i = 0; i < nActive; i++){
            rule.push_back(1);
            rule.push_back(outputRows[i].back());
        }
    }
    Int maxActive = 1;
    if(mode == 3 or mode == 4){
        for(auto &row: outputRows)
            maxActive = std::max(maxActive, (Int)row.size());
        rules[0][1] = maxActive;
        for(auto &row: outputRows){
            rule.push_back(row.size());
            for(auto &r: row)
                rule.push_back(r);
            rule.resize((rule.size() + maxActive) / (maxActive + 1) * (maxActive + 1));
        }
    }
    return maxActive;
}


/* ================================== voxelize ================================== */
template <typename T>
void voxelize_fp(/* cuda float N*C */ at::Tensor feats, // N * 3 -> M * 3 (N >= M)
              /* cuda float M*C */ at::Tensor output_feats,
              /* cuda Int M*(maxActive+1) */ at::Tensor output_map, Int mode, Int nActive, Int maxActive, Int nPlane){

    auto iF = feats.data<T>();
    auto oF = output_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_fp_cuda<T>(nActive, maxActive, nPlane, iF, oF, rules, mode==4);
}

template <typename T>
void voxelize_bp(/* cuda float M*C */ at::Tensor d_output_feats, /* cuda float N*C */ at::Tensor d_feats, /* cuda Int M*(maxActive+1) */ at::Tensor output_map,
            Int mode, Int nActive, Int maxActive, Int nPlane){
    auto d_oF = d_output_feats.data<T>();
    auto d_iF = d_feats.data<T>();

    Int *rules = output_map.data<Int>();

    voxelize_bp_cuda<T>(nActive, maxActive, nPlane, d_oF, d_iF, rules, mode==4);
}

/* ================================== point_recover ================================== */
template <typename T>
void point_recover_fp(/* cuda float M*C */ at::Tensor feats, /* cuda float N*C */ at::Tensor output_feats, /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    auto iF = feats.data<T>();
    auto oF = output_feats.data<T>();

    Int *rules = idx_map.data<Int>();

    voxelize_bp_cuda<T>(nActive, maxActive, nPlane, iF, oF, rules, false);
}


template <typename T>
void point_recover_bp(/* cuda float N*C */ at::Tensor d_output_feats, /* cuda float M*C */ at::Tensor d_feats,  /* cuda Int M*(maxActive+1) */ at::Tensor idx_map,
                Int nActive, Int maxActive, Int nPlane){
    auto d_oF = d_output_feats.data<T>();
    auto d_iF = d_feats.data<T>();

    Int *rules = idx_map.data<Int>();

    voxelize_fp_cuda<T>(nActive, maxActive, nPlane, d_oF, d_iF, rules, false);
}
