#include "datatype.h"

template <Int dimension> SparseGrid<dimension>::SparseGrid() : ctr(0) {
    // Sparsehash needs a key to be set aside and never used
    Point<dimension> empty_key;
    for(Int i = 0; i < dimension; i++){
        empty_key[i] = std::numeric_limits<Int>::min();
    }
    mp.set_empty_key(empty_key);
}

ConnectedComponent::ConnectedComponent(){}

void ConnectedComponent::addPoint(Int pt_idx){
	pt_idxs.push_back(pt_idx);
}
