"""Bits2Bites — v2 dataset (Bits2Bites + Bits2Bites2 merged, ~1150 patients).

Overrides on top of ``../dental/_base_dental.py``: class weights recomputed
against the v2 label distribution (see tools/ToothFairy4M_class_weights.py,
run over data/dental_landmarks_only_v2/labels.csv) and a shorter epoch
budget. Kept separate from configs/dental/ so the published fold1/mesh_only
runs there stay reproducible. Point ``-e`` at a ``*_v2`` data root when
training against this base.
"""

_base_ = ["../dental/_base_dental.py"]

epoch = 100
eval_epoch = 100

# per-class CrossEntropy weights, recomputed over data/dental_landmarks_only_v2
class_weights = [
    [0.6941, 0.7503, 4.4127],  # right sagittal
    [0.7241, 0.7241, 4.2045],  # left sagittal
    [0.5909, 0.6122, 1.6730, 13.0795],  # anterior bite
    [0.4748, 9.5917, 1.2662],  # transverse
    [1.6427, 0.7188],  # midline
]

model = dict(class_weights=class_weights)
