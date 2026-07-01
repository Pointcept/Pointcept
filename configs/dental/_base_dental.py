"""
Bits2Bites — shared configuration for multi-task occlusal classification.

One intra-oral scan (mesh point cloud + per-tooth landmark one-hot) is mapped to
five clinical occlusal labels by a single backbone with five classification
heads (MTL). Backbone-specific and single-task (STL) configs inherit from this
file; see ``cls-ptv3-base.py`` / ``cls-spunet-base.py``.

Task order (``num_classes_list``):
    0 right sagittal (3)  1 left sagittal (3)  2 anterior bite (4)
    3 transverse (3)      4 midline (2)
"""

_base_ = ["../_base_/default_runtime.py"]

# ---- training knobs ----
batch_size = 8
batch_size_val = 8
batch_size_test = 8
epoch = 200
eval_epoch = 200
num_worker = 4
empty_cache = False
enable_amp = True
clip_grad = 1.0
enable_wandb = True  # see BITS2BITES.md "Weights & Biases (wandb) setup"
# __import__ instead of a module-level `import os`: Config deepcopies every
# top-level name when merging base configs, and can't deepcopy a module object.
wandb_project = __import__("os").environ.get("WANDB_PROJECT", "bits2bites")
fold_val = 1  # held-out fold (1..5); consumed by tools/dental_fold.py

# ---- data ----
dataset_type = "DentalDataset"
data_root = "data/dental_landmarks_mesh"
num_classes_list = [3, 3, 4, 3, 2]

# 3 xyz + 6-D per-tooth landmark one-hot = 9-D input features
feat_keys = ["coord", "point_label_onehot"]
label_keys = ("label_0", "label_1", "label_2", "label_3", "label_4")

# per-class CrossEntropy weights (see tools/ToothFairy4M_class_weights.py)
class_weights = [
    [0.7185, 0.7791, 3.0794],  # right sagittal
    [0.7386, 0.8228, 2.3214],  # left sagittal
    [0.6173, 0.6849, 1.1905, 12.5000],  # anterior bite
    [0.4762, 16.6667, 1.1905],  # transverse
    [1.2821, 0.8197],  # midline
]

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomShift", shift=((-0.02, 0.02),) * 3),
            dict(
                type="RandomRotate",
                angle=[-0.1, 0.1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(
                type="RandomDropout",
                dropout_ratio=0.5,
                dropout_application_ratio=0.5,
            ),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", *label_keys),
                feat_keys=feat_keys,
            ),
        ],
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=False,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", *label_keys),
                feat_keys=feat_keys,
            ),
        ],
    ),
    # Standalone testing evaluates the held-out (val) fold with labels via
    # MultiClsTester. Point data_root at unlabelled scans for pure inference.
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        test_mode=False,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "name", *label_keys),
                feat_keys=feat_keys,
            ),
        ],
    ),
)

# ---- model (backbone + backbone_embed_dim set by the backbone config) ----
model = dict(
    type="MultiTaskClassifier",
    num_classes_list=num_classes_list,
    class_weights=class_weights,
    loss_type="ce",
)

hooks = [
    dict(type="IterationTimer"),
    dict(type="InformationWriter"),
    dict(type="MultiClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
]

test = dict(type="MultiClsTester")
