_base_ = ["../_base_/default_runtime.py"]

# =========================
# 3DP cluster classification
# =========================

# ---- runtime ----
save_path = "exp/3dp/cls-ptv3"
batch_size = 32
batch_size_val = 16
num_worker = 8
empty_cache = False
enable_amp = True

# ---- classes (EDIT THIS) ----
# Put your 3DP classes here (string labels used in labels_*.csv)
class_names =  [
    "marche",
    "accroupi",
    "escalade",
]

# ---- model ----
# For clusters, XYZ only is often enough to start.
# If you add intensity or other features, change:
#   - data Collect feat_keys
#   - backbone.in_channels
model = dict(
    type="DefaultClassifier",
    num_classes=len(class_names),
    backbone_embed_dim=512,
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,  # XYZ
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.2,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# ---- optimizer / scheduler ----
epoch = 200
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# ---- dataset ----
dataset_type = "ThreeDPClusterDataset"
data_root = "data/3dp_clusters"

# number of points per cluster used by the Dataset sampler (before transforms)
num_points = 2048

data = dict(
    num_classes=len(class_names),
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        num_points=num_points,
        use_intensity=False,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        class_names=class_names,
        num_points=num_points,
        use_intensity=False,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord"],
            ),
        ],
        test_mode=False,
    ),
)

# ---- hooks / tester (classification) ----
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
  #  dict(type="PreciseEvaluator", test_last=False),
]
test = dict(type="ClsTester", verbose=True)

# ---- inference helper settings (used by the provided 3DP service script) ----
infer = dict(
    transform=[
        dict(type="NormalizeCoord"),
        dict(
            type="GridSample",
            grid_size=0.02,
            hash_type="fnv",
            mode="train",
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord"),
            feat_keys=["coord"],
        ),
    ],
)
