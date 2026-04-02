_base_ = ["../_base_/default_runtime.py"]
# misc custom setting
batch_size = 64  # bs: total bs in all gpus
num_worker = 112
batch_size_val = 8
empty_cache = False
enable_amp = False
find_unused_parameters = True

# model settings
model = dict(
    type="DefaultClassifier",
    num_classes=40,
    backbone_embed_dim=576,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=1.1,
        rescale_coords=1.2,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    freeze_backbone=True,
)

# scheduler settings
epoch = 300
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

# dataset settings
dataset_type = "ScanObjectNNHardestDataset"
data_root = "data/scanobjectnn_eval"
cache_data = False
class_names = [
    "bag",
    "bin",
    "box",
    "cabinet",
    "chair",
    "desk",
    "display",
    "door",
    "shelf",
    "table",
    "bed",
    "pillow",
    "sink",
    "sofa",
    "toilet",
]

data = dict(
    num_classes=15,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        if_color=True,
        if_normal=True,
        class_names=class_names,
        transform=[
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1/2, -1/2], axis="x", p=1.0),
            dict(type="RandomScale", scale=[0.9, 1.1], anisotropic=True),
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
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "color", "normal"],
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        if_color=True,
        if_normal=True,
        class_names=class_names,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="RandomRotate", angle=[-1/2, -1/2], axis="x", p=1.0),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "color", "normal"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        if_color=True,
        if_normal=True,
        class_names=class_names,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                # dict(type="RandomRotate", angle=[-1/2, -1/2], axis="x", p=1.0),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "color", "normal"],
                ),
            ],
            aug_transform=[
                [{"type": "RandomScale", "scale": [1, 1], "anisotropic": True}],
                [
                    {"type": "RandomScale", "scale": [1.0, 1.0], "anisotropic": True},
                    {
                        "type": "RandomFlip",
                        "p": 0.5,
                    },
                ],
                [{"type": "RandomScale", "scale": [0.8, 0.8], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [0.85, 0.85], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [0.9, 0.9], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [0.95, 0.95], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [1.05, 1.05], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [1.1, 1.1], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [1.15, 1.15], "anisotropic": True}],
                [{"type": "RandomScale", "scale": [1.2, 1.2], "anisotropic": True}],
            ],
        ),
    ),
)


# hooks
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="ClsVotingTester", num_repeat=1)
