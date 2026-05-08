_base_ = ["../_base_/default_runtime.py", "../_base_/dataset/shapenet_part.py"]

batch_size = 32
mix_prob = 0.0
clip_grad = 3.0
empty_cache = False
enable_amp = True
test = dict(type="ShapeNetPartSegTester", verbose=True)
model = dict(
    type="DefaultSegmentorV2",
    num_classes=50,
    backbone_out_channels=1386,
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

epoch = 100
eval_epoch = 100

optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.02)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# dataset settings
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"

data = dict(
    num_classes=50,
    ignore_index=-1,  # dummy ignore
    train=dict(
        type=dataset_type,
        split=["train", "val"],
        if_color=True,
        data_root=data_root,
        transform=[
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='y', p=0.5),
            # dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="Voxelize", voxel_size=0.01, hash_type='fnv', mode='train'),
            # dict(type="SphereCrop", point_max=2500, mode='random'),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "cls_token"),
                # keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "color", "normal"],
            ),
        ],
        loop=2,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        if_color=True,
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            # dict(type="SphereCrop", point_max=2500, mode='center'),
            dict(type="CenterShift", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "cls_token",
                    "origin_segment",
                    "inverse",
                ),
                # keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "color", "normal"],
            ),
        ],
        loop=1,
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        if_color=True,
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
        ],
        loop=1,
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "cls_token"),
                    feat_keys=("coord", "color", "normal"),
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


# hook
hooks = [
    dict(
        type="CheckpointLoader",
        keywords="module.student.backbone",
        replacement="module.backbone",
    ),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ShapeNetPartSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
