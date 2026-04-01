_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/partnete.py",
]

# misc custom setting
batch_size = 64  # bs: total bs in all gpus
num_worker = 96
mix_prob = 0.8
clip_grad = 3.0
empty_cache = False
enable_amp = True

test = dict(type="PartNetEPartSegTester", verbose=True)
num_parts = [
    4,
    4,
    2,
    3,
    4,
    4,
    4,
    6,
    3,
    3,
    2,
    2,
    6,
    3,
    2,
    4,
    7,
    3,
    3,
    3,
    3,
    2,
    2,
    3,
    3,
    2,
    3,
    2,
    5,
    3,
    2,
    4,
    4,
    4,
    2,
    3,
    3,
    3,
    4,
    2,
    5,
    2,
    3,
    5,
    2,
]

class_names = [
    "Scissors",
    "Lighter",
    "Box",
    "Camera",
    "StorageFurniture",
    "Safe",
    "Toilet",
    "Chair",
    "Oven",
    "USB",
    "Remote",
    "Switch",
    "Laptop",
    "Phone",
    "Bottle",
    "Mouse",
    "Table",
    "Keyboard",
    "Eyeglasses",
    "Faucet",
    "KitchenPot",
    "Knife",
    "Window",
    "Pen",
    "WashingMachine",
    "Clock",
    "Refrigerator",
    "Pliers",
    "Microwave",
    "Toaster",
    "Printer",
    "Kettle",
    "TrashCan",
    "Door",
    "Cart",
    "Dishwasher",
    "Suitcase",
    "Dispenser",
    "Display",
    "Bucket",
    "Lamp",
    "Globe",
    "Stapler",
    "CoffeeMachine",
    "FoldingChair",
]

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=sum(num_parts),
    backbone_out_channels=54,
    backbone=dict(
        type="PT-v3m3",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(54, 108, 216, 432, 576),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        enc_mode=False,
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
    freeze_backbone=False,
)

# scheduler settings
epoch = 800
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
dataset_type = "PartNetEDataset"
data_root = "data/partnete"
meta_path = "pointcept/datasets/preprocessing/partnete/meta_info.json"
data = dict(
    num_classes=sum(num_parts),
    ignore_index=-1,
    train=dict(
        type=dataset_type,
        split="few_shot",
        data_root=data_root,
        class_names=class_names,
        num_parts=num_parts,
        meta_path=meta_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=1.0
            ),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            # dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            # dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            # dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            # dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "cls_token"),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        num_parts=num_parts,
        meta_path=meta_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "inverse",
                    "cls_token",
                ),
                feat_keys=("coord", "color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        num_parts=num_parts,
        meta_path=meta_path,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
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
    dict(type="PartNetEPartSegEvaluator", num_parts=num_parts, write_part_iou=False),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]
