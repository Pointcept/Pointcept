_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0.8
empty_cache = False
enable_amp = True
find_unused_parameters = True

# trainer
train = dict(
    type="MultiDatasetTrainer",
)

# model settings
model = dict(
    type="PPT-v1m1",
    backbone=dict(
        type="SpUNet-v1m3",
        in_channels=4,
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
        conditions=("SemanticKITTI", "nuScenes", "Waymo"),
        zero_init=False,
        norm_decouple=True,
        norm_adaptive=False,
        norm_affine=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=96,
    context_channels=256,
    conditions=("SemanticKITTI", "nuScenes", "Waymo"),
    template="[x]",
    clip_model="ViT-B/16",
    # fmt: off
    class_name=(
        # SemanticKITTI
        "car", "bicycle", "motorcycle", "truck", "other vehicle",
        "person", "person who rides a bicycle", "person who rides a motorcycle", "road", "parking",
        "path for pedestrians at the side of a road", "other ground", "building", "fence", "vegetation",
        "trunk", "terrain", "pole", "traffic sign",
        # nuScenes
        "barrier", "bicycle", "bus", "car", "construction vehicle",
        "motorcycle", "pedestrian", "traffic cone", "trailer", "truck",
        "path suitable or safe for driving", "other flat", "sidewalk", "terrain", "man made", "vegetation",
        # waymo
        "car", "truck", "bus", "other vehicle", "person who rides a motorcycle",
        "person who rides a bicycle", "pedestrian", "sign", "traffic light", "pole",
        "construction cone", "bicycle", "motorcycle", "building", "vegetation",
        "tree trunk", "curb", "road", "lane marker", "other ground", "horizontal surface that can not drive",
        "surface when pedestrians most likely to walk on",
    ),
    valid_index=(
        [i for i in range(19)],
        [i for i in range(19, 19 + 16)],
        [i for i in range(19 + 16, 19 + 16 + 22)],
    ),
    # fmt: on
    backbone_mode=False,
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
# param_dicts = [dict(keyword="modulation", lr=0.0002)]

# dataset settings
data = dict(
    num_classes=19,
    ignore_index=-1,
    names=[
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other-vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other-ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic-sign",
    ],
    train=dict(
        type="ConcatDataset",
        datasets=[
            # nuScenes
            dict(
                type="NuScenesDataset",
                split="train",
                data_root="data/nuscenes",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='x', p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis='y', p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "nuScenes"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
            # SemanticKITTI
            dict(
                type="SemanticKITTIDataset",
                split="train",
                data_root="data/semantic_kitti",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "SemanticKITTI"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
            # Waymo
            dict(
                type="WaymoDataset",
                split="training",
                data_root="data/waymo",
                transform=[
                    # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
                    # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
                    dict(
                        type="RandomRotate",
                        angle=[-1, 1],
                        axis="z",
                        center=[0, 0, 0],
                        p=0.5,
                    ),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
                    # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="y", p=0.5),
                    dict(
                        type="PointClip",
                        point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2),
                    ),
                    dict(type="RandomScale", scale=[0.9, 1.1]),
                    # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
                    dict(type="RandomFlip", p=0.5),
                    dict(type="RandomJitter", sigma=0.005, clip=0.02),
                    # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
                    dict(
                        type="GridSample",
                        grid_size=0.05,
                        hash_type="fnv",
                        mode="train",
                        return_grid_coord=True,
                    ),
                    # dict(type="SphereCrop", point_max=1000000, mode="random"),
                    # dict(type="CenterShift", apply_z=False),
                    dict(type="Update", keys_dict={"condition": "Waymo"}),
                    dict(type="ToTensor"),
                    dict(
                        type="Collect",
                        keys=("coord", "grid_coord", "segment", "condition"),
                        feat_keys=("coord", "strength"),
                    ),
                ],
                test_mode=False,
                ignore_index=-1,
                loop=1,
            ),
        ],
    ),
    val=dict(
        type="SemanticKITTIDataset",
        split="val",
        data_root="data/semantic_kitti",
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="Update", keys_dict={"condition": "SemanticKITTI"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "origin_segment",
                    "condition",
                    "inverse",
                ),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
    ),
    test=dict(
        type="SemanticKITTIDataset",
        split="val",
        data_root="data/semantic_kitti",
        transform=[
            dict(type="PointClip", point_cloud_range=(-35.2, -35.2, -4, 35.2, 35.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="Update", keys_dict={"condition": "SemanticKITTI"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=-1,
    ),
)
