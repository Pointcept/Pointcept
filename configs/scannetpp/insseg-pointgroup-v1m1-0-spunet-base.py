_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# misc custom setting
batch_size = 1  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True

wandb_project_name = "pointcept-prs"
wandb_tags = ["PointGroup"]
enable_wandb = False
use_step_logging = True
log_every = 500
save_freq = 5


num_classes = 100
segment_ignore_index = (-1, 0, 1, 2, 16, 19, 20, 24, 26, 33, 36, 48, 53, 63, 64, 73, 74)

# channels = (32, 64, 128, 256, 256, 128, 96, 96),
class_names = [
    "wall",
    "ceiling",
    "floor",
    "table",
    "door",
    "ceiling lamp",
    "cabinet",
    "blinds",
    "curtain",
    "chair",
    "storage cabinet",
    "office chair",
    "bookshelf",
    "whiteboard",
    "window",
    "box",
    "window frame",
    "monitor",
    "shelf",
    "doorframe",
    "pipe",
    "heater",
    "kitchen cabinet",
    "sofa",
    "windowsill",
    "bed",
    "shower wall",
    "trash can",
    "book",
    "plant",
    "blanket",
    "tv",
    "computer tower",
    "kitchen counter",
    "refrigerator",
    "jacket",
    "electrical duct",
    "sink",
    "bag",
    "picture",
    "pillow",
    "towel",
    "suitcase",
    "backpack",
    "crate",
    "keyboard",
    "rack",
    "toilet",
    "paper",
    "printer",
    "poster",
    "painting",
    "microwave",
    "board",
    "shoes",
    "socket",
    "bottle",
    "bucket",
    "cushion",
    "basket",
    "shoe rack",
    "telephone",
    "file folder",
    "cloth",
    "blind rail",
    "laptop",
    "plant pot",
    "exhaust fan",
    "cup",
    "coat hanger",
    "light switch",
    "speaker",
    "table lamp",
    "air vent",
    "clothes hanger",
    "kettle",
    "smoke detector",
    "container",
    "power strip",
    "slippers",
    "paper bag",
    "mouse",
    "cutting board",
    "toilet paper",
    "paper towel",
    "pot",
    "clock",
    "pan",
    "tap",
    "jar",
    "soap dispenser",
    "binder",
    "bowl",
    "tissue box",
    "whiteboard eraser",
    "toilet brush",
    "spray bottle",
    "headphones",
    "stapler",
    "marker",
]
# model settings
#    channels=(32, 32, 32, 64, 64, 32, 64, 64)
# channels = (32, 64, 128, 256, 256, 128, 96, 96),
model = dict(
    type="PG-v1m1",
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=6,
        num_classes=0,
        channels=(32, 32, 32, 64, 64, 32, 64, 64),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    backbone_out_channels=64,
    semantic_num_classes=num_classes,
    semantic_ignore_index=-1,
    segment_ignore_index=segment_ignore_index,
    instance_ignore_index=-1,
    cluster_thresh=1.5,
    cluster_closed_points=300,
    cluster_propose_points=100,
    cluster_min_points=50,
)

# scheduler settings
epoch = 800
optimizer = dict(type="SGD", lr=0.1, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="PolyLR")

# dataset settings
dataset_type = "ScanNetPPDataset"
# data_root = "data/scannetpp"
data_root = "./raw_dataset/scannetpp_v2_sgi"

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.5
            ),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis='z', p=0.75),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.1),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            # dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            # dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.8, mode="random"),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("color", "normal"),
            ),
        ],
        test_mode=False,
    ),
    trainEval=dict(
        type="ScanNetPPTrainEvalSample",
        split="train",
        data_root=data_root,
        nsamples=12,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                    "origin_coord": "grid_coord",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                    "origin_coord": "grid_coord",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type="ScanNetPPTestInstanceSegmentation",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="Copy",
                keys_dict={
                    "coord": "origin_coord",
                    "segment": "origin_segment",
                    "instance": "origin_instance",
                    "origin_coord": "grid_coord",
                },
            ),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=1000000, mode='center'),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(
                type="InstanceParser",
                segment_ignore_index=segment_ignore_index,
                instance_ignore_index=-1,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "segment",
                    "instance",
                    "origin_coord",
                    "origin_segment",
                    "origin_instance",
                    "instance_centroid",
                    "bbox",
                ),
                feat_keys=("color", "normal"),
                offset_keys_dict=dict(offset="coord", origin_offset="origin_coord"),
            ),
        ],
        test_mode=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="InsSegEvaluator",
        segment_ignore_index=segment_ignore_index,
        instance_ignore_index=-1,
        write_cls_ap=True,
        use_eval_train=True,
    ),
    dict(type="CheckpointSaver", save_freq=save_freq),
]


# Tester
test = dict(type="InstanceSegTest", verbose=True)
