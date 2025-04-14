# ScanNetpp Benchmark constants
# Semantic classes, 100
CLASS_LABELS_PP = (
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
)

# Instance classes, 84
INST_LABELS_PP = (
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
    "monitor",
    "shelf",
    "heater",
    "kitchen cabinet",
    "sofa",
    "bed",
    "trash can",
    "book",
    "plant",
    "blanket",
    "tv",
    "computer tower",
    "refrigerator",
    "jacket",
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
    "printer",
    "poster",
    "painting",
    "microwave",
    "shoes",
    "socket",
    "bottle",
    "bucket",
    "cushion",
    "basket",
    "shoe rack",
    "telephone",
    "file folder",
    "laptop",
    "plant pot",
    "exhaust fan",
    "cup",
    "coat hanger",
    "light switch",
    "speaker",
    "table lamp",
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
)

_base_ = [
    "../_base_/default_runtime.py",
    "../_base_/dataset/scannetpp.py",
]

# misc custom setting
batch_size = 4  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True


evaluate_interval = []

wandb_project_name = "pointcept"
wandb_tags = ["oneformer"]
enable_wandb = False
use_step_logging = True
log_every = 500
save_freq = 5

class_names = INST_LABELS_PP
class_ids = [CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP]
num_classes = len(class_names)
# segment_ignore_index = (-1, 0, 1, 2, 16, 19, 20, 24, 26,
#                         33, 36, 48, 53, 63, 64, 73, 74)
segment_ignore_index = (-1,)
semantic_num_classes = num_classes
num_channels = 32
weight = None

num_instance_classes = 84
num_semantic_classes = 100
model = dict(
    type="OneFormer3D",
    in_channels=6,
    num_channels=num_channels,
    voxel_size=0.02,
    num_classes=num_instance_classes,
    min_spatial_shape=128,
    query_thr=2,
    backbone=dict(
        type="SpUNet-OneFormer",
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True,
    ),
    decoder=dict(
        type="OneFormer-Decoder",
        num_layers=6,
        num_classes=num_instance_classes,
        num_instance_queries=400,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        in_channels=num_channels,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=True,
    ),
    criterion=dict(
        type="OneFormer-ScanNetUnifiedCriterion",
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type="OneFormer-ScanNetSemanticCriterion", ignore_index=-1, loss_weight=0.2
        ),
        inst_criterion=dict(
            type="OneFormer-InstanceCriterion",
            matcher=dict(
                type="OneFormer-HungarianMatcher",
                costs=[
                    dict(type="OneFormer-QueryClassificationCost", weight=0.5),
                    dict(type="OneFormer-MaskBCECost", weight=1.0),
                    dict(type="OneFormer-MaskDiceCost", weight=1.0),
                ],
            ),
            loss_weight=[0.5, 1.0, 1.0, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True,
        ),
    ),
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=300,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel="linear",
        stuff_classes=[-1],
    ),
)

epoch = 900
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.05)
# scheduler = dict(type="PolyLR", power=0.9)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[1e-4, 3e-4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=25,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="head", lr=3e-4)]


# dataset settings
dataset_type = "ScanNetPPSPPCDataset"
data_root = "data/scannetpp"

voxel_cfg = {"scale": 50, "spatial_shape": [128, 512], "max_npoint": 250000}
data = dict(
    # for the data, we need to load all categories
    num_classes=num_classes,
    ignore_label=-1,
    ignore_index=-1,
    names=class_names,
    ids=class_ids,
    train=dict(
        type=dataset_type,
        split="train_grid1mm_chunk6x6_stride3x3",
        data_root=data_root,
        transform=[],
        voxel_cfg=voxel_cfg,
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        voxel_cfg=voxel_cfg,
        training=False,
        with_elastic=False,
        transform=[],
        test_mode=False,
    ),
    trainEval=dict(
        type="ScanNetPPSPPCDatasetTrainSample",
        split="train",
        data_root=data_root,
        voxel_cfg=voxel_cfg,
        nsamples=25,
        training=False,
        with_elastic=False,
        transform=[],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        voxel_cfg=voxel_cfg,
        training=False,
        with_elastic=False,
        transform=[],
        test_mode=False,
    ),
)


hooks = [
    dict(type="CheckpointLoader", keywords="module.", replacement="module."),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(
        type="SPInsEvaluator",
        segment_ignore_index=segment_ignore_index,
        semantic_ignore_index=(-1,),
        instance_ignore_index=-1,
        use_eval_train=True,
        write_cls_ap=True,
    ),
    dict(type="CheckpointSaver", save_freq=save_freq),
]

# Tester
tester_evaluator = "SP"
test = dict(type="InstanceSegTest", verbose=True)
