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
num_worker = 1
mix_prob = 0
empty_cache = False
enable_amp = True
evaluate = True


evaluate_interval = []

wandb_project_name = "pointcept"
wandb_tags = ["spFormer"]
enable_wandb = False
use_step_logging = True
log_every = 500
save_freq = 5

class_names = INST_LABELS_PP
class_ids = [CLASS_LABELS_PP.index(c) for c in INST_LABELS_PP]
num_classes = len(class_names)
segment_ignore_index = (-1,)
semantic_num_classes = num_classes
num_channels = 32
weight = None

model = dict(
    type="SPFormer",
    input_channel=6,
    blocks=5,
    block_reps=2,
    media=32,
    normalize_before=True,
    return_blocks=True,
    pool="mean",
    num_class=len(class_names),
    decoder=dict(
        num_layer=6,
        num_query=400,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn="gelu",
        iter_pred=True,
        attn_mask=True,
        pe=False,
    ),
    criterion=dict(
        loss_weight=[0.5, 1.0, 1.0, 0.5],
        cost_weight=[0.5, 1.0, 1.0],
        non_object_weight=0.1,
    ),
    test_cfg=dict(
        topk_insts=300,
        score_thr=0.0,
        npoint_thr=100,
    ),
    norm_eval=False,
    fix_module=[],
)


voxel_cfg = {"scale": 50, "spatial_shape": [128, 512], "max_npoint": 250000}

epoch = 900
optimizer = dict(type="AdamW", lr=1e-5, weight_decay=0.05)
# scheduler = dict(type="PolyLR", power=0.9)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[1e-5, 3e-5],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=25.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="head", lr=3e-5)]


# dataset settings
dataset_type = "ScanNetPPSPPCDataset"
data_root = "data/scannetpp"


data = dict(
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


save_freq = 2
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
