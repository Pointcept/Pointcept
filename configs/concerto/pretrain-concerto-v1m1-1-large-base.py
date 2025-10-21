_base_ = ["../_base_/default_runtime.py"]
seed = None

crop_h = 518
crop_w = 518
patch_size = 14
# misc custom setting
batch_size = 96  # bs: total bs in all gpus
num_worker = 320
mix_prob = 0
clip_grad = 2.0

empty_cache = True
enable_amp = True
amp_dtype = "bfloat16"
evaluate = False
find_unused_parameters = True

# model settings
model = dict(
    type="Concerto-v1m1",
    patch_h=crop_h // patch_size,
    patch_w=crop_w // patch_size,
    image_weight_name="dinov2_vitg14_reg",
    image_weight_path="facebook/dinov2-with-registers-giant",
    backbone_out_channels=1664,
    embedding_channels=64,
    student_pretrained=False,
    enc2d_upcast_level=3,
    # backbone - student & teacher
    backbone=dict(
        type="PT-v3m2",
        in_channels=9,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(64, 128, 256, 512, 768),
        enc_num_head=(4, 8, 16, 32, 48),
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
        enc_mode=True,
        traceable=True,
        mask_token=True,
    ),
    teacher_custom=dict(
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
    ),
    head_in_channels=1536,
    head_hidden_channels=4096,
    head_embed_channels=256,
    head_num_prototypes=4096,
    enc2d_head_in_channels=1536,
    enc2d_head_hidden_channels=4096,
    enc2d_head_embed_channels=256,
    enc2d_head_num_prototypes=4096,
    num_global_view=2,
    num_local_view=4,
    mask_size_start=0.1,
    mask_size_base=0.4,
    mask_size_warmup_ratio=0.05,
    mask_ratio_start=0.3,
    mask_ratio_base=0.7,
    mask_ratio_warmup_ratio=0.05,
    mask_jitter=0.01,
    teacher_temp_start=0.04,
    teacher_temp_base=0.07,
    teacher_temp_warmup_ratio=0.05,
    student_temp=0.1,
    mask_loss_weight=1 / 8,
    roll_mask_loss_weight=1 / 8,
    unmask_loss_weight=2 / 8,
    enc2d_loss_weight=4 / 8,
    momentum_base=0.994,
    momentum_final=1,
    match_max_k=8,
    match_max_r=0.32,
    up_cast_level=2,
    enc2d_cos_shift=True,
    sonata_model_type="online",
)

# scheduler settings
epoch = 100
base_lr = 0.004
lr_decay = 0.9  # layer-wise lr decay

base_wd = 0.04  # wd scheduler enable in hooks
final_wd = 0.2  # wd scheduler enable in hooks

dec_depths = model["backbone"]["enc_depths"]
param_dicts = [
    dict(
        keyword=f"enc{e}.block{b}.",
        lr=base_lr * lr_decay ** (sum(dec_depths) - sum(dec_depths[:e]) - b - 1),
    )
    for e in range(len(dec_depths))
    for b in range(dec_depths[e])
]
del dec_depths

optimizer = dict(type="AdamW", lr=base_lr, weight_decay=base_wd)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[base_lr] + [g["lr"] for g in param_dicts],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# dataset settings
transform = [
    dict(
        type="Update",
        keys_dict={
            "index_valid_keys": (
                "coord",
                "color",
                "normal",
                "superpoint",
                "strength",
                "segment",
                "instance",
                "correspondence",
                "global_correspondence",
            )
        },
    ),
    dict(
        type="ImgAugmentation",
        crop_h=crop_h,
        crop_w=crop_w,
        patch_h=crop_h // patch_size,
        patch_w=crop_w // patch_size,
        patch_size=patch_size,
        imgtransforms=[
            dict(type="ImgChromaticJitter", p=0.95, std=0.05),
            dict(type="ImgGaussianBlur", p=0.5),
            dict(
                type="Imgnormalize",
                mean=IMAGENET_DEFAULT_MEAN,
                std=IMAGENET_DEFAULT_STD,
            ),
        ],
    ),
    dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train"),
    dict(type="RandomDropColor", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropColor", drop_ratio=0.2, drop_application_ratio=0.5),
    dict(type="RandomDropNormal", drop_ratio=1.0, drop_application_ratio=0.2),
    dict(type="RandomDropNormal", drop_ratio=0.2, drop_application_ratio=0.5),
    dict(type="Copy", keys_dict={"coord": "origin_coord"}),
    dict(
        type="MultiViewGenerator",
        global_view_num=2,
        global_view_scale=(0.4, 1.0),
        local_view_num=4,
        local_view_scale=(0.1, 0.4),
        global_shared_transform=[
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        global_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
        ],
        local_transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.8),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.8),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(
                type="RandomColorJitter",
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.02,
                p=0.8,
            ),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="NormalizeColor"),
        ],
        max_size=65536,
        enc2d_max_size=65536,
        enc2d_scale=(0.8, 1),
    ),
    dict(type="ToTensor"),
    dict(type="Update", keys_dict={"grid_size": 0.02}),
    dict(
        type="Collect",
        keys=(
            "global_origin_coord",
            "global_coord",
            "global_color",
            "global_offset",
            "local_origin_coord",
            "local_coord",
            "local_color",
            "local_offset",
            "grid_size",
            "name",
            "images",
            "global_correspondence",
            "img_num",
        ),
        offset_keys_dict=dict(),
        global_feat_keys=("global_coord", "global_color", "global_normal"),
        local_feat_keys=("local_coord", "local_color", "local_normal"),
    ),
]

data = dict(
    train=dict(
        type="ConcatDataset",
        datasets=[
            # ARKit
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["Training", "Validation"],
                data_root="data/arkitscenes",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # ScanNet
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val", "test"],
                data_root="data/scannet",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # ScanNet++
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=[
                    "train",
                    "val",
                    "test",
                ],
                data_root="data/scannetpp",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # S3DIS
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
                data_root="data/s3dis",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # HM3D
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val"],
                data_root="data/hm3d",
                transform=transform,
                test_mode=False,
                # force_label=False,
                loop=1,
            ),
            # Structured3D
            dict(
                type="DefaultImagePointDataset",
                crop_h=crop_h,
                crop_w=crop_w,
                patch_size=patch_size,
                split=["train", "val"],
                data_root="data/structured3d",
                transform=transform,
                test_mode=False,
                loop=1,
            ),
            # # RE10K
            # dict(
            #     type="DefaultImagePointDataset",
            #     crop_h = crop_h,
            #     crop_w = crop_w,
            #     patch_size = patch_size,
            #     split=["train", "test"],
            #     data_root="data/re10k",
            #     transform=transform,
            #     test_mode=False,
            #     loop=1,
            # ),
        ],
    )
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="WeightDecaySchedular", base_value=base_wd, final_value=final_wd),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=10),
]
