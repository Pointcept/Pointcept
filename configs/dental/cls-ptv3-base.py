"""Bits2Bites — PointTransformerV3 backbone, multi-task (MTL) occlusal classification."""

_base_ = ["_base_dental.py"]

model = dict(
    backbone_embed_dim=128,
    backbone=dict(
        type="PT-v3m1",
        in_channels=9,  # 3 xyz + 6 landmark one-hot
        enc_channels=(16, 32, 48, 64, 128),
        enc_num_head=(1, 2, 3, 4, 8),
        dec_channels=(32, 32, 64, 96),  # unused under enc_mode, kept for reference
        dec_num_head=(2, 2, 4, 6),
        enable_flash=False,  # set True on Ampere+ GPUs
        enc_mode=True,  # encoder-only feature extractor (was cls_mode pre-v1.7.0)
    ),
)

optimizer = dict(type="AdamW", lr=0.0001, weight_decay=0.01)
scheduler = dict(type="CosineAnnealingLR", total_steps=200)
