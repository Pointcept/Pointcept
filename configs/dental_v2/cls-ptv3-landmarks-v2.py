"""Bits2Bites — ptv3 landmarks-only variant, v2 dataset/weights/epoch budget.

Backbone/optimizer block duplicated from ../dental/cls-ptv3-base.py (the
config loader errors on duplicate keys if a file tries to multi-inherit two
configs that share a common ancestor, so this can't just list both bases).

Landmark-only samples are ~240 pts (vs ~198k for the mesh variants), so
batch size can go up substantially without exceeding GPU memory.
"""

_base_ = ["_base_dental_v2.py"]

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
scheduler = dict(type="CosineAnnealingLR", total_steps=100)

batch_size = 20
batch_size_val = 20
batch_size_test = 20
