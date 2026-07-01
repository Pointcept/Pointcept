"""Bits2Bites — SpUNet backbone, multi-task (MTL) occlusal classification."""

_base_ = ["_base_dental.py"]

model = dict(
    backbone_embed_dim=256,  # SpUNet encoder-mode output width
    backbone=dict(
        type="SpUNet-v1m1",
        in_channels=9,  # 3 xyz + 6 landmark one-hot
        num_classes=0,  # feature extractor
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=True,  # encoder-only + global mean pool (was cls_mode pre-v1.7.0)
    ),
)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
scheduler = dict(type="MultiStepLR", milestones=[0.6, 0.8], gamma=0.1)
