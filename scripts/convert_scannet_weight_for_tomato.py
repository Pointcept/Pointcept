"""
将 ScanNet 预训练的 PTv3 权重转换为 Tomato 数据集可用的格式

主要处理：
1. 删除 seg_head（分类头）- 类别数不匹配 (20 -> 3)
2. 删除 embedding.stem.conv（输入层）- 输入通道不匹配 (6 -> 3)

用法:
    python scripts/convert_scannet_weight_for_tomato.py \
        --input model_best.pth \
        --output model_best_for_tomato.pth
"""

import argparse
import torch


def convert_weight(input_path, output_path):
    print(f"Loading checkpoint from: {input_path}")
    ckpt = torch.load(input_path, map_location="cpu")

    state_dict = ckpt["state_dict"]
    print(f"Original state_dict has {len(state_dict)} keys")

    # 需要删除的层
    keys_to_remove = []

    for key in state_dict.keys():
        # 删除分类头 (num_classes: 20 -> 3)
        if "seg_head" in key:
            keys_to_remove.append(key)
            print(f"  Remove (seg_head): {key} -> {state_dict[key].shape}")

        # 删除输入嵌入层 (in_channels: 6 -> 3)
        if "embedding.stem.conv" in key:
            keys_to_remove.append(key)
            print(f"  Remove (embedding): {key} -> {state_dict[key].shape}")

    # 删除不兼容的层
    for key in keys_to_remove:
        del state_dict[key]

    print(f"\nRemoved {len(keys_to_remove)} keys")
    print(f"New state_dict has {len(state_dict)} keys")

    # 保存新的权重
    new_ckpt = {
        "epoch": 0,
        "state_dict": state_dict,
        "best_metric_value": 0,
    }

    torch.save(new_ckpt, output_path)
    print(f"\nSaved converted weight to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="model_best.pth",
                        help="Input checkpoint path")
    parser.add_argument("--output", type=str, default="model_best_for_tomato.pth",
                        help="Output checkpoint path")
    args = parser.parse_args()

    convert_weight(args.input, args.output)
