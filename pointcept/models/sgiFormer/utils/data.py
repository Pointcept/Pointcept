"""
# This file includes code adapted from SGIFormer:
# https://github.com/RayYoh/SGIFormer
# Original author: Lei Yao (rayyohhust@gmail.com)
"""

import torch


def split_offset(value, offset):
    ret = []
    st = 0
    for end in offset:
        ret.append(value[st:end])
        st = end
    return ret


def process_label(labels, segment_ignore_index=(-1, 0, 1), semantic_ignore_index=-1):
    for _label in sorted(segment_ignore_index)[::-1]:
        if _label == semantic_ignore_index:
            continue
        labels[labels >= _label] -= 1
    return labels


def process_instance(
    instance, segment, segment_ignore_index=(-1, 0, 1), instance_ignore_index=-1
):
    mask = torch.ones_like(instance).bool()
    for _label in segment_ignore_index:
        mask[segment == _label] = False
    instance[~mask] = instance_ignore_index
    _, inverse = torch.unique(instance[mask], return_inverse=True)
    instance[mask] = inverse
    return instance
