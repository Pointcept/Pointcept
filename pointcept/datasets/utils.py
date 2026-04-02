"""
Utils for Datasets

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com), Yujia Zhang (yujia.zhang.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import random
from collections.abc import Mapping, Sequence
import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F

from torch_scatter import scatter_min
from pointcept.models.utils import offset2batch


def collate_fn(batch):
    """
    collate function for point cloud which support dict and list,
    'coord' is necessary to determine 'offset'
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgement should before Sequence
        return list(batch)
    elif isinstance(batch[0], list):
        batch = [torch.tensor(data) for data in batch]
        return torch.cat(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        if "img_num" in batch[0].keys():
            max_img_num = max([d["img_num"] for d in batch])
        batch = {
            key: (
                (
                    collate_fn([d[key] for d in batch])
                    if "offset" not in key
                    # offset -> bincount -> concat bincount-> concat offset
                    else torch.cumsum(
                        collate_fn(
                            [d[key].diff(prepend=torch.tensor([0])) for d in batch]
                        ),
                        dim=0,
                    )
                )
                if "correspondence" not in key
                else collate_fn(
                    [
                        F.pad(
                            d[key].permute(0, 2, 1),
                            (0, max_img_num - d[key].shape[1]),
                            value=-1,
                        ).permute(0, 2, 1)
                        for d in batch
                    ]
                )
            )
            for key in batch[0]
        }
        return batch
    else:
        return default_collate(batch)


def pairwise_concatenate(tensors_list, dim=0):
    new_list = []
    n = len(tensors_list)
    for i in range(0, n - 1, 2):
        tensor1 = tensors_list[i]
        tensor2 = tensors_list[i + 1]
        concatenated_tensor = torch.cat([tensor1, tensor2], dim=dim)
        new_list.append(concatenated_tensor)
    if n % 2 != 0:
        new_list.append(tensors_list[-1])
    return new_list


def regroup_batch(batch, N, original_offsets, data_keys):
    num_segments = len(original_offsets)
    grouped_segments = {key: [[] for _ in range(N)] for key in data_keys}
    start_idx = 0
    for i in range(num_segments):
        end_idx = original_offsets[i]
        group_idx = i % N
        for key in data_keys:
            segment = batch[key][start_idx:end_idx]
            grouped_segments[key][group_idx].append(segment)
        start_idx = end_idx

    segment_lengths = original_offsets[1:] - original_offsets[:-1]
    segment_lengths = torch.cat([original_offsets[:1], segment_lengths])
    new_lengths_order = []
    split_lengths_order = []
    for i in range(N):
        segment_lengths_i = segment_lengths[i::N]
        n = segment_lengths_i.shape[0]
        num_pairs = n // 2
        paired_part = segment_lengths_i[: num_pairs * 2]
        reshaped_part = paired_part.view(num_pairs, 2, *segment_lengths_i.shape[1:])
        sums = torch.sum(reshaped_part, dim=1)
        first_parts = reshaped_part[:, 0]
        if n % 2 != 0:
            last_element = segment_lengths_i[-1]
            last_element = last_element.unsqueeze(0)
            final_result = torch.cat([sums, last_element], dim=0)
        else:
            final_result = sums
        new_lengths_order.append(final_result)
        split_lengths_order.append(first_parts)
    new_lengths_order = torch.stack(new_lengths_order, dim=1)
    new_lengths_order = new_lengths_order.flatten()
    new_offsets = torch.cumsum(new_lengths_order, dim=0)
    split_lengths_order = torch.stack(split_lengths_order, dim=1).flatten()

    new_batch = {}
    new_batch_imgs = []
    new_batch_img_num = 0
    img_num_offset = torch.cat(
        [torch.tensor([0]), torch.cumsum(batch["img_num"], dim=0)]
    )
    for key in data_keys:
        final_segments_in_order = []
        for i in range(N):
            grouped_segments[key][i] = pairwise_concatenate(
                grouped_segments[key][i], dim=0
            )
        for i in range(len(grouped_segments[key][0])):
            for j in range(N):
                final_segments_in_order.append(grouped_segments[key][j][i])
        new_batch[key] = torch.vstack(final_segments_in_order)
        if "correspondence" in key:
            current_start = 0
            N0, v, n_dim = new_batch[key].shape
            v2 = v * 2
            batch_correspondence_mix = -torch.ones(
                (N0, v2, n_dim),
                dtype=new_batch[key].dtype,
                device=new_batch[key].device,
            )
            for k, end in enumerate(new_offsets):
                len_part1 = split_lengths_order[k]
                split_point = current_start + len_part1
                if split_point > current_start:
                    mask1 = torch.any(
                        new_batch[key][current_start:split_point]
                        != torch.tensor([-1, -1]),
                        dim=2,
                    )
                    valid_index1 = torch.where(mask1)
                    if len(valid_index1[1]) == 0:
                        count1 = 0
                    else:
                        count1 = max(valid_index1[1])
                    batch_correspondence_mix[current_start:split_point, 0:count1] = (
                        new_batch[key][current_start:split_point, 0:count1]
                    )
                    if k % N == 0 and N == 2:
                        new_batch_imgs.append(
                            batch["images"][
                                img_num_offset[k // N * 2] : img_num_offset[k // N * 2]
                                + count1
                            ]
                        )
                if end > split_point:
                    mask2 = torch.any(
                        new_batch[key][split_point:end] != torch.tensor([-1, -1]), dim=2
                    )
                    valid_index2 = torch.where(mask2)
                    if len(valid_index2[1]) == 0:
                        count2 = 0
                    else:
                        count2 = max(valid_index2[1])
                    batch_correspondence_mix[
                        split_point:end, count1 : count1 + count2
                    ] = new_batch[key][split_point:end, 0:count2]
                    if k % N == 0 and N == 2:
                        new_batch_imgs.append(
                            batch["images"][
                                img_num_offset[k // N * 2 + 1] : img_num_offset[
                                    k // N * 2 + 1
                                ]
                                + count2
                            ]
                        )
                current_start = end

            if N == 2:
                new_batch_img_num = torch.tensor([i.shape[0] for i in new_batch_imgs])
                new_batch_imgs = torch.vstack(new_batch_imgs)
            else:
                new_batch_img_num = None
                new_batch_imgs = None
            new_batch[key] = batch_correspondence_mix
    return new_batch, new_offsets, new_batch_imgs, new_batch_img_num


def point_collate_fn(batch, mix_prob=0):
    assert isinstance(
        batch[0], Mapping
    )  # currently, only support input_dict, rather than input_list
    batch = collate_fn(batch)
    if random.random() < mix_prob:
        valid_keys = [
            "coord",
            "grid_coord",
            "origin_coord",
            "color",
            "normal",
            "feat",
            "correspondence",
        ]
        if "instance" in batch.keys():
            offset = batch["offset"]
            start = 0
            num_instance = 0
            for i in range(len(offset)):
                if i % 2 == 0:
                    num_instance = max(batch["instance"][start : offset[i]])
                if i % 2 != 0:
                    mask = batch["instance"][start : offset[i]] != -1
                    batch["instance"][start : offset[i]] += num_instance * mask
                start = offset[i]
        offset_assets = [asset for asset in batch.keys() if "offset" in asset]
        for offset_asset in offset_assets:
            batch[offset_asset] = torch.cat(
                [batch[offset_asset][1:-1:2], batch[offset_asset][-1].unsqueeze(0)],
                dim=0,
            )

        # Recompute grid_coord after mixing, because each scene's grid_coord was
        # independently shifted before mixing and is no longer consistent with
        # the merged coord. Only done when grid_size is available (e.g. LitePT
        # configs); other configs are unaffected.
        if "grid_coord" in batch and "grid_size" in batch:
            batch_idx = offset2batch(batch["offset"])
            scaled_coord = batch["coord"] / batch["grid_size"][0]
            grid_coord = torch.floor(scaled_coord).to(torch.int64)
            min_coord, _ = scatter_min(grid_coord, batch_idx, dim=0)
            batch["grid_coord"] = grid_coord - min_coord[batch_idx]
        offset_assets = [asset for asset in batch.keys() if "_offset" in asset]
        for offset_asset in offset_assets:
            offset_prefix = offset_asset.split("_")[0]
            valid_keys_with_prefix = [
                offset_prefix + "_" + valid_key for valid_key in valid_keys
            ]
            valid_keys_with_prefix = [
                valid_key_with_prefix
                for valid_key_with_prefix in valid_keys_with_prefix
                if valid_key_with_prefix in batch.keys()
            ]
            if "global" in offset_asset:
                N = 2
            elif "local" in offset_asset:
                N = 4
            updated_batch, new_offset, imgs, img_num = regroup_batch(
                batch, N, batch[offset_asset], valid_keys_with_prefix
            )
            batch[offset_asset] = new_offset
            batch.update(updated_batch)
            if "global" in offset_asset:
                batch["images"] = imgs
                batch["img_num"] = img_num

        if "img_num" in batch.keys():
            n = batch["img_num"].shape[0]
            num_pairs = n // 2
            len_pairs = num_pairs * 2
            pairs_tensor = batch["img_num"][:len_pairs]

            if num_pairs == 0:
                pass
            else:
                summed_pairs = pairs_tensor.view(-1, 2).sum(dim=1)
                if n % 2 != 0:
                    last_element = batch["img_num"][-1:]
                    result = torch.cat((summed_pairs, last_element))
                else:
                    result = summed_pairs
                batch["img_num"] = result
    return batch


def gaussian_kernel(dist2: np.array, a: float = 1, c: float = 5):
    return a * np.exp(-dist2 / (2 * c**2))
