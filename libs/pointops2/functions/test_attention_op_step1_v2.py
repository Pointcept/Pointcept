import torch
import pointops
from torch_scatter import (
    scatter_max,
    scatter_mean,
    scatter_add,
    scatter_min,
    scatter_sum,
)

torch.manual_seed(1)

M = 800000
N = 35000
C = 96
h = 6
query = torch.rand(N, h, C // h).cuda()
key = torch.rand(N, h, C // h).cuda()

index_0 = torch.rand(M)
index_0[index_0 < 0] = 0
index_0 = (index_0 * N).long().cuda()

index_1 = torch.rand(M)
index_1[index_1 < 0] = 0
index_1 = (index_1 * N).long().cuda()

query.requires_grad = True
key.requires_grad = True


attn_flat = pointops.attention_step1(
    query.float(), key.float(), index_0.int(), index_1.int()
)
loss = attn_flat.sum()
loss.backward()
print(
    "attn_flat.shape: {}, attn_flat[:20,:10]: {}".format(
        attn_flat.shape, attn_flat[:20, :10]
    )
)
print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
input()


# rearrange index for acceleration
index_0, indices = torch.sort(index_0)  # [M,]
index_1 = index_1[indices]  # [M,]
index_0_counts = index_0.bincount()

print("index_0_counts.shape: ", index_0_counts.shape)

n_max = index_0_counts.max()
index_0_offsets = index_0_counts.cumsum(dim=-1)  # [N]

print("v1 index_0_offsets.shape: ", index_0_offsets.shape)

index_0_offsets = torch.cat(
    [torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0
)  # [N+1]

# print("index_0[:100]: ", index_0[:100])
print("n_max: ", n_max)
print("index_0_offsets.shape: ", index_0_offsets.shape)
# input()

print("index_0_offsets[:100]: ", index_0_offsets[:100])
print("index_1[:20]: ", index_1[:20])


attn_flat = pointops.attention_step1(
    query.float(), key.float(), index_0.int(), index_1.int()
)
# loss = attn_flat.sum()
# loss.backward()
# # attn_flat = pointops.attention_step1(query.float(), key.float(), index_0.int(), index_1.int())
# # loss = attn_flat.sum()
# # loss.backward()
# print("attn_flat.shape: {}, attn_flat[:20,:10]: {}".format(attn_flat.shape, attn_flat[:20,:10]))
# print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
# print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# input()

print("query.is_contiguous(): ", query.is_contiguous())
print("key.is_contiguous(): ", key.is_contiguous())
print("index_0.is_contiguous(): ", index_0.is_contiguous())
print("index_1.is_contiguous(): ", index_1.is_contiguous())

attn_flat_v2 = pointops.attention_step1_v2(
    query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max
)
loss = attn_flat_v2.sum()
loss.backward()

# attn_flat_v2 = pointops.attention_step1_v2(query.float(), key.float(), index_1.int(), index_0_offsets.int(), n_max)
# loss = attn_flat_v2.sum()
# loss.backward()

print(
    "attn_flat_v2.shape: {}, attn_flat_v2[:20,:10]: {}".format(
        attn_flat_v2.shape, attn_flat_v2[:20, :10]
    )
)
print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# input()

# mask = attn_flat_v2.sum(-1) != 0
# print("mask.sum(): ", mask.sum())
# print("attn_flat_v2[mask] - attn_flat[mask]: ", ((attn_flat_v2[mask] - attn_flat[mask])**2).max())


print(
    "((attn_flat-attn_flat_v2)**2 < 1e-8).all(): ",
    ((attn_flat - attn_flat_v2) ** 2 < 1e-8).all(),
)

selected = 10000
print(
    "torch.max((attn_flat[:selected]-attn_flat_v2[:selected])**2, 0): ",
    torch.max((attn_flat[:selected] - attn_flat_v2[:selected]) ** 2, 0),
)
