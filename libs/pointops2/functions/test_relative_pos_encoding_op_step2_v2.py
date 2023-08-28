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

M = 80000
N = 3500
hdim = 16
h = 6
L = 31
attn = torch.rand(M, h).cuda()
v = torch.rand(N, h, hdim).cuda()
table = torch.rand(L, h, hdim, 3).cuda()

index_0 = torch.rand(M)
index_0[index_0 < 0] = 0
index_0 = (index_0 * N).long().cuda()

index_1 = torch.rand(M)
index_1[index_1 < 0] = 0
index_1 = (index_1 * N).long().cuda()

rel_index = torch.rand(M, 3)
rel_index[rel_index < 0] = 0
rel_index = (rel_index * L).long().cuda()


# rearrange index for acceleration
index_0, indices = torch.sort(index_0)  # [M,]
index_1 = index_1[indices]  # [M,]
rel_index = rel_index[indices]
index_0_counts = index_0.bincount()

print("index_0_counts.shape: ", index_0_counts.shape)

n_max = index_0_counts.max()
index_0_offsets = index_0_counts.cumsum(dim=-1)  # [N]

print("v1 index_0_offsets.shape: ", index_0_offsets.shape)

index_0_offsets = torch.cat(
    [torch.zeros(1, dtype=torch.long).cuda(), index_0_offsets], 0
)  # [N+1]


attn.requires_grad = True
v.requires_grad = True
table.requires_grad = True


output = pointops.attention_step2_with_rel_pos_value(
    attn, v, index_0.int(), index_1.int(), table, rel_index.int()
)
loss = output.mean()
loss.backward()

print(
    "output.shape: {}, output[:5,:10,:5]: {}".format(output.shape, output[:5, :10, :5])
)
print("attn.grad[:5, :3]: ", attn.grad[:5, :3])
print("v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
print("table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# input()

attn_grad = attn.grad.clone()
v_grad = v.grad.clone()
table_grad = table.grad.clone()

attn.grad.zero_()
v.grad.zero_()
table.grad.zero_()

# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_0.is_contiguous(): ", index_0.is_contiguous())
# print("index_1.is_contiguous(): ", index_1.is_contiguous())

output_v2 = pointops.attention_step2_with_rel_pos_value_v2(
    attn, v, index_0_offsets.int(), n_max, index_1.int(), table, rel_index.int()
)
loss = output_v2.mean()
loss.backward()

print(
    "output_v2.shape: {}, output_v2[:5,:10,:5]: {}".format(
        output_v2.shape, output_v2[:5, :10, :5]
    )
)
print("v2 attn.grad[:5, :3]: ", attn.grad[:5, :3])
print("v2 v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
print("v2 table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# input()

print("((output-output_v2)**2).max(): ", ((output - output_v2) ** 2).max())

print("((attn_grad-attn.grad)**2).max(): ", ((attn_grad - attn.grad) ** 2).max())

print("((v_grad-v.grad)**2).max(): ", ((v_grad - v.grad) ** 2).max())

print("((table_grad-table.grad)**2).max(): ", ((table_grad - table.grad) ** 2).max())

# print("torch.max((attn_flat-attn_flat_v2)**2): ", torch.max((attn_flat-attn_flat_v2)**2))
