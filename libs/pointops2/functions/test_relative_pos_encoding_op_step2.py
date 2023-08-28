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

attn.requires_grad = True
v.requires_grad = True
table.requires_grad = True

v_flat = v[index_1]  # [M, h, hdim]
table_x, table_y, table_z = (
    table[:, :, :, 0],
    table[:, :, :, 1],
    table[:, :, :, 2],
)  # [L, h, hdim]
rel_index_x, rel_index_y, rel_index_z = (
    rel_index[:, 0],
    rel_index[:, 1],
    rel_index[:, 2],
)  # [M]
rel_pos_encoding = (
    table_x[rel_index_x] + table_y[rel_index_y] + table_z[rel_index_z]
)  # [M, h, hdim]
v_flat_new = v_flat + rel_pos_encoding  # [M, h, hdim]
output = attn.unsqueeze(-1) * v_flat_new  # [M, h, hdim]
output = scatter_sum(src=output, index=index_0, dim=0, dim_size=N)  # [N, h, hdim]
loss = output.mean()
loss.backward()

print(
    "output.shape: {}, output[:5,:10,:5]: {}".format(output.shape, output[:5, :10, :5])
)
print("attn.grad[:5, :3]: ", attn.grad[:5, :3])
print("v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
print("table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
input()

# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_0.is_contiguous(): ", index_0.is_contiguous())
# print("index_1.is_contiguous(): ", index_1.is_contiguous())

# output_v2 = pointops.attention_step2_with_rel_pos_value(attn, v, index_0.int(), index_1.int(), table, rel_index.int())
# loss = output_v2.mean()
# loss.backward()

# print("output_v2.shape: {}, output_v2[:5,:10,:5]: {}".format(output_v2.shape, output_v2[:5,:10,:5]))
# print("v2 attn.grad[:5, :3]: ", attn.grad[:5, :3])
# print("v2 v.grad[:5, :3, :5]: ", v.grad[:5, :3, :5])
# print("v2 table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# input()

# print("((output-output_v2)**2).max(): ", ((output-output_v2)**2).max())

# print("torch.max((attn_flat-attn_flat_v2)**2): ", torch.max((attn_flat-attn_flat_v2)**2))
