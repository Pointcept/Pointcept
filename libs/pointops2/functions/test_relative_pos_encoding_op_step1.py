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
query = torch.rand(N, h, hdim).cuda()
table = torch.rand(L, h, hdim, 3).cuda()

index = torch.rand(M)
index[index < 0] = 0
index = (index * N).long().cuda()

rel_index = torch.rand(M, 3)
rel_index[rel_index < 0] = 0
rel_index = (rel_index * L).long().cuda()

query.requires_grad = True
table.requires_grad = True

# query_flat = query[index] #[M, h, hdim]
# table_x, table_y, table_z = table[:,:,:,0], table[:,:,:,1], table[:,:,:,2] #[L, h, hdim]
# rel_index_x, rel_index_y, rel_index_z = rel_index[:,0], rel_index[:,1], rel_index[:,2] #[M]
# rel_pos_encoding = table_x[rel_index_x] + table_y[rel_index_y] + table_z[rel_index_z] #[M, h, hdim]
# output = (query_flat * rel_pos_encoding).sum(-1) #[M, h]
# loss = output.mean()
# loss.backward()

# print("output.shape: {}, output[:5,:10]: {}".format(output.shape, output[:5,:10]))
# print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
# print("table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
# input()

# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_0.is_contiguous(): ", index_0.is_contiguous())
# print("index_1.is_contiguous(): ", index_1.is_contiguous())

output_v2 = pointops.dot_prod_with_idx(query, index.int(), table, rel_index.int())
loss = output_v2.mean()
loss.backward()

print(
    "output_v2.shape: {}, output_v2[:5,:10]: {}".format(
        output_v2.shape, output_v2[:5, :10]
    )
)
print("v2: query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("v2: table.grad[:5, :3, :5, :2]: ", table.grad[:5, :3, :5, :2])
input()

# print("((output-output_v2)**2).max(): ", ((output-output_v2)**2).max())

# print("torch.max((attn_flat-attn_flat_v2)**2): ", torch.max((attn_flat-attn_flat_v2)**2))
