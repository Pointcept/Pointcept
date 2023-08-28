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
table_q = torch.rand(L, h, hdim, 3).cuda()
key = torch.rand(N, h, hdim).cuda()
table_k = torch.rand(L, h, hdim, 3).cuda()

index_q = torch.rand(M)
index_q[index_q < 0] = 0
index_q = (index_q * N).long().cuda()

index_k = torch.rand(M)
index_k[index_k < 0] = 0
index_k = (index_k * N).long().cuda()

rel_index = torch.rand(M, 3)
rel_index[rel_index < 0] = 0
rel_index = (rel_index * L).long().cuda()

query.requires_grad = True
table_q.requires_grad = True
key.requires_grad = True
table_k.requires_grad = True

output1 = pointops.dot_prod_with_idx(query, index_q.int(), table_q, rel_index.int())
output2 = pointops.dot_prod_with_idx(key, index_k.int(), table_k, rel_index.int())
output = output1 + output2
# loss = output.mean()
# loss.backward()

# print("output.shape: {}, output[:5,:10]: {}".format(output.shape, output[:5,:10]))
# print("query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
# print("table_q.grad[:5, :3, :5, :2]: ", table_q.grad[:5, :3, :5, :2])
# print("key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
# print("table_k.grad[:5, :3, :5, :2]: ", table_k.grad[:5, :3, :5, :2])
# input()

# print("query.is_contiguous(): ", query.is_contiguous())
# print("key.is_contiguous(): ", key.is_contiguous())
# print("index_0.is_contiguous(): ", index_0.is_contiguous())
# print("index_1.is_contiguous(): ", index_1.is_contiguous())

output_v2 = pointops.dot_prod_with_idx_v2(
    query, index_q.int(), key, index_k.int(), table_q, table_k, rel_index.int()
)
loss = output_v2.mean()
loss.backward()

print(
    "output_v2.shape: {}, output_v2[:5,:10]: {}".format(
        output_v2.shape, output_v2[:5, :10]
    )
)
print("v2 query.grad[:5, :3, :5]: ", query.grad[:5, :3, :5])
print("v2 table_q.grad[:5, :3, :5, :2]: ", table_q.grad[:5, :3, :5, :2])
print("v2 key.grad[:5, :3, :5]: ", key.grad[:5, :3, :5])
print("v2 table_k.grad[:5, :3, :5, :2]: ", table_k.grad[:5, :3, :5, :2])
# input()

print("((output-output_v2)**2).max(): ", ((output - output_v2) ** 2).max())
