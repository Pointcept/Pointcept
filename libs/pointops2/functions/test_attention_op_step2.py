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
softmax_attn_flat = torch.rand(M, h).cuda()
value = torch.rand(N, h, C // h).cuda()

index_0 = torch.rand(M)
index_0[index_0 < 0] = 0
index_0 = (index_0 * N).long().cuda()

index_1 = torch.rand(M)
index_1[index_1 < 0] = 0
index_1 = (index_1 * N).long().cuda()

softmax_attn_flat.requires_grad = True
value.requires_grad = True

# value_flat = value[index_1] #[M, num_heads, C // num_heads]
# x = (softmax_attn_flat.unsqueeze(-1) * value_flat).reshape(M, C)
# x = scatter_sum(src=x, index=index_0, dim=0, dim_size=N) #[N, C]
# loss = x.sum()
# loss.backward()

# print("x.shape: {}, x[:5,:10]: {}".format(x.shape, x[:5,:10]))
# print("softmax_attn_flat.grad[:5, :10]: ", softmax_attn_flat.grad[:5, :10])
# print("value.grad[:5, :3, :5]: ", value.grad[:5, :3, :5])
# input()

print("softmax_attn_flat.is_contiguous(): ", softmax_attn_flat.is_contiguous())
print("value.is_contiguous(): ", value.is_contiguous())
print("index_0.is_contiguous(): ", index_0.is_contiguous())
print("index_1.is_contiguous(): ", index_1.is_contiguous())

x_v2 = pointops.attention_step2(
    softmax_attn_flat.float(), value.float(), index_0.int(), index_1.int()
)
x_v2 = x_v2.view(N, C)
loss = x_v2.sum()
loss.backward()

print("x_v2.shape: {}, x_v2[:5,:10]: {}".format(x_v2.shape, x_v2[:5, :10]))

print("softmax_attn_flat.grad[:5, :10]: ", softmax_attn_flat.grad[:5, :10])
print("value.grad[:5, :3, :5]: ", value.grad[:5, :3, :5])
input()

print("((x-x_v2)**2 < 1e-8).all(): ", ((x - x_v2) ** 2 < 1e-8).all())

print("torch.max((x-x_v2)**2): ", torch.max((x - x_v2) ** 2))
