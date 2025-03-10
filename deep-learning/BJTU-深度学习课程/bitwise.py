# /usr/bin/env python3
# -*- coding:utf-8 -*-
import torch


x_uint = torch.Tensor([[5, 6, 2], [1, 1, 0], [0, 5, 1]]).type(torch.uint8)

x_bool = torch.Tensor([[1, 0, 1], [1, 0, 0], [0, 1, 1]]).type(torch.bool)
print(x_bool)
print(~x_bool)
print(x_uint)
print()

# torch.bitwise_not(input, out=None) → Tensor
# 计算给定输入张量的按位非. 输入张量必须是整数或布尔类型. 对于布尔张量，它计算逻辑非.
x_not = torch.bitwise_not(x_bool)
# print(x_not)

# masked_fill_(mask, value)
# 使用mask为True的value填充self张量的元素. mask的形状必须与基础张量的形状一起广播 .
print(x_uint.masked_fill(x_not, 0))
print(x_uint.masked_fill(~x_bool, 0))

print(x_uint.dtype==torch.uint8)

# 注意 uint8 相乘后溢出
x_uint1 = torch.Tensor([[15, 17, 12], [9,4,2]]).type(torch.uint8) # [255, 33, 144] => 溢出 17*17-256->33
print(x_uint1*x_uint1)

# x_uint2 = torch.Tensor([[10, 10, 10], [2,2,2]]).type(torch.uint8)
# mask = x_uint2<=9
# x_uint1[mask] = 2*x_uint1[mask]*x_uint2[mask]
#
# mask = x_uint2>9
# x_uint1[mask] = x_uint1[mask]*x_uint2[mask]
# print(x_uint1)