import torch

# 种子数相同时（比如这里的 1），每次运行代码时产生的随机数序列都会完全一样。
torch.manual_seed(1)

# 用浮点数列表创建一维张量
V_data = [1., 2., 3.]
V = torch.tensor(V_data) 
print(f"V: {V}")

# 创建二维张量
M_data = [[1., 2., 3.],[4., 5., 6.]]
M = torch.tensor(M_data)
print(f"M: {M}")

# 创建三维张量
T_data = [
    [[1., 2.], [3., 4.]],
    [[5., 6.], [7., 8.]]
]
T = torch.tensor(T_data)
print(f"T: {T}")

print(V[0])
# 获取一个Python数字
print(V[0].item())
print(M[0])
print(T[0])

# 创建指定维度的随机数，这里表示张量里面有三个二维，二维里面有四个一维，每个一维中5个值
x = torch.randn((3,4,5))
print(x)
