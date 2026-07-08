import torch

# 运算
x = torch.tensor([1., 2., 3.])
y = torch.tensor([2., 3., 4.])
z = x + y
print(f"x+y = {z}")

# 张量连接
x_1 = torch.randn(2, 5)
y_1 = torch.randn(3, 5)
z_1 = torch.cat([x_1, y_1], 0)
print(z_1)

x_2 = torch.randn(2, 3)
y_2 = torch.randn(2, 5)
z_2 = torch.cat([x_2, y_2], 1)
print(z_2)

# 张量变形
x_3 = torch.randn(2, 3, 4)
print(f"x_3:{x_3}")
print(x_3.view(2,12))
# 维度为-1会自动推导数量
print(x_3.view(2,-1))