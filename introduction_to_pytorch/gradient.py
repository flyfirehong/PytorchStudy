import torch

def test1():
    # 追踪张量的计算过程，requires_grad意味着是否需要计算梯度
    x = torch.tensor([1., 2., 3.], requires_grad = True)
    y = torch.tensor([4., 5., 6.], requires_grad = True)
    z = x + y
    print(z)
    # 可以得到z的计算图
    print(z.grad_fn)

    s = z.sum()
    print(s.grad_fn)
    # 计算梯度
    s.backward()
    print(x.grad)

def test2():
    x = torch.randn(2, 2)
    y = torch.randn(2, 2)
    # 默认状态下开关状态是False
    print(x.requires_grad, y.requires_grad)

    # 赋值为True
    x = x.requires_grad_()
    y = y.requires_grad_()
    z = x + y
    print(z.grad_fn)
    # 只要输入有一个变量需要梯度，输出就需要梯度
    print(z.requires_grad)

    # 新张量和原来的张量共享同一块内存，但new_z只有数据，不再有计算图和梯度历史
    new_z = z.detach()
    print(new_z.grad_fn)


if __name__ == "__main__":
    test1()

