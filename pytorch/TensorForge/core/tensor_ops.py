import torch

def tensor_gym():
    """Drill every creation + shape op"""
    a = torch.randn(1000, 784)
    idx = torch.randperm(1000)
    # train, test = a[idx[:800]], a[idx[800:]]

    b = torch.arange(24).reshape(2, 3, 4)
    # c = b.permute(2,0,1) #swap dims: [4, 2, 3]

    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    y = x @ x.T
    y.sum().backward()
    print(f'grad = \n{x.grad}')
    print(f'should be [[2, 4], [6, 8]]')
    return x.grad 