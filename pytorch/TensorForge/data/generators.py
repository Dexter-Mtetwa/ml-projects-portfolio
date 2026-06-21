import torch
# import math

def generate_spiral(n_points=1000, noise=0.2):
    """Classic non-linear dataset"""
    X, y = [], []
    for class_idx in range(3):
        r = torch.linspace(0, 1, n_points//3)
        t = torch.linspace(class_idx*4, (class_idx+1)*4, n_points//3) + torch.randn(n_points//3)*noise
        X.append(torch.stack([r*torch.cos(t), r*torch.sin(t)], dim=1))
        y.append(torch.full((n_points//3,), class_idx))
    return torch.cat(X), torch.cat(y)

def generate_xor(n_points=1000):
    """XOR problem - needs non-linear model"""
    X = torch.randint(0, 2, (n_points, 2)).float()
    y = (X[:, 0]!=X[:, 1]).long()
    return X, y