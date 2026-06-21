import torch
import torch.nn.functional as F
# import math

def relu_manual(x):
    return torch.maximum(x, torch.zeros_like(x))

def sigmoid_manual(x):
    return 1 / (1 + torch.exp(-x))

def softmax_manual(x, dim=-1):
    exp_x = torch.exp(x - x.max(dim, keepdim=True)[0]) # stability trick
    return exp_x / exp_x.sum(dim, keepdim=True)

def verify():
    x = torch.randn(5, 3)
    print(f'ReLU diff: {torch.abs(relu_manual(x) - F.relu(x)).max()}')
    print(f'Sigmoid diff: {torch.abs(sigmoid_manual(x) - torch.sigmoid(x)).max()}')
    print(f'Softmax diff: {torch.abs(softmax_manual(x) - F.softmax(x, dim=1)).max()}')