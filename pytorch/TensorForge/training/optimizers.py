import torch

class SGDFullFromScratch:
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.state = {}
        for p in self.params:
            self.state[p] = {'velocity': torch.zeros_like(p)}

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                v = self.state[p]['velocity']
                v.mul_(self.momentum).add_(p.grad)
                p.sub_(self.lr*v)
