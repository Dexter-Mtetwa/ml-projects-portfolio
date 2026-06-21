import torch
import torch.nn.functional as F

def mse_manual(pred, target):
    return ((pred - target)**2).mean()

def cross_entropy_manual(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    nll = -log_probs[torch.arange(logits.size(0)), targets]
    return nll.mean()

def verify():
    logits = torch.randn(8, 5)
    targets = torch.randint(0, 5, (8,))
    loss1 = cross_entropy_manual(logits, targets)
    loss2 = mse_manual(logits, targets)
    print(f'CrossEntropy diff: {abs(loss1-loss2).item():.10f}')