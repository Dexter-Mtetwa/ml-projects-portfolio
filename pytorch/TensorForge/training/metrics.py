import torch

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean()

def confusion_matrix(pred, target, n_classes):
    pred = pred.argmax(dim=1)
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for t, p in zip(target, pred):
        cm[t, p] += 1
    return cm