import torch
import torch.nn.functional as F
from utils.console import print_loss

def train_epoch(model, loader, optimizer, device, clip_grad=False):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Grad clipping
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print_loss(batch_idx, loss.item())
    return total_loss / len(loader)

def eval_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total # somewhat accuracy