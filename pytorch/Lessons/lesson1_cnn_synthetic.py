import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import numpy as np

# 1. DATASET: Generate fake images w/ patterns per class
class SyntheticImageDataset(Dataset):
    """

    Each class = colored square in different corner + noise
    Class 0: red square top-left, Class 1: green top-right, etc
    """

    def __init__(self, n_samples=5000, img_size=32, n_classes=10):
        self.n_classes = n_classes
        self.img_size = img_size
        self.X = torch.zeros(n_samples, 3, img_size, img_size)
        self.y = torch.randint(0, n_classes, (n_samples,))

        for i in range(n_samples):
            cls = self.y[i].item()
            img = torch.randn(3, img_size, img_size) * 0.3 # noise background

            # Put class-specific pattern
            row = (cls // 3) * 8 # 0, 8, 16 for 3 rows
            col = (cls % 3) * 8
            color = torch.tensor([cls % 3 == 0, cls % 3 == 1, cls % 3 == 2]).float()

            img[:, row:row+8, col:col+8] += color.view(3, 1, 1) * 2.0
            self.X[i] = img

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    




# 2. CNN Model: Conv2d + Pool + FC
class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Conv keeps H,W same if padding = 1, kernel = 3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) # [B,3,32,32] -> [B,16,32,32]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # [B,16,16,16] -> [B,32,16,16]
        self.pool = nn.MaxPool2d(2, 2) # halves H,W: 32->16->8

        # After 2 pools: 32x32 -> 8x8, Channels = 32
        self.fc = nn.Linear(32*8*8, num_classes)

    def forward(self, x):
        # Print shapes once to learn
        if not hasattr(self, '_printed'):
            print(f'Input: {x.shape}')

        x = self.pool(F.relu(self.conv1(x)))
        if not hasattr(self, '_printed'):
            print(f'After conv1 + pool: {x.shape}')

        x = self.pool(F.relu(self.conv2(x)))
        if not hasattr(self, '_printed'):
            print(f'After conv2 + pool: {x.shape}')

        x = x.view(x.size(0), -1) # flatten: [B,32*8*8]
        if not hasattr(self, '_printed'):
            print(f'After flatten: {x.shape}')
            self._printed = True

        x = self.fc(x)
        return x
    




# 3. Training Loop
def train(model, loader, optimizer, device):
    model.train()
    total_loss, correct = 0, 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(1) == y).sum().item()

        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx:3d} | Loss: {loss.item():.4f}')

    acc = correct / len(loader.dataset)
    return total_loss / len(loader), acc


# test
def test(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
    return correct / len(loader.dataset)






if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Create dataset + dataloader
    train_ds = SyntheticImageDataset(n_samples=4000)
    test_ds = SyntheticImageDataset(n_samples=1000)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    print(f'Train samples: {len(train_ds)}, Test samples: {len(test_ds)}')
    print(f'One batch shape: {next(iter(train_loader))[0].shape}')

    # Model + optimizer
    model = TinyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f'\n Total params: {sum(p.numel() for p in model.parameters()):,}')

    # Train
    for epoch in range(1, 21):
        train_loss, train_acc = train(model, train_loader, optimizer, device)
        test_acc = test(model, test_loader, device)
        print(f'Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Test Acc: {test_acc*100:.1f}%')

    print('✅ CNN trained on synthetic data. No internet needed!')