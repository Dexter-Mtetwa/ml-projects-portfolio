import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# import matplotlib.pyplot as plt

# let's Reuse data from lesson 1
class SyntheticImageDataset(Dataset):
    def __init__(self, n_samples=5000, img_size=32, n_classes=10):
        self.X = torch.randn(n_samples, 3, img_size, img_size) * 0.3
        self.y = torch.randint(0, n_classes, (n_samples,))

        for i in range(n_samples):
            cls = self.y[i].item()
            row, col = (cls // 3) * 8, (cls % 3) * 8
            color = torch.tensor([cls % 3 == 0, cls % 3 == 1, cls % 3 == 2]).float()
            self.X[i, :, row:row+8, col:col+8] += color.view(3, 1, 1) * 2.0

    def __len__(self): return len(self.X)
    def __getitem__(self, index): return self.X[index], self.y[index]



# CNN Without BatchNorm
class CNN_NoBN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv keeps H,W same if padding = 1, kernel = 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # [B,32,16,16]
        x = self.pool(F.relu(self.conv2(x))) # [B,64,8,8]
        x = x.view(x.size(0), -1) # safe flatten
        return self.fc(x)
        


# CNN w/ BatchNorm
class CNN_WithBN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv keeps H,W same if padding = 1, kernel = 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32) # Normalize each of 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2) 
        self.fc = nn.Linear(64*8*8, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)
    

def train_one(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def test_one(model, loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return correct / len(loader.dataset)


def compare_batch_sizes():
    """Show why batch_size mattes for BatchNorm"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = SyntheticImageDataset(n_samples=2000)

    batch_sizes = [4, 32, 256]
    results = {}

    for bs in batch_sizes:
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
        model = CNN_WithBN().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        losses = []
        for epoch in range(30):
            loss = train_one(model, loader, optimizer, device)
            losses.append(loss)

        results[bs] = losses
        print(f'Batch {bs:3d} | Final loss: {losses[-1]:.4f}')


"""
# Plot
plt.figure(figsize=(8,4))
for bs, losses in results.items():
    plt.plot(losses, label=f'batch_size={bs}')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('BatchNorm: Small batches = noisy stats')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('saved/batchnorm_bs_comparison.png')
print('📊 Plot saved: saved/batchnorm_bs_comparison.png')

"""

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_ds = SyntheticImageDataset(n_samples=4000)
    test_ds = SyntheticImageDataset(n_samples=1000)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    print('='*50)
    print('Training w/out BatchNorm')
    print('='*50)
    model_no_bn = CNN_NoBN().to(device)
    opt_no_bn = torch.optim.SGD(model_no_bn.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(1, 21):
        loss = train_one(model_no_bn, train_loader, opt_no_bn, device)
        acc = test_one(model_no_bn, test_loader, device)
        if epoch % 5 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%')

    print('\n' + '='*50)
    print('Training w/ BatchNorm')
    print('='*50)
    model_bn = CNN_WithBN().to(device)
    opt_bn = torch.optim.SGD(model_bn.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(1, 21):
        loss = train_one(model_bn, train_loader, opt_bn, device)
        acc = test_one(model_bn, test_loader, device)
        if epoch % 5 == 0:
            print(f'Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%')

    print('\n' + '='*50)
    print('Testing different batch sizes w/ BatchNorm')
    print('='*50)
    compare_batch_sizes()

    print('\n✅ Key lesson BatchNorm lets you use higher LR + converges faster')