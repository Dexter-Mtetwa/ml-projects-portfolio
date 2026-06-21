"""
CNN + optionals (BatchNorm, MaxPool, Dropout, Data Augmentation and LR scheduler)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms


class BetterCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # input: [batch, 3, 28, 28]
            nn.Conv2d(3, 32, kernel_size=3, padding=1), #[B,32,28,28]
            nn.BatchNorm2d(32), # optional: stabilize training
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # optional: downsample 28x28 -> 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1), #[B,64,14,14]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2) # 14x14 -> 7x7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # optional: prevent overfitting
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Data augmentation as optional
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # optional: augment
    transforms.RandomRotation(10),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# still using dummy data for simplicity, Replace w/ real dataset + transform
X = torch.randn(1000, 3, 28, 28)
y = torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=28)

# real dataset: CIFAR10
# trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
# train_loader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

# training w/ LR scheduler optional
model = BetterCNN(num_classes=10)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1) # optional
epochs = 50


for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step() # optinal: decay LR
    print(f'Epoch: {epoch+1}, Loss: {running_loss/len(loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')