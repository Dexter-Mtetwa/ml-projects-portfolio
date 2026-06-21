"""
CNN essentials (conv layer, loss, optimizer and training loop)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # input: [batch, 3, 28, 28] like MNIST but with 3 channels
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) #[B,32,28,28]
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*28*28, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


# dummy data: 1000 samples, 3 channels, 28*28
X = torch.randn(1000, 3, 28, 28)
y = torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=28)

# essentials: loss, optimizer, training loop
model = SimpleCNN(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
epochs = 50


for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        y_preds = model(batch_x)
        loss = criterion(y_preds, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')