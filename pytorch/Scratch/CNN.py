"""
CNN
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Conv2d + ReLU + Linear
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # input: [batch, 3, 28, 28]
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) #[B,32,28,28]
        self.batchnorm1 = nn.BatchNorm2d(32) 
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2) #[B,32,14,14]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) #[B,64,14,14]
        self.batchnorm2 = nn.BatchNorm2d(64) 
        self.maxpool2 = nn.MaxPool2d(2, 2) #[B,64,7,7]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) #[B,128,7,7]
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*7*7, num_classes) # -> [batch, num_classes]
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.batchnorm1(self.conv1(x))))
        x = self.maxpool2(self.relu(self.batchnorm2(self.conv2(x))))
        x = self.relu(self.conv3(x))
        x = self.flatten(x)
        out = self.fc(x)
        return out


# dummy data: 1000 samples, 3 channels, 28*28
X = torch.randn(1000, 3, 28, 28)
y = torch.randint(0, 10, (1000,))
loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=28)


model = CNN(num_classes=10)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()
epochs = 50


for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        y_preds = model(batch_x)
        loss = loss_fn(y_preds, batch_y)
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')