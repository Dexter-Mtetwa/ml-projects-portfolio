"""
MLP - a nueral net of Linear Layers, fully connected layers - each input is fully connected to each output
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=3, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=32)
        self.output = nn.Linear(in_features=32, out_features=num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.output(x)
        return x


# Sequential model
seq_model = nn.Sequential(
    nn.Linear(3, 16), nn.ReLU(),
    nn.Linear(16, 32), nn.ReLU(),
    nn.Linear(32, 1)
)

model = MLP()
optimizer = torch.optim.SGD(seq_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
epochs = 50

# data batches
X = torch.randn(1000, 3)
y = torch.randint(0, 2, (1000,), dtype=torch.float)
loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=64)

print(y.shape)

for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        y_preds = seq_model(batch_x)
        loss = loss_fn(y_preds, batch_y.reshape(-1, 1))
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss}')