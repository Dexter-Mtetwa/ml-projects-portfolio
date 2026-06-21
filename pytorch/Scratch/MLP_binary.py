"""
    MLP + binary classification

    1 output + BCEWithLogitsLoss()
        - Output: 1 logit per sample. logit > 0 means 'dog', logit < 0 means 'cat'
        - Label: y is float 0.0 for cat, 1.0 for dog
        - Loss: nn.BCEWithLogitsLoss()
        - Prediction: prob = torch.sigmoid(logits), then prob > 0.5
        - Standard for binary. More stable numerically
            model = nn.Linear(20, 1)
            logits = model(x) # Shape [N, 1]
            loss = nn.BCEWithLogitsLoss()(logits.squeeze(), y.float())
            pred = torch.sigmoid(logits) > 0.5

    2 outputs + CrossEntropyLoss
        - Output: 2 logits per sample = [logit_cat, logit_dog]
        - Label: y is long 0 for cat, 1 for dog
        - Loss: nn.CrossEntropyLoss() handles softmax internally
        - Prediction: pred = logits.argmax(dim=1)
        - Same math, just written as multi-class
            model = nn.Linear(20, 2)
            logits = model(x) # Shape [N, 2]
            loss = nn.CrossEntropyLoss(logits, y.long())
            pred = logits.argmax(dim=1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP_Binary(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_dim, 1) # 1 logit output

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x).squeeze(1) # don't put sigmoid in forward


# Sequential model
seq_model = nn.Sequential(
    nn.Linear(20, 64), 
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
    nn.Sigmoid()
)


# data batches
n, d = 5000, 20
X = torch.randn(n, d)
w = torch.randn(d, 1)
y = (X @ w + 0.5 * torch.randn(n, 1) > 0).float().squeeze() # 0.0 or 1.0

loader = DataLoader(TensorDataset(X, y), shuffle=True, batch_size=64)

print(y.shape)

model = MLP_Binary(d, 64)
criterion = nn.BCEWithLogitsLoss() # expects raw logits
optimizer = optim.Adam(seq_model.parameters(), lr=1e-3)
epochs = 20


for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = seq_model(batch_x)
        loss = criterion(logits, batch_y.view(-1, 1))
        loss.backward()
        optimizer.step()

    print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# Predict prob = torch.sigmoid(model(x))
