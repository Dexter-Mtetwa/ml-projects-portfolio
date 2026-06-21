import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc = nn.Linear(32*8*8, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) #32 -> 16
        x = self.pool(F.relu(self.conv2(x))) #16 -> 8
        x = x.view(x.size(0), -1) # flatten
        return self.fc(x)