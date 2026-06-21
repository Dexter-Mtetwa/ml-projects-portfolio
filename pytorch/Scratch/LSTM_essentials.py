"""
    LSTM - the fix for vanilla RNNs forgetting long stuff.
    Essentials - nn.LSTM + nn.Linear + CrossEntropyLoss + reset hidden states between batches
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Model: LSTM + Linear on top
class SimpleLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_classes=5):
        super().__init__()
        # input: [batch, seq_len, input_size]
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes) # map last hidden to output

    def forward(self, x, hidden=None):
        # x: [batch, seq_len, input_size]
        out, hidden = self.lstm(x, hidden) # out: [batch, seq_len, hidden_size]
        out = out[:, -1, :] # take last timestep hidden state
        out = self.fc(out) # -> [batch, num_classes]
        return out, hidden

# dummy data: 500 sequences, length 20, 10 features
X = torch.randn(500, 20, 10)
y = torch.randint(0, 5, (500,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# essentials: loss, optimizer, training loop + reset hidden
model = SimpleLSTM(input_size=10, hidden_size=64, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = int(input('enter epochs: '))

for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        hidden = None # reset hidden states each batch so sequences don't leak
        outputs, _ = model(batch_x, hidden)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

