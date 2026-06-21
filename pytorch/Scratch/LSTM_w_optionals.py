"""
    LSTM + optionals
    - bidirectional = True , reads context both ways
    - dropout , stops overfitting on deep stacks
    - pack_padded_sequence , skips padding tokens for efficiency
    - attention , focuses on important timesteps instead of just the end - replaces "out = out[:, -1, :]" with "out = self.attn(out)"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


# Simple attention module optional
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden*2] if bidirectional
        weights = torch.softmax(self.attn(lstm_out), dim=1) # [batch, seq_len, 1]
        context = torch.sum(weights * lstm_out, dim=1) # weighted sum
        return context


class BetterLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_classes=5):
        super().__init__()
        # input: [batch, seq_len, input_size]
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3) # optional: 2 layers + dropout between
        self.attn = Attention(hidden_size * 2) # optional: attention
        self.fc = nn.Linear(hidden_size * 2, num_classes) # * 2 because bidirectional

    def forward(self, x, lengths, hidden=None):
        # x: [batch, seq_len, input_size]
        # pack for variable length sequences optional
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, hidden = self.lstm(packed, hidden) 
        out, _ = pad_packed_sequence(packed_out, batch_first=True) # [batch, seq_len, hidden * 2]

        out = self.attn(out) # optional: attention instead of just last timestep
        out = self.fc(out) # -> [batch, num_classes]
        return out, hidden

# dummy variable-length data
seqs, labels, lengths = [], [], []
for i in range(200):
    length = torch.randint(5, 20, (1,)).item()
    seqs.append(torch.randn(length, 10))
    labels.append(torch.randint(0, 5, (1,)).item())
    lengths.append(length)

# pad sequences
X = nn.utils.rnn.pad_sequence(seqs, batch_first=True) # [batch, max_seq, 10]
y = torch.tensor(labels)
lengths = torch.tensor(lengths)
loader = DataLoader(list(zip(X, y, lengths)), batch_size=32, shuffle=True)

# essentials: loss, optimizer, training loop + reset hidden
model = BetterLSTM(input_size=10, hidden_size=64, num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = int(input('enter epochs: '))

for epoch in range(epochs):
    for batch_x, batch_y, batch_len in loader:
        optimizer.zero_grad()
        hidden = None # reset hidden states each batch so sequences don't leak
        outputs, _ = model(batch_x, batch_len, hidden)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

