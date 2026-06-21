"""
    LSTM (for text sentiment) + Embedding (add nn.Embedding to turn words -> vectors, then feed to the LSTM)
    - nn.Embedding + nn.LSTM + nn.Linear + reset hidden + CrossEntropyLoss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Fake vocab + data for demo, Replace w/ real tokenizer + IMDB
vocab_size = 5000 # size of word dict
seq_len = 100 # max words per review
num_classes = 2 # positive/negative

# Random "tokenized" reviews: each int = word index
X = torch.randint(1, vocab_size, (2000, seq_len))
y = torch.randint(0, 2, (2000,)) # 0 = neg, 1 = pos
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)


# Model w/ Embedding
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # essentials: words -> vectors
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden=None):
        # x: [batch, seq_len] of word indices
        embedded = self.embedding(x) # -> [batch, seq_len, embed_dim]
        out, hidden = self.lstm(embedded, hidden) # -> [batch, seq_len, hidden]
        out = out[:, -1, :] # last timestep
        out = self.fc(out) # -> [batch, num_classes]
        return out, hidden
    

model = SentimentLSTM(vocab_size, embed_dim=128, hidden_size=128, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = int(input('enter epochs: '))

# Training loop w/ hidden reset
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        hidden = None # reset memory each batch
        outputs, _ = model(batch_x, hidden)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(loader)}')

