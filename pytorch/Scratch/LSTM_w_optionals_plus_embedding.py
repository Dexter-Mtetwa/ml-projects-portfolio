"""
    LSTM + Embedding (add nn.Embedding to turn words -> vectors, then feed to the LSTM)
    - adds bidirectional, dropout, pack_padded_sequence for real variable-length reviews
    - nn.Embedding is what makes LSTM work on text - w/out it you'd feed one-hot vectors and trainig would crawl
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
import random

class BetterSentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_size=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0) # essentials: words -> vectors
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, 
                            num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3) # optional: 2 layers + dropout between
        self.fc = nn.Linear(hidden_size * 2, num_classes) # * 2 because bidirectional

    def forward(self, x, lengths):
        embedded = self.embedding(x) # -> [batch, seq_len, embed_dim]

        # pack padded sequences optional: skip padding in LSTM compute
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed) 
        out, _ = pad_packed_sequence(packed_out, batch_first=True)

        # concat last hidden states from both directions
        h_forward = h_n[-2, :, :] # last layer, forward
        h_backward = h_n[-1, :, :] # last layer, backward
        h_cat = torch.cat((h_forward, h_backward), dim=1)

        out = self.fc(h_cat)
        return out

# dummy variable-length reviews
vocab_size = 5000 # size of word dict
seqs, labels, lengths = [], [], []
for i in range(200):
    length = random.randint(20, 100)
    seq = torch.randint(1, vocab_size, (length,))
    seqs.append(seq)
    labels.append(random.randint(0, 1))
    lengths.append(length)


X = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
y = torch.tensor(labels)
lengths = torch.tensor(lengths)
loader = DataLoader(list(zip(X, y, lengths)), batch_size=64, shuffle=True)

# essentials: loss, optimizer, training loop + reset hidden
model = BetterSentimentLSTM(vocab_size, embed_dim=128, hidden_size=128, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = int(input('enter epochs: '))

for epoch in range(epochs):
    for batch_x, batch_y, batch_len in loader:
        optimizer.zero_grad()
        outputs = model(batch_x, batch_len)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, Loss: {loss.item()}')

"""
To run on real IMDB:
1. swap dummy X, y w/ torchtext.datasets.IMDB
2. build vocab w/ torchtext.vocab.build_vocab_from_iterator
3. tokenize + numericalize reviews before fedding
"""