import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# 1. DATA: let's use synthetic text so no internet needed
text = """
To be or not to be, that is the question.
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""" * 200 # Repeat to make bigger dataset

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch:i for i, ch in enumerate(chars)}
idx_to_char = {i:ch for i, ch in enumerate(chars)}

print(f'Vocab size: {vocab_size}')
print(f'Chars: {"".join(chars)}')

# Convert text to tensor of indices
data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)


# 2. DATASET: Create sequences
class CharDataset(torch.utils.data.Dataset):
    def __init__(self, data, seq_len=100):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1] # next char prediction
        return x, y
    

# 3. MODEL: LSTM 'LLM'
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim) # char -> vector
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size) # predict next char

    def forward(self, x, hidden=None):
        # x:[B, seq_len] of indices
        x = self.embed(x) # [B, seq_len, embed_dim]
        out, hidden = self.lstm(x, hidden) # [B, seq_len, hidden_dim]
        logits = self.fc(out) # [B, seq_len, vocab_size]
        return logits, hidden
    

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # prevent exploding
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def generate_text(model, start_str, length=200, temp=0.8, device='cpu'):
    model.eval()
    chars = [char_to_idx[c] for c in start_str]
    hidden = None

    for _ in range(length):
        x = torch.tensor([chars[-100:]], dtype=torch.long).to(device) # last 100 characters
        logits, hidden = model(x, hidden)
        logits = logits[0, -1] / temp # temperature controls randomness
        probs = F.softmax(logits, dim=-1)
        next_char = torch.multinomial(probs, 1).item()
        chars.append(next_char)

    return ''.join(idx_to_char[i] for i in chars)


os.makedirs('saved', exist_ok=True)

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_len = 100
    batch_size = 64

    dataset = CharDataset(data, seq_len)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CharLSTM(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    print(f'Total params: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Training on {len(dataset)} sequences]\n')

    for epoch in range(1, 31):
        loss = train_epoch(model, loader, optimizer, device)
        if epoch % 5 == 0:
            print(f'Epoch {epoch:2d} | Loss: {loss:.4f}')
            sample = generate_text(model, 'To be', length=150, temp=0.8, device=device)
            print(f'Sample: {sample[:100]}...\n')

    # save model
    torch.save({
        'model_state': model.state_dict(),
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
    }, 'saved/char_lstm.pt')
    print('Model saved to saved/char_lstm.py')

    # Final generation
    print('='*50)
    print('FINAL GENERATED TEXT')
    print('='*50)
    final = generate_text(model, 'Whether', length=300, temp=0.7, device=device)
    print(final)
    print('\n✅ Congrats nika you just trained an LLM from scratch!')

    # To load model later
    def load_char_model(path, device):
        ckpt = torch.load(path, map_location=device)
        model = CharLSTM(ckpt['vocab_size']).to(device)
        model.load_state_dict(ckpt['model_state'])
        return model, ckpt['char_to_idx'], ckpt['idx_to_char']
    
    # usage: model, c21, i2c = load_char_model('saved/char_lstm.pt', device)