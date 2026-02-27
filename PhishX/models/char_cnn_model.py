import torch
import torch.nn as nn
import torch.nn.functional as F

class CharCNN(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, num_classes=1, max_len=200):
        super(CharCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Conv layers with multiple filter sizes (n-grams)
        self.conv1 = nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embed_dim, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embed_dim, 64, kernel_size=7, padding=3)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 3, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, max_len)
        x = self.embedding(x) # (batch_size, max_len, embed_dim)
        x = x.transpose(1, 2) # (batch_size, embed_dim, max_len)
        
        x1 = F.relu(self.conv1(x))
        x1 = self.pool(x1).squeeze(-1)
        
        x2 = F.relu(self.conv2(x))
        x2 = self.pool(x2).squeeze(-1)
        
        x3 = F.relu(self.conv3(x))
        x3 = self.pool(x3).squeeze(-1)
        
        combined = torch.cat((x1, x2, x3), dim=1)
        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits

# Mapping characters to integers for CNN
class CharTokenizer:
    def __init__(self, max_len=200):
        self.max_len = max_len
        self.chars = "abcdefghijklmnopqrstuvwxyz0123456789-._~:/?#[]@!$&'()*+,;=%"
        self.char_to_int = {c: i + 1 for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1

    def tokenize(self, url):
        url = url.lower()
        seq = [self.char_to_int.get(c, 0) for c in url][:self.max_len]
        # Padding
        seq += [0] * (self.max_len - len(seq))
        return torch.tensor(seq).long()
