import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, T, H)
        attn_weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        weighted_sum = (attn_weights * x).sum(dim=1)       # (B, H)
        return weighted_sum

class EmotionBiLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=129, hidden_dim=128, num_layers=2, n_classes=6, bidirectional=True):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attn_pooling = AttentionPooling(lstm_out_dim)
        self.norm = nn.LayerNorm(lstm_out_dim)
        self.fc = nn.Linear(lstm_out_dim, n_classes)

    def forward(self, x):
        # x: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)

        out, _ = self.lstm(x)  # out: (B, T, H*2)
        out = self.norm(out)
        pooled = self.attn_pooling(out)  # (B, H*2)

        logits = self.fc(pooled)
        return logits
