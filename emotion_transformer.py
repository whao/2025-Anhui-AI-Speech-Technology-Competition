import torch
import torch.nn as nn

class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=129, n_classes=6, dim_model=128, num_heads=4, num_layers=4, dropout=0.1, max_len=300):
        super(EmotionTransformer, self).__init__()

        self.input_fc = nn.Linear(input_dim, dim_model)  # 映射到 Transformer 的维度
        self.positional_encoding = PositionalEncoding(dim_model, dropout, max_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, dim_feedforward=dim_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_fc = nn.Linear(dim_model, n_classes)

    def forward(self, x):
        # x: (B, 129, T) → (B, T, 129)
        x = x.permute(0, 2, 1)

        x = self.input_fc(x)                # (B, T, dim_model)
        x = self.positional_encoding(x)     # 加上位置编码

        x = self.transformer_encoder(x)     # (B, T, dim_model)

        x = x.mean(dim=1)                   # 全局平均池化 → (B, dim_model)

        logits = self.output_fc(x)          # (B, n_classes)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 生成 (max_len, d_model) 的位置编码矩阵
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)      # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)      # 奇数维
        pe = pe.unsqueeze(0)                              # (1, max_len, d_model)

        self.register_buffer('pe', pe)  # 不作为参数参与训练

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]  # 加上前 T 个位置的编码
        return self.dropout(x)
