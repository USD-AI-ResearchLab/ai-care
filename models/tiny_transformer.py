import torch
import torch.nn as nn

class TinyTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model=32,
        n_heads=2,
        layers=2,
        num_classes=10,
        seq_len=1
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # Project flat input to transformer space
        self.input_proj = nn.Linear(input_dim, d_model * seq_len)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True,
            dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.flatten(1)                       # (B, input_dim)
        x = self.input_proj(x)                 # (B, d_model * seq_len)
        x = x.view(x.size(0), self.seq_len, -1)  # (B, seq_len, d_model)
        out = self.encoder(x)                  # (B, seq_len, d_model)
        return self.fc(out.mean(dim=1))         # (B, num_classes)
