import torch.nn as nn
import torch

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.net(x)
