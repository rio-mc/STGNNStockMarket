import torch.nn as nn

class SharedClassifierHead(nn.Module):
    """Shared linear probe head used by every model."""
    def __init__(self, in_dim: int, out_channels: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_dim, out_channels)

    def forward(self, x):
        x = self.dropout(x)
        return self.fc(x)
