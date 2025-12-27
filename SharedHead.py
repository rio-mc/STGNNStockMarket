import torch.nn as nn

class SharedClassifierHead(nn.Module):
    """
    Same head used by every model (LSTM/GRU/STGNN):
    Linear(in -> 2*base_hidden) -> ReLU -> Dropout
    Linear(2*base_hidden -> base_hidden) -> ReLU -> Dropout
    Linear(base_hidden -> out)
    """
    def __init__(self, in_dim: int, base_hidden: int, out_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, base_hidden * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden * 2, base_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_hidden, out_channels),
        )

    def forward(self, x):
        return self.net(x)
