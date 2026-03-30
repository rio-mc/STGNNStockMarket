import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Optional


class SharedClassifierHead(nn.Module):
    """
    Shared classifier head used by every model.
    """

    def __init__(
        self,
        in_dim: int,
        out_channels: int = 1,
        dropout: float = 0.0,
        base_hidden: Optional[int] = None,
        use_weight_norm: bool = True,
        temperature: float = 1.0,
    ):
        super().__init__()
        do = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        def lin(a: int, b: int) -> nn.Module:
            layer = nn.Linear(a, b)
            return weight_norm(layer) if use_weight_norm else layer

        if base_hidden is not None and int(base_hidden) > 0:
            h = int(base_hidden)
            self.net = nn.Sequential(
                do,
                lin(in_dim, h),
                nn.ReLU(),
                do,
                lin(h, out_channels),
            )
        else:
            self.net = nn.Sequential(
                do,
                lin(in_dim, out_channels),
            )

        self.register_buffer("T", torch.tensor(float(temperature)))

    def set_temperature(self, temperature: float):
        self.T.fill_(float(temperature))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        T = self.T.clamp(1e-3, 100.0)
        return logits / T