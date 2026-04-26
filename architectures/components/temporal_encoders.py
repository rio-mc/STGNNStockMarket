import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanTemporalEncoder(nn.Module):
    """
    Non-parametric temporal reduction.

    Input:
        x: [B, N, T, F]

    Output:
        h: [B, N, F]
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=2)


class TCNTemporalEncoder(nn.Module):
    """
    Lightweight temporal convolution encoder used by STGNN.

    Input:
        x: [B, N, T, F]

    Output:
        h: [B, N, H]
    """

    def __init__(
        self,
        in_channels: int,
        tcn_channels: int,
        hidden_dim: int,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            tcn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.conv2 = nn.Conv1d(
            tcn_channels,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_in = x.shape

        z = x.contiguous().view(B * N, T, F_in)
        z = z.permute(0, 2, 1)  # [B*N, F, T]

        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))

        z = z.permute(0, 2, 1)  # [B*N, T, H]
        z = z.mean(dim=1)       # [B*N, H]
        z = self.norm(z)

        return z.view(B, N, -1)

    def _init_weights(self):
        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)


class GRUTemporalEncoder(nn.Module):
    """
    Per-node GRU encoder for panel models.

    Input:
        x: [B, N, T, F]

    Output:
        h: [B, N, H]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.out_dim = hidden_dim * self.num_directions
        self.norm = nn.LayerNorm(self.out_dim)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_in = x.shape
        z = x.contiguous().view(B * N, T, F_in)
        out, _ = self.gru(z)
        last = self.norm(out[:, -1, :])
        return last.view(B, N, self.out_dim)

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)


class LSTMTemporalEncoder(nn.Module):
    """
    Per-node LSTM encoder for panel models.

    Input:
        x: [B, N, T, F]

    Output:
        h: [B, N, H]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.bidirectional = bool(bidirectional)
        self.num_directions = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        self.out_dim = hidden_dim * self.num_directions
        self.norm = nn.LayerNorm(self.out_dim)

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_in = x.shape
        z = x.contiguous().view(B * N, T, F_in)
        out, _ = self.lstm(z)
        last = self.norm(out[:, -1, :])
        return last.view(B, N, self.out_dim)

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden_size = self.hidden_dim
                param.data[hidden_size:2 * hidden_size].fill_(1.0)
