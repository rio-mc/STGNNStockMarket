from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv


class EdgeConditionedNNConvLayer(nn.Module):
    """
    Reusable edge-conditioned graph layer.

    This is the graph operator used by:
        - NNConvGraphClassifier
        - STGNNClassifier

    Input:
        h: [B, N, H]
        edge_index: [2, E]
        edge_attr: [E, 1] or None

    Output:
        h_out: [B, N, H]
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
        aggr: str = "mean",
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)

        edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

        self.conv = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=edge_mlp,
            aggr=aggr,
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, H = h.shape

        h_flat = h.contiguous().view(B * N, H)

        if edge_index is None:
            raise ValueError("edge_index is required for EdgeConditionedNNConvLayer")

        edge_index = edge_index.to(h.device)

        if edge_attr is None:
            edge_attr = torch.ones(
                (edge_index.size(1), 1),
                dtype=h.dtype,
                device=h.device,
            )
        else:
            edge_attr = edge_attr.to(h.device).to(dtype=h.dtype)

        h_spat = self.conv(h_flat, edge_index, edge_attr)
        h_spat = self.norm(h_spat + h_flat)
        h_spat = F.relu(h_spat)
        h_spat = self.dropout(h_spat)

        return h_spat.view(B, N, H)

    def _init_weights(self):
        for m in self.conv.nn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
