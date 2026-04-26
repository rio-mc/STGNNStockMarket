import torch
import torch.nn as nn
from typing import Optional

from architectures.shared_head import SharedClassifierHead
from architectures.components.temporal_encoders import MeanTemporalEncoder
from architectures.components.graph_layers import EdgeConditionedNNConvLayer
from architectures.components.readouts import NeighbourAttentionReadout


class NNConvGraphClassifier(nn.Module):
    """
    Non-recurrent graph baseline.

    Composition:
        mean temporal encoder
        -> edge-conditioned NNConv graph layer
        -> neighbour attention readout
        -> shared classifier head

    This deliberately shares the graph operator and readout used by STGNN,
    while removing the TCN temporal encoder.
    """

    def __init__(
        self,
        edge_index: torch.LongTensor,
        num_nodes: int,
        feature_dim: int,
        gcn_hidden: int = 32,
        out_dim: int = 1,
        dropout: float = 0.3,
        rep_dim: int = 128,
        head_hidden: int = 128,
    ):
        super().__init__()

        self.edge_index = edge_index
        self.num_nodes = int(num_nodes)
        self.feature_dim = int(feature_dim)
        self.gcn_hidden = int(gcn_hidden)

        self.temporal = MeanTemporalEncoder()

        self.input_proj = nn.Linear(feature_dim, gcn_hidden)
        self.input_norm = nn.LayerNorm(gcn_hidden)

        self.graph = EdgeConditionedNNConvLayer(
            hidden_dim=gcn_hidden,
            dropout=dropout,
            aggr="mean",
        )

        self.readout = NeighbourAttentionReadout()

        self.head_norm = nn.LayerNorm(gcn_hidden * 2)
        self.rep_proj = nn.Linear(gcn_hidden * 2, rep_dim)
        self.rep_norm = nn.LayerNorm(rep_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = SharedClassifierHead(
            in_dim=rep_dim,
            base_hidden=head_hidden,
            out_channels=out_dim,
            dropout=dropout,
        )

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        target_node_index: Optional[int] = None,
    ) -> torch.Tensor:
        B, N, T, F_in = x.shape

        if N != self.num_nodes:
            raise ValueError(f"[NNConvGraph] Expected {self.num_nodes} nodes, got {N}")

        if target_node_index is None:
            raise ValueError("NNConvGraphClassifier requires target_node_index")

        if edge_index is None:
            edge_index = self.edge_index

        h = self.temporal(x)                   # [B, N, F]
        h = self.input_norm(self.input_proj(h)) # [B, N, H]
        h = self.graph(h, edge_index, edge_attr)

        combined = self.readout(
            h=h,
            edge_index=edge_index,
            target_node_index=int(target_node_index),
        )

        combined = self.head_norm(combined)
        rep = self.rep_norm(self.rep_proj(combined))
        rep = self.dropout(rep)

        return self.classifier(rep)

    def _init_weights(self):
        for m in (self.input_proj, self.rep_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for ln in (self.input_norm, self.head_norm, self.rep_norm):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
