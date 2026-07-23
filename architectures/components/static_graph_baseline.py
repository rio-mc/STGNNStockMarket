from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from architectures.components.readouts import NeighbourAttentionReadout
from architectures.components.temporal_encoders import MeanTemporalEncoder
from architectures.shared_head import SharedClassifierHead


class StaticGraphBaseline(nn.Module):
    """
    Shared non-recurrent graph baseline skeleton.

    GCN, GAT, and GraphSAGE should differ only by the message-passing
    operator. Keeping the encoder, readout, representation adapter, and
    classifier head identical makes model-suite comparisons defensible.
    """

    def __init__(
        self,
        *,
        edge_index: torch.LongTensor,
        num_nodes: int,
        feature_dim: int,
        graph_operator: str,
        gcn_hidden: int = 32,
        out_dim: int = 1,
        dropout: float = 0.15,
        rep_dim: int = 128,
        head_hidden: int = 128,
        heads: int = 1,
    ) -> None:
        super().__init__()

        self.edge_index = edge_index
        self.num_nodes = int(num_nodes)
        self.feature_dim = int(feature_dim)
        self.gcn_hidden = int(gcn_hidden)
        self.rep_dim = int(rep_dim)
        self.graph_operator = str(graph_operator).strip().lower()

        self.temporal = MeanTemporalEncoder()
        self.input_proj = nn.Linear(self.feature_dim, self.gcn_hidden)
        self.input_norm = nn.LayerNorm(self.gcn_hidden)

        if self.graph_operator == "gcn":
            self.graph = GCNConv(self.gcn_hidden, self.gcn_hidden)
        elif self.graph_operator == "gat":
            self.graph = GATConv(
                self.gcn_hidden,
                self.gcn_hidden,
                heads=int(heads),
                concat=False,
                dropout=float(dropout),
            )
        elif self.graph_operator == "graphsage":
            self.graph = SAGEConv(self.gcn_hidden, self.gcn_hidden, aggr="mean")
        else:
            raise ValueError(
                "graph_operator must be one of: gcn, gat, graphsage"
            )

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(float(dropout))
        self.readout = NeighbourAttentionReadout()

        self.head_norm = nn.LayerNorm(self.gcn_hidden * 2)
        self.rep_proj = nn.Linear(self.gcn_hidden * 2, self.rep_dim)
        self.rep_norm = nn.LayerNorm(self.rep_dim)

        self.classifier = SharedClassifierHead(
            in_dim=self.rep_dim,
            base_hidden=head_hidden,
            out_channels=out_dim,
            dropout=dropout,
        )

        self.target_node_index: int | None = None
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        target_node_index: Optional[int] = None,
    ) -> torch.Tensor:
        del edge_attr

        if x.dim() != 4:
            raise ValueError(
                f"Expected x with shape [batch, num_nodes, seq_len, feature_dim], got {tuple(x.shape)}"
            )

        batch_size, num_nodes, _seq_len, feat_dim = x.shape
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_nodes != self.num_nodes:
            raise ValueError(
                f"num_nodes mismatch: got {num_nodes}, expected {self.num_nodes}"
            )
        if feat_dim != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: got {feat_dim}, expected {self.feature_dim}"
            )

        graph_edge_index = edge_index if edge_index is not None else self.edge_index
        if graph_edge_index is None:
            raise ValueError(f"edge_index is required for {self.__class__.__name__}")
        if graph_edge_index.dim() != 2 or graph_edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(graph_edge_index.shape)}"
            )

        resolved_target = (
            target_node_index
            if target_node_index is not None
            else self.target_node_index
        )
        if resolved_target is None:
            raise ValueError(
                f"target_node_index must be provided for {self.__class__.__name__}"
            )
        if int(resolved_target) < 0 or int(resolved_target) >= num_nodes:
            raise IndexError(
                f"target_node_index out of range: {resolved_target} for num_nodes={num_nodes}"
            )

        h = self.temporal(x)
        h = self.input_norm(self.input_proj(h))

        h_flat = h.reshape(batch_size * num_nodes, self.gcn_hidden)
        h_flat = self.graph(h_flat, graph_edge_index)
        h_flat = self.activation(h_flat)
        h_flat = self.drop(h_flat)
        h = h_flat.reshape(batch_size, num_nodes, self.gcn_hidden)

        combined = self.readout(
            h=h,
            edge_index=graph_edge_index,
            target_node_index=int(resolved_target),
        )

        combined = self.head_norm(combined)
        rep = self.rep_norm(self.rep_proj(combined))
        rep = self.drop(rep)
        return self.classifier(rep)

    def _init_weights(self) -> None:
        for m in (self.input_proj, self.rep_proj):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for ln in (self.input_norm, self.head_norm, self.rep_norm):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
