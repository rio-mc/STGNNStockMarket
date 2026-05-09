from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from architectures.shared_head import SharedClassifierHead


class GATGraphClassifier(nn.Module):
    """
    Static GAT graph baseline.

    Composition:
        mean temporal pooling
        -> two GATConv graph layers
        -> target-node readout
        -> shared classifier head

    Expected input:
        x: [batch_size, num_nodes, seq_len, feature_dim]
        edge_index: PyG edge index over batched graph copies
    """

    def __init__(
        self,
        edge_index,
        num_nodes: int,
        feature_dim: int,
        gcn_hidden: int,
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
        self.out_dim = int(out_dim)
        self.dropout = float(dropout)
        self.rep_dim = int(rep_dim)
        self.head_hidden = int(head_hidden)
        self.heads = int(heads)

        self.input_norm = nn.LayerNorm(self.feature_dim)

        self.gat1 = GATConv(
            self.feature_dim,
            self.gcn_hidden,
            heads=self.heads,
            concat=False,
            dropout=self.dropout,
        )

        self.gat2 = GATConv(
            self.gcn_hidden,
            self.rep_dim,
            heads=self.heads,
            concat=False,
            dropout=self.dropout,
        )

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        self.classifier = SharedClassifierHead(
            in_dim=self.rep_dim,
            base_hidden=self.head_hidden,
            out_channels=self.out_dim,
            dropout=self.dropout,
        )

        self.target_node_index: int | None = None

    def _resolve_target_positions(
        self,
        batch_size: int,
        num_nodes: int,
        target_node_index: int | None,
        device: torch.device,
    ) -> torch.Tensor:
        if target_node_index is None:
            raise ValueError("target_node_index must be provided for GATGraphClassifier")

        if target_node_index < 0 or target_node_index >= num_nodes:
            raise IndexError(
                f"target_node_index out of range: {target_node_index} "
                f"for num_nodes={num_nodes}"
            )

        offsets = torch.arange(batch_size, device=device, dtype=torch.long) * num_nodes
        return offsets + int(target_node_index)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        target_node_index: int | None = None,
    ) -> torch.Tensor:
        del edge_attr

        if x.dim() != 4:
            raise ValueError(
                f"Expected x with shape [batch, num_nodes, seq_len, feature_dim], "
                f"got {tuple(x.shape)}"
            )

        batch_size, num_nodes, _seq_len, feat_dim = x.shape

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
            raise ValueError("edge_index is required for GATGraphClassifier")

        if graph_edge_index.dim() != 2 or graph_edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(graph_edge_index.shape)}"
            )

        node_feat = x.mean(dim=2)
        node_feat = node_feat.reshape(batch_size * num_nodes, feat_dim)
        node_feat = self.input_norm(node_feat)

        h = self.gat1(node_feat, graph_edge_index)
        h = self.activation(h)
        h = self.drop(h)

        h = self.gat2(h, graph_edge_index)
        h = self.activation(h)
        h = self.drop(h)

        target_positions = self._resolve_target_positions(
            batch_size=batch_size,
            num_nodes=num_nodes,
            target_node_index=(
                target_node_index
                if target_node_index is not None
                else self.target_node_index
            ),
            device=h.device,
        )

        target_repr = h[target_positions]
        return self.classifier(target_repr)