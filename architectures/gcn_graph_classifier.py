# file: architectures/gcn_graph_classifier.py
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNGraphClassifier(nn.Module):
    """
    Simple GCN baseline:
    - temporal mean-pooling over each node's lookback window
    - two GCN layers over the shared graph
    - classify the selected target node for each graph in batch

    Expected input:
    - x: [batch_size, num_nodes, seq_len, feature_dim]
    - edge_index: batched PyG edge index corresponding to batch_size copies
      of the same base graph
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

        self.input_norm = nn.LayerNorm(self.feature_dim)

        self.gcn1 = GCNConv(self.feature_dim, self.gcn_hidden)
        self.gcn2 = GCNConv(self.gcn_hidden, self.rep_dim)

        self.activation = nn.ReLU()
        self.drop = nn.Dropout(self.dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.rep_dim, self.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden, self.out_dim),
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
            raise ValueError("target_node_index must be provided for GCNGraphClassifier")
        if target_node_index < 0 or target_node_index >= num_nodes:
            raise IndexError(
                f"target_node_index out of range: {target_node_index} for num_nodes={num_nodes}"
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
                f"Expected x with shape [batch, num_nodes, seq_len, feature_dim], got {tuple(x.shape)}"
            )

        batch_size, num_nodes, _seq_len, feat_dim = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"num_nodes mismatch: got {num_nodes}, expected {self.num_nodes}")
        if feat_dim != self.feature_dim:
            raise ValueError(f"feature_dim mismatch: got {feat_dim}, expected {self.feature_dim}")

        graph_edge_index = edge_index if edge_index is not None else self.edge_index
        if graph_edge_index is None:
            raise ValueError("edge_index is required for GCNGraphClassifier")

        if graph_edge_index.dim() != 2 or graph_edge_index.size(0) != 2:
            raise ValueError(
                f"edge_index must have shape [2, E], got {tuple(graph_edge_index.shape)}"
            )

        node_feat = x.mean(dim=2)
        node_feat = node_feat.reshape(batch_size * num_nodes, feat_dim)
        node_feat = self.input_norm(node_feat)

        h = self.gcn1(node_feat, graph_edge_index)
        h = self.activation(h)
        h = self.drop(h)

        h = self.gcn2(h, graph_edge_index)
        h = self.activation(h)
        h = self.drop(h)

        target_positions = self._resolve_target_positions(
            batch_size=batch_size,
            num_nodes=num_nodes,
            target_node_index=target_node_index if target_node_index is not None else self.target_node_index,
            device=h.device,
        )
        target_repr = h[target_positions]
        logits = self.classifier(target_repr)
        return logits

