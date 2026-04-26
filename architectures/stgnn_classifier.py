import torch
import torch.nn as nn
from typing import Optional

from torch_geometric.nn import GATConv, GCNConv, SAGEConv

from .shared_head import SharedClassifierHead
from .components.temporal_encoders import TCNTemporalEncoder
from .components.graph_layers import EdgeConditionedNNConvLayer
from .components.readouts import NeighbourAttentionReadout


class GraphOperatorLayer(nn.Module):
    """
    Batched graph operator wrapper.

    Input:
        x: [B, N, H]
        edge_index: batched edge index over B copies of the graph
        edge_attr: optional edge features, used only for nnconv
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        graph_model: str = "gcn",
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.graph_model = str(graph_model).strip().lower()

        if self.graph_model == "nnconv":
            self.op = EdgeConditionedNNConvLayer(
                hidden_dim=hidden_dim,
                dropout=dropout,
                aggr="mean",
            )
        elif self.graph_model == "gcn":
            self.op = GCNConv(hidden_dim, hidden_dim)
        elif self.graph_model == "graphsage":
            self.op = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        elif self.graph_model == "gat":
            self.op = GATConv(
                hidden_dim,
                hidden_dim,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported graph_model='{graph_model}'. "
                f"Expected one of: gcn, graphsage, gat, nnconv."
            )

        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, N, H], got {tuple(x.shape)}")

        batch_size, num_nodes, hidden_dim = x.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Hidden dim mismatch: got {hidden_dim}, expected {self.hidden_dim}"
            )

        x_flat = x.reshape(batch_size * num_nodes, hidden_dim)

        if self.graph_model == "nnconv":
            h = self.op(x, edge_index, edge_attr)
        else:
            h = self.op(x_flat, edge_index)
            h = h.reshape(batch_size, num_nodes, hidden_dim)

        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class STBlock(nn.Module):
    """
    One reusable spatio-temporal block:

        TCN temporal encoder -> selectable graph layer

    Input:
        x: [B, N, T, F] for first block
        or [B, N, 1, H] for later blocks

    Output:
        h_seq: [B, N, 1, H]
    """

    def __init__(
        self,
        in_channels: int,
        tcn_channels: int,
        gcn_hidden: int,
        tcn_kernel: int,
        dropout: float,
        graph_model: str = "gcn",
    ):
        super().__init__()

        self.temporal = TCNTemporalEncoder(
            in_channels=in_channels,
            tcn_channels=tcn_channels,
            hidden_dim=gcn_hidden,
            kernel_size=tcn_kernel,
        )

        self.graph = GraphOperatorLayer(
            hidden_dim=gcn_hidden,
            dropout=dropout,
            graph_model=graph_model,
        )

    def forward(
        self,
        x_seq: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.temporal(x_seq)
        h = self.graph(h, edge_index, edge_attr)
        return h.unsqueeze(2)


class STGNNClassifier(nn.Module):
    """
    Full spatio-temporal graph model.

    Composition:
        stacked STBlocks
        -> neighbour attention readout
        -> shared classifier head
    """

    def __init__(
        self,
        edge_index: torch.LongTensor,
        num_nodes: int,
        feature_dim: int,
        tcn_channels: int = 32,
        tcn_kernel: int = 3,
        gcn_hidden: int = 32,
        stgnn_blocks: int = 2,
        out_dim: int = 1,
        dropout: float = 0.3,
        rep_dim: int = 128,
        head_hidden: int = 128,
        graph_model: str = "gcn",
    ):
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.gcn_hidden = int(gcn_hidden)
        self.edge_index = edge_index
        self.num_nodes = int(num_nodes)
        self.graph_model = str(graph_model).strip().lower()

        self.blocks = nn.ModuleList()
        for i in range(stgnn_blocks):
            in_ch = feature_dim if i == 0 else gcn_hidden
            self.blocks.append(
                STBlock(
                    in_channels=in_ch,
                    tcn_channels=tcn_channels,
                    gcn_hidden=gcn_hidden,
                    tcn_kernel=tcn_kernel,
                    dropout=dropout,
                    graph_model=self.graph_model,
                )
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

        self.bottleneck = nn.Linear(gcn_hidden, 3)
        self.embed_norm = nn.LayerNorm(gcn_hidden)

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        target_node_index: Optional[int] = None,
    ) -> torch.Tensor:
        _batch_size, num_nodes, _seq_len, _feat_dim = x.shape

        if num_nodes != self.num_nodes:
            raise ValueError(f"[STGNN] Expected {self.num_nodes} nodes, got {num_nodes}")

        if target_node_index is None:
            raise ValueError("Must supply target_node_index")

        if edge_index is None:
            edge_index = self.edge_index

        h_seq = x
        for block in self.blocks:
            h_new = block(h_seq, edge_index=edge_index, edge_attr=edge_attr)
            h_seq = h_seq + h_new if h_seq.shape == h_new.shape else h_new

        h_final = h_seq.squeeze(2)

        combined = self.readout(
            h=h_final,
            edge_index=edge_index,
            target_node_index=int(target_node_index),
        )

        combined = self.head_norm(combined)
        rep = self.rep_norm(self.rep_proj(combined))
        rep = self.dropout(rep)

        return self.classifier(rep)

    def embed(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            h_seq = x
            for block in self.blocks:
                h_seq = block(h_seq, edge_index=edge_index, edge_attr=edge_attr)

            h_final = h_seq.squeeze(2)
            return self.bottleneck(self.embed_norm(h_final))

    def _init_weights(self):
        for m in (self.rep_proj, self.bottleneck):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        for ln in (self.head_norm, self.rep_norm, self.embed_norm):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
