from __future__ import annotations

from architectures.components.static_graph_baseline import StaticGraphBaseline


class GraphSAGEGraphClassifier(StaticGraphBaseline):
    """GraphSAGE static graph baseline using the shared graph-baseline skeleton."""

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
    ):
        super().__init__(
            edge_index=edge_index,
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            graph_operator="graphsage",
            gcn_hidden=gcn_hidden,
            out_dim=out_dim,
            dropout=dropout,
            rep_dim=rep_dim,
            head_hidden=head_hidden,
        )
