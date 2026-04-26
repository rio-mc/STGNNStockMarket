import torch
import torch.nn as nn
from architectures.shared_head import SharedClassifierHead


class PanelGRUClassifier(nn.Module):
    """
    Multi-asset temporal model without graph structure.

    Input:
        x: [B, N, T, F]

    Process:
        - GRU per node
        - target node representation only
        - classify
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        rep_dim: int = 128,
        head_hidden: int = 128,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        enc_dim = hidden_dim * self.num_directions

        self.norm = nn.LayerNorm(enc_dim)

        self.rep_proj = nn.Linear(enc_dim, rep_dim)
        self.rep_norm = nn.LayerNorm(rep_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = SharedClassifierHead(
            in_dim=rep_dim,
            base_hidden=head_hidden,
            out_channels=1,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index=None,
        edge_attr=None,
        target_node_index=None,
    ) -> torch.Tensor:
        # x: [B, N, T, F]
        B, N, T, F = x.shape

        x_flat = x.contiguous().view(B * N, T, F)

        out, _ = self.gru(x_flat)
        last = out[:, -1, :]
        last = self.norm(last)

        h = last.view(B, N, -1)  # [B, N, H]

        if target_node_index is None:
            # fallback: infer target from final target-flag channel
            target_scores = x[:, :, :, -1].mean(dim=-1)  # [B, N]
            target_idx = int(target_scores[0].argmax().item())
        else:
            target_idx = int(target_node_index)

        target_h = h[:, target_idx, :]  # [B, H]

        rep = self.rep_norm(self.rep_proj(target_h))
        logits = self.classifier(rep)
        return logits