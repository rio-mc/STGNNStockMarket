import torch
import torch.nn as nn

from architectures.shared_head import SharedClassifierHead


class PanelLSTMClassifier(nn.Module):
    """
    Multi-asset temporal model without graph structure.

    Input:
        x: [B, N, T, F]

    Contract:
        - Encodes every node independently with an LSTM.
        - Uses only the target node representation for classification.
        - Does not use edge_index, edge_attr, neighbour pooling, or graph message passing.
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

        self.lstm = nn.LSTM(
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

        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        edge_index=None,
        edge_attr=None,
        target_node_index=None,
    ) -> torch.Tensor:
        B, N, T, F_in = x.shape

        x_flat = x.contiguous().view(B * N, T, F_in)
        out, _ = self.lstm(x_flat)

        last = out[:, -1, :]
        last = self.norm(last)

        h = last.view(B, N, -1)

        if target_node_index is None:
            target_scores = x[:, :, :, -1].mean(dim=-1)
            target_idx = int(target_scores[0].argmax().item())
        else:
            target_idx = int(target_node_index)

        target_h = h[:, target_idx, :]

        rep = self.rep_norm(self.rep_proj(target_h))
        rep = self.dropout(rep)

        return self.classifier(rep)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                hidden_size = self.hidden_dim
                param.data[hidden_size:2 * hidden_size].fill_(1.0)
