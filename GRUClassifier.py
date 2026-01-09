import torch
import torch.nn as nn
from SharedHead import SharedClassifierHead

class GRUClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        out_channels: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3,
        rep_dim: int = 128,
        head_hidden: int = 128
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # === STEP 1: GRU ===
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # === STEP 2: Normalisation + Classifier ===
        enc_dim = hidden_dim * self.num_directions
        self.norm = nn.LayerNorm(enc_dim)

        self.rep_proj = nn.Linear(enc_dim, rep_dim)
        self.rep_norm = nn.LayerNorm(rep_dim)

        self.classifier = SharedClassifierHead(
            in_dim=rep_dim,
            base_hidden=head_hidden,
            out_channels=out_channels,
            dropout=dropout
        )

        # === STEP 3: Weight Initialisation ===
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, feature_dim]
        output, _ = self.gru(x)
        last_hidden = output[:, -1, :]
        last_hidden = self.norm(last_hidden)

        rep = self.rep_norm(self.rep_proj(last_hidden))
        logits = self.classifier(rep)
        
        return logits

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
