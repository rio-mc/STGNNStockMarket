import torch
import torch.nn as nn
from SharedHead import SharedClassifierHead

class LSTMClassifier(nn.Module):
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
        # === STEP 1: Initialisation ===
        # ------------------------------------

        #   1. Parent abstract class initialisation
        super().__init__()

        #   2. Model parameters
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # === STEP 2: Build LSTM ===
        # ------------------------------------
        
        #   1. LSTM layer
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # === STEP 3: Input Normalisation ===
        # ------------------------------------
        enc_dim = hidden_dim * self.num_directions
        self.norm = nn.LayerNorm(enc_dim)

        # === Shared representation adapter  ===
        self.rep_proj = nn.Linear(enc_dim, rep_dim)
        self.rep_norm = nn.LayerNorm(rep_dim)

        # === Shared classifier head ===
        self.classifier = SharedClassifierHead(
            in_dim=rep_dim,
            base_hidden=head_hidden,
            out_channels=out_channels,
            dropout=dropout
        )

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ====================================
		# === Helper to produce LSTM output

        #   1. LSTM encoding
        output, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*num_directions)

        #   2. Select last timestep
        last_hidden = output[:, -1, :]

        #   3. Normalise features
        last_hidden = self.norm(last_hidden)

        #   4. Classify
        rep = self.rep_norm(self.rep_proj(last_hidden))
        logits = self.classifier(rep)

        return logits

    def _init_weights(self):
        # ====================================
		# === Helper to initialise layer weights
        for m in self.modules():

            #   1. Glorot uniform
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            #   2. Layer normalisation
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        #   3. Handle parameters explicitly
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget-gate bias trick: set biases of forget gate to 1
                # Gate order in PyTorch LSTM: [i, f, g, o]
                hidden_size = self.hidden_dim
                param.data[hidden_size:2*hidden_size].fill_(1.0)