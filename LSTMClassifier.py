import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        out_channels: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.3
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

        #   2. Weight normalisation
        self._init_weights()

        # === STEP 3: Input Normalisation ===
        # ------------------------------------
        self.norm = nn.LayerNorm(hidden_dim * self.num_directions)

        # === STEP 4: LSTM Classifier ===
        # ------------------------------------
        
        #    1. Fully-connected projection & non-linearity stack
        #      Applies an expand–contract pattern ("hourglass") to increase feature capacity,
        #      then compress to retain only the most salient information before the final output.
        self.classifier = nn.Sequential(

            #   2. Project from LSTM/STGNN output size to a larger intermediate space
            #      Expanding hidden_dim → hidden_dim*2 gives the network room to form richer,
            #      more separable feature combinations before narrowing back down.
            nn.Linear(hidden_dim * self.num_directions, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            #   3. Reduce dimensionality while retaining non-linear capacity
            #      Forces the network to compress features into a smaller space,
            #      acting as a bottleneck to encourage generalisable representations.
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            #   4. Final projection to output space (e.g., 1 for binary classification)
            #      Converts the compressed representation to task-specific outputs.
            nn.Linear(hidden_dim, out_channels)
        )


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
        logits = self.classifier(last_hidden)
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