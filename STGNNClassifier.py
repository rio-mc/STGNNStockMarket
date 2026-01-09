import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv
from typing import Optional
from SharedHead import SharedClassifierHead

class STBlock(nn.Module):
    def __init__(self, in_channels, tcn_channels, gcn_hidden, tcn_kernel, dropout):
        # === STEP 1: Initialisation ===
        # ------------------------------------

        #   1. Parent abstract class initialisation
        super().__init__()

        #   2. Temporal convolutions
        self.conv1 = nn.Conv1d(in_channels,   tcn_channels, kernel_size=tcn_kernel, padding=tcn_kernel // 2)
        self.conv2 = nn.Conv1d(tcn_channels,  gcn_hidden,   kernel_size=tcn_kernel, padding=tcn_kernel // 2)

        #   3. Normalisation
        self.norm_t = nn.LayerNorm(gcn_hidden)

        #   4. Edge MLP for NNConv (edge-conditioned weights)
        nn_edge_mlp = nn.Sequential(
            nn.Linear(1, gcn_hidden),
            nn.ReLU(),
            nn.Linear(gcn_hidden, gcn_hidden * gcn_hidden)
        )

        #   5. Graph convolutions (edge-conditioned)
        self.gcn = NNConv(
            in_channels=gcn_hidden,
            out_channels=gcn_hidden,
            nn=nn_edge_mlp,
            aggr='mean'
        )
        self.norm_g  = nn.LayerNorm(gcn_hidden)
        self.dropout = nn.Dropout(dropout)

        #   6. Weight normalisation
        self._init_weights()

    def forward(
        self,
        h_seq: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # === STEP 1: Temporal Convolutions ===
        # ------------------------------------
        x = h_seq.permute(0, 2, 1)         # [B*N, F, T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(0, 2, 1)             # [B*N, T, gcn_hidden]

        # === STEP 2: Temporal pooling + norm ===
        # ------------------------------------ 
        x = x.mean(dim=1)                  # [B*N, gcn_hidden]
        x = self.norm_t(x)

		# === STEP 3: Graph conv + residual + norm + dropout ===
        # ------------------------------------
        h_spat = self.gcn(x, edge_index, edge_attr)
        h_spat = self.norm_g(h_spat + x)
        h_spat = F.relu(h_spat)
        h_spat = self.dropout(h_spat)

		# === STEP 4: Return with time dimensions ===
        # ------------------------------------
        return h_spat.unsqueeze(1)         # [B*N, 1, gcn_hidden]
    
    def _init_weights(self):
        # ====================================
		# === Helper to initialise layer weights

        #   1. Conv1d (ReLU downstream) -> Kaiming/He uniform
        for conv in (self.conv1, self.conv2):
            nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

        #   2. LayerNorm -> weight=1, bias=0
        for ln in (self.norm_t, self.norm_g):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

        #   3. Edge MLP inside NNConv -> Linear layers Glorot; biases 0
        for m in self.gcn.nn.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class STGNNClassifier(nn.Module):
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
        head_hidden: int = 128
    ):
        # === STEP 1: Initialisation ===
        # ------------------------------------

        #   1. Parent abstract class initialisation
        super().__init__()

        #   2. Model parameters
        self.feature_dim = feature_dim
        self.gcn_hidden  = gcn_hidden

        #   3. Graph parameters
        self.edge_index  = edge_index
        self.num_nodes   = num_nodes

        # === STEP 2: Stacked ST blocks ===
        # ------------------------------------
        self.blocks = nn.ModuleList()
        for i in range(stgnn_blocks):
            in_ch = feature_dim if i == 0 else gcn_hidden
            self.blocks.append(
                STBlock(
                    in_channels=in_ch,
                    tcn_channels=tcn_channels,
                    gcn_hidden=gcn_hidden,
                    tcn_kernel=tcn_kernel,
                    dropout=dropout
                )
            )

        # === STEP 3: target + context -> logits (shared head) ===
        # ------------------------------------
        self.head_norm = nn.LayerNorm(gcn_hidden * 2)   # keep: stabilizes combined vector
        self.rep_proj  = nn.Linear(gcn_hidden * 2, rep_dim)
        self.rep_norm  = nn.LayerNorm(rep_dim)

        # === Shared classifier head (identical across all models) ===
        self.classifier = SharedClassifierHead(
            in_dim=rep_dim,
            base_hidden=head_hidden,
            out_channels=out_dim,
            dropout=dropout
        )

        self.bottleneck = nn.Linear(gcn_hidden, 3)  # used in embed()
        self.norm      = nn.LayerNorm(gcn_hidden)
        self.dropout   = nn.Dropout(dropout)

		# === STEP 4: Initialise weights ===
        # ------------------------------------
        self._init_weights()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.LongTensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        target_node_index: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass for STGNN with neighbour-only context aggregation.

        Args:
            x: [B, N, T, F] input tensor
            edge_index: Graph connectivity [2, E] (optional, defaults to self.edge_index)
            edge_attr: Edge features [E, *] or None
            target_node_index: Index of node to classify
        """
        # 1. Validate shapes
        B, N, T, F_in = x.shape
        if N != self.num_nodes:
            raise ValueError(f"[STGNN] Expected {self.num_nodes} nodes, got {N}")

        # 2. Fallback to stored graph if not provided
        if edge_index is None:
            edge_index = self.edge_index

        # 3. Node-wise sequences: [B*N, T, F]
        h = x.permute(0, 1, 3, 2).reshape(B * N, T, F_in)

        # 4. ST blocks (+ residual if shapes match)
        for block in self.blocks:
            h_new = block(h, edge_index, edge_attr)
            h = h + h_new if h_new.shape == h.shape else h_new

        # 5. Aggregate over time → [B, N, H]
        h_final = h.mean(dim=1).view(B, N, self.gcn_hidden)

        # 6. Target/context split
        if target_node_index is None:
            raise ValueError("Must supply target_node_index")

        target_h = h_final[:, target_node_index, :]  # [B, H]

        # --- Neighbour-only context aggregation ---
        src, dst = edge_index
        mask = (src == target_node_index) | (dst == target_node_index)
        neigh_idx = torch.unique(
            torch.cat([
                src[mask], dst[mask],
                torch.tensor([target_node_index], device=edge_index.device)  # include self
            ])
        )
        # context_h = h_final.index_select(1, neigh_idx).mean(dim=1)  # [B, H]

        # Optional: attention-based context
        q = target_h.unsqueeze(1)                                  # [B, 1, H]
        K = h_final.index_select(1, neigh_idx)                     # [B, M, H]
        att = torch.softmax((q @ K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=-1)
        context_h = (att @ K).squeeze(1)                           # [B, H]

        # 7. Combine target + context, classify
        combined = torch.cat([target_h, context_h], dim=-1)  # [B, 2H]
        combined = self.head_norm(combined)
        rep = self.rep_norm(self.rep_proj(combined))         # [B, rep_dim]
        rep = self.dropout(rep)
        logits = self.classifier(rep)
        return logits

    def embed(self, x: torch.Tensor, edge_index: torch.LongTensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ====================================
		# === Helper to produce embedding at evaluation time
        self.eval()
        with torch.no_grad():
            B, N, T, F_in = x.shape
            h = x.permute(0, 1, 3, 2).reshape(B * N, T, F_in)
            for block in self.blocks:
                h = block(h, edge_index, edge_attr)
            h_final = h.mean(dim=1).view(B, N, self.gcn_hidden)
            return self.bottleneck(self.norm(h_final))

    def _init_weights(self):
        # ====================================
		# === Helper to initialise layer weights

        #   1. Linear layers -> Glorot; biases 0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # (STBlock initialises its own convs and NNConv edge MLP)