from typing import Optional

import torch
import torch.nn as nn


class TargetNodeReadout(nn.Module):
    """
    Selects the target node representation.

    Input:
        h: [B, N, H]

    Output:
        target_h: [B, H]
    """

    def forward(
        self,
        h: torch.Tensor,
        target_node_index: Optional[int] = None,
        x_with_target_flag: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if target_node_index is None:
            if x_with_target_flag is None:
                raise ValueError("target_node_index is required unless x_with_target_flag is supplied")

            # Target flag is expected to be the final feature channel.
            # x_with_target_flag: [B, N, T, F]
            target_scores = x_with_target_flag[:, :, :, -1].mean(dim=-1)  # [B, N]
            target_node_index = int(target_scores[0].argmax().item())

        return h[:, int(target_node_index), :]


class NeighbourAttentionReadout(nn.Module):
    """
    Target-conditioned attention over the target node's graph neighbourhood.

    Input:
        h: [B, N, H]
        edge_index: [2, E]

    Output:
        combined: [B, 2H], concatenated target and neighbourhood context.

    Notes:
        - Includes the target node in its own neighbourhood.
        - If edge_index is empty, falls back to target-only context.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        h: torch.Tensor,
        edge_index: Optional[torch.LongTensor],
        target_node_index: int,
    ) -> torch.Tensor:
        B, N, H = h.shape
        target_idx = int(target_node_index)

        target_h = h[:, target_idx, :]

        if edge_index is None or edge_index.numel() == 0:
            neigh_idx = torch.tensor(
                [target_idx],
                dtype=torch.long,
                device=h.device,
            )
        else:
            edge_index = edge_index.to(h.device)

            # PyG batches graph edge indices with node offsets.
            # Convert back to local node ids so we can index h[:, node, :].
            local_edge_index = edge_index % N

            src, dst = local_edge_index
            mask = (src == target_idx) | (dst == target_idx)

            neigh_idx = torch.unique(
                torch.cat(
                    [
                        src[mask],
                        dst[mask],
                        torch.tensor([target_idx], dtype=torch.long, device=h.device),
                    ]
                )
            )

        neighbours = h.index_select(1, neigh_idx)

        q = target_h.unsqueeze(1)
        scores = (q @ neighbours.transpose(1, 2)) / (H ** 0.5)
        att = torch.softmax(scores, dim=-1)
        context_h = (att @ neighbours).squeeze(1)

        return torch.cat([target_h, context_h], dim=-1)