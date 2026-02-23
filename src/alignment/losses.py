"""
losses.py — Contrastive Alignment and Combined Loss Functions

Implements the loss functions for training the Syntax-BERT model:

1. AlignmentLoss (NT-Xent / InfoNCE):
   Aligns BERT sentence embeddings (h_B) with GNN graph embeddings (h_G).
   Positive pairs: (h_B_i, h_G_i) from the same sentence.
   Negative pairs: all other combinations in the batch.

2. GNNContrastiveLoss:
   Auxiliary self-supervised loss for the GNN branch to learn meaningful
   graph representations independently of the alignment objective.

3. CombinedLoss:
   Combines SimCSE loss, alignment loss, and GNN auxiliary loss:
   L = L_SimCSE + λ * L_align + μ * L_GNN
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AlignmentLoss(nn.Module):
    """NT-Xent (Normalized Temperature-scaled Cross-Entropy) contrastive loss
    for aligning BERT and GNN sentence representations.

    Given a batch of N sentences, we have N pairs (h_B_i, h_G_i).
    The loss encourages h_B_i to be similar to h_G_i (positive) and
    dissimilar to h_G_j for j ≠ i (negatives), and vice versa.

    This is symmetric: we compute the loss in both directions
    (BERT→GNN and GNN→BERT) and average them.

    Args:
        temperature: Scaling factor for cosine similarities. Lower values
                     make the distribution sharper (harder negatives).
    """

    def __init__(self, temperature: float = 0.05) -> None:
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_bert: Tensor, h_gnn: Tensor) -> Tensor:
        """Compute the symmetric NT-Xent alignment loss.

        Args:
            h_bert: (N, D) — BERT sentence embeddings (or projections).
            h_gnn:  (N, D) — GNN graph embeddings (or projections).

        Returns:
            Scalar loss value.
        """
        # L2-normalize embeddings
        h_bert = F.normalize(h_bert, p=2, dim=-1)
        h_gnn = F.normalize(h_gnn, p=2, dim=-1)

        batch_size = h_bert.size(0)
        labels = torch.arange(batch_size, device=h_bert.device)

        # Cosine similarity matrices scaled by temperature
        # logits_b2g[i, j] = sim(h_bert_i, h_gnn_j) / τ
        logits_b2g = torch.mm(h_bert, h_gnn.t()) / self.temperature
        logits_g2b = logits_b2g.t()

        # Symmetric cross-entropy: BERT→GNN + GNN→BERT
        loss_b2g = self.criterion(logits_b2g, labels)
        loss_g2b = self.criterion(logits_g2b, labels)

        return (loss_b2g + loss_g2b) / 2.0


class GNNContrastiveLoss(nn.Module):
    """Auxiliary contrastive loss for the GNN branch.

    Encourages the GNN to learn meaningful syntactic representations
    by contrasting graph embeddings of different sentences. Uses the
    same underlying NT-Xent formulation but with augmented graph views.

    For dependency trees, augmentation strategies include:
    - Edge dropout: randomly remove a fraction of dependency edges
    - Node feature noise: add Gaussian noise to BERT embeddings
    - Subgraph sampling: use a random subtree

    In the simplest form (no augmentation), this reduces to ensuring
    GNN embeddings are discriminative across different sentences.

    Args:
        temperature: Scaling factor for similarities.
        edge_drop_rate: Fraction of edges to randomly drop for augmentation.
    """

    def __init__(self, temperature: float = 0.1, edge_drop_rate: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.edge_drop_rate = edge_drop_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, h_gnn_1: Tensor, h_gnn_2: Tensor) -> Tensor:
        """Compute contrastive loss between two GNN views.

        Args:
            h_gnn_1: (N, D) — GNN embeddings from view 1 (original or augmented).
            h_gnn_2: (N, D) — GNN embeddings from view 2 (augmented differently).

        Returns:
            Scalar loss value.
        """
        h_gnn_1 = F.normalize(h_gnn_1, p=2, dim=-1)
        h_gnn_2 = F.normalize(h_gnn_2, p=2, dim=-1)

        batch_size = h_gnn_1.size(0)
        labels = torch.arange(batch_size, device=h_gnn_1.device)

        logits_12 = torch.mm(h_gnn_1, h_gnn_2.t()) / self.temperature
        logits_21 = logits_12.t()

        loss_12 = self.criterion(logits_12, labels)
        loss_21 = self.criterion(logits_21, labels)

        return (loss_12 + loss_21) / 2.0

    @staticmethod
    def drop_edges(edge_index: Tensor, drop_rate: float) -> Tensor:
        """Randomly drop edges from a graph for augmentation.

        Args:
            edge_index: (2, E) — edge connectivity tensor.
            drop_rate: Fraction of edges to drop.

        Returns:
            Filtered edge_index with (1 - drop_rate) * E edges remaining.
        """
        if drop_rate <= 0.0:
            return edge_index
        num_edges = edge_index.size(1)
        keep_mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
        return edge_index[:, keep_mask]


class CombinedLoss(nn.Module):
    """Combined training objective.

    L = L_SimCSE + λ * L_align + μ * L_GNN

    Where:
    - L_SimCSE: SimCSE's contrastive loss (computed inside BertForCL.forward)
    - L_align: NT-Xent alignment loss between h_B and h_G
    - L_GNN: Auxiliary GNN contrastive loss

    Args:
        lambda_align: Weight for alignment loss.
        mu_gnn: Weight for GNN auxiliary loss.
        align_temperature: Temperature for alignment NT-Xent.
        gnn_temperature: Temperature for GNN contrastive loss.
    """

    def __init__(
        self,
        lambda_align: float = 0.1,
        mu_gnn: float = 0.05,
        align_temperature: float = 0.05,
        gnn_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_align = lambda_align
        self.mu_gnn = mu_gnn

        self.alignment_loss = AlignmentLoss(temperature=align_temperature)
        self.gnn_loss = GNNContrastiveLoss(temperature=gnn_temperature)

    def forward(
        self,
        loss_simcse: Tensor,
        h_bert: Tensor,
        h_gnn: Tensor,
        h_bert_proj: Tensor | None = None,
        h_gnn_proj: Tensor | None = None,
        h_gnn_aug: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the combined loss.

        Args:
            loss_simcse: Scalar — SimCSE contrastive loss (from BertForCL).
            h_bert: (N, D) — BERT sentence embeddings.
            h_gnn: (N, D) — GNN graph embeddings.
            h_bert_proj: (N, P) — projected BERT embeddings (optional).
            h_gnn_proj: (N, P) — projected GNN embeddings (optional).
            h_gnn_aug: (N, D) — augmented GNN embeddings for auxiliary loss (optional).

        Returns:
            Dict with keys: "total", "simcse", "alignment", "gnn"
        """
        # Use projected embeddings for alignment if available
        align_bert = h_bert_proj if h_bert_proj is not None else h_bert
        align_gnn = h_gnn_proj if h_gnn_proj is not None else h_gnn

        # Alignment loss
        loss_align = self.alignment_loss(align_bert, align_gnn)

        # GNN auxiliary loss (only if augmented view is provided)
        if h_gnn_aug is not None and self.mu_gnn > 0:
            loss_gnn = self.gnn_loss(h_gnn, h_gnn_aug)
        else:
            loss_gnn = torch.tensor(0.0, device=loss_simcse.device)

        # Combined loss
        total = loss_simcse + self.lambda_align * loss_align + self.mu_gnn * loss_gnn

        return {
            "total": total,
            "simcse": loss_simcse.detach(),
            "alignment": loss_align.detach(),
            "gnn": loss_gnn.detach(),
        }
