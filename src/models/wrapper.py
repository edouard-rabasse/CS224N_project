"""
wrapper.py — SyntaxBertModel: Wraps SimCSE's BertForCL + SyntaxGNNEncoder

This module is the core architectural contribution. It combines:
  1. BERT branch (h_B): SimCSE's contrastive learning encoder
  2. GNN branch (h_G): Syntactic dependency graph encoder

During training, both branches produce sentence embeddings that are aligned
via a contrastive loss. At inference time, only the BERT branch is used —
the GNN branch is discarded, since BERT has internalized syntactic structure.

Key design choices:
  - Node features for the GNN come from BERT's last hidden states, aligned
    to the syntactic parser's word-level tokenization (first-subword strategy).
  - An optional projection head maps both h_B and h_G to a shared space
    before computing the alignment loss.
  - Supports stop-gradient and freeze modes for ablation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch

from src.models.gnn_encoder import SyntaxGNNEncoder


@dataclass
class SyntaxBertOutput:
    """Output of the SyntaxBertModel forward pass."""

    h_bert: Tensor  # (batch_size, hidden_dim) — BERT sentence embedding
    h_gnn: Optional[Tensor] = None  # (batch_size, hidden_dim) — GNN graph embedding
    h_bert_proj: Optional[Tensor] = None  # (batch_size, proj_dim)  — projected BERT embedding
    h_gnn_proj: Optional[Tensor] = None  # (batch_size, proj_dim)  — projected GNN embedding
    bert_hidden_states: Optional[Tensor] = None  # (batch_size, seq_len, hidden_dim)
    simcse_loss: Optional[Tensor] = None  # scalar — SimCSE contrastive loss from BertForCL


class ProjectionHead(nn.Module):
    """MLP projection head for alignment (following SimCLR / BYOL pattern).

    Maps embeddings to a lower-dimensional space where the alignment contrastive
    loss is computed. This prevents the alignment objective from collapsing the
    representation space used for downstream tasks.
    """

    def __init__(self, in_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SyntaxBertModel(nn.Module):
    """Combined BERT + GNN model for syntax-aware sentence embeddings.

    Args:
        bert_model: A SimCSE BertForCL (or RobertaForCL) instance.
        gnn_config: Dict with GNN hyperparameters (from Hydra config).
        alignment_config: Dict with alignment hyperparameters.
        tokenizer: HuggingFace tokenizer (for subword-to-word alignment info).
    """

    def __init__(
        self,
        bert_model: nn.Module,
        gnn_config: dict,
        alignment_config: dict,
    ) -> None:
        super().__init__()

        self.bert = bert_model
        self.hidden_dim = self.bert.config.hidden_size

        # ---- GNN Branch ----
        self.gnn = SyntaxGNNEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=gnn_config.get("hidden_dim", self.hidden_dim),
            num_layers=gnn_config.get("num_layers", 2),
            conv_type=gnn_config.get("conv_type", "gat"),
            heads=gnn_config.get("heads", 4),
            dropout=gnn_config.get("dropout", 0.1),
            pooling=gnn_config.get("pooling", "mean"),
        )

        # ---- Projection heads for alignment ----
        proj_dim = alignment_config.get("projector_dim", 0)
        if proj_dim > 0:
            self.bert_projector = ProjectionHead(self.hidden_dim, proj_dim)
            self.gnn_projector = ProjectionHead(self.hidden_dim, proj_dim)
        else:
            self.bert_projector = None
            self.gnn_projector = None

        # ---- Strategy flags ----
        self.stop_grad_gnn = alignment_config.get("stop_grad_gnn", False)
        self.freeze_gnn = gnn_config.get("freeze", False)

        if self.freeze_gnn:
            self._freeze_gnn()

    def _freeze_gnn(self) -> None:
        """Freeze all GNN parameters."""
        for param in self.gnn.parameters():
            param.requires_grad = False

    def _freeze_bert(self) -> None:
        """Freeze all BERT parameters (used in phase 1 of freeze-then-align)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def _unfreeze_bert(self) -> None:
        """Unfreeze all BERT parameters (used in phase 2 of freeze-then-align)."""
        for param in self.bert.parameters():
            param.requires_grad = True

    def _unfreeze_gnn(self) -> None:
        """Unfreeze all GNN parameters."""
        for param in self.gnn.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        graph_batch: Optional[Batch] = None,
        token_to_word_maps: Optional[list[list[int]]] = None,
        sent_emb: bool = False,
        **bert_kwargs,
    ) -> SyntaxBertOutput:
        """Forward pass through BERT and (optionally) GNN branches.

        Args:
            input_ids: (batch_size, seq_len) — tokenized input IDs.
            attention_mask: (batch_size, seq_len) — attention mask.
            graph_batch: PyG Batch of dependency graphs. If None, GNN branch is skipped
                         (inference mode — only BERT is used).
            token_to_word_maps: List of lists mapping each BERT subword position to a
                                Stanza word index. Used to aggregate subword embeddings
                                into word-level features for the GNN. Length = batch_size.
            sent_emb: If True, run in inference mode (no GNN, no training-specific logic).
            **bert_kwargs: Additional kwargs for the BERT model (e.g., token_type_ids).

        Returns:
            SyntaxBertOutput with BERT and GNN embeddings.
        """
        # ---- BERT branch ----
        # SimCSE's forward expects sent_emb=True for inference
        if sent_emb or graph_batch is None:
            bert_out = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sent_emb=True,
                **bert_kwargs,
            )
            return SyntaxBertOutput(
                h_bert=bert_out.pooler_output,
                bert_hidden_states=None,
            )

        # Training mode: get BERT hidden states + SimCSE contrastive outputs
        # First, run BERT to get hidden states (we need these for GNN node features)
        bert_outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **bert_kwargs,
        )

        last_hidden = bert_outputs.last_hidden_state  # (batch_size * num_sent, seq_len, hidden_dim)

        # Pool to get h_B (using SimCSE's pooler)
        h_bert = self.bert.pooler(attention_mask, bert_outputs)
        if hasattr(self.bert, "mlp") and self.bert.model_args.mlp_only_train:
            h_bert = self.bert.mlp(h_bert)

        # ---- GNN branch ----
        # Populate GNN node features from BERT hidden states
        # We use the first sentence in each SimCSE pair (the original, not the augmented)
        h_gnn = self._compute_gnn_embeddings(last_hidden, graph_batch, token_to_word_maps)

        # Stop-gradient on GNN branch if configured
        if self.stop_grad_gnn:
            h_gnn = h_gnn.detach()

        # ---- Projections for alignment ----
        h_bert_proj = self.bert_projector(h_bert) if self.bert_projector is not None else None
        h_gnn_proj = self.gnn_projector(h_gnn) if self.gnn_projector is not None else None

        return SyntaxBertOutput(
            h_bert=h_bert,
            h_gnn=h_gnn,
            h_bert_proj=h_bert_proj,
            h_gnn_proj=h_gnn_proj,
            bert_hidden_states=last_hidden,
        )

    def _compute_gnn_embeddings(
        self,
        bert_hidden: Tensor,
        graph_batch: Batch,
        token_to_word_maps: Optional[list[list[int]]],
    ) -> Tensor:
        """Extract per-word BERT features, set as GNN node features, and encode.

        If token_to_word_maps is provided, aggregates subword embeddings by
        mean-pooling per word. Otherwise, assumes graph_batch.x is already set.

        Args:
            bert_hidden: (batch_total, seq_len, hidden_dim) — BERT last hidden states.
            graph_batch: PyG Batch with edge_index and batch vectors.
            token_to_word_maps: Subword → word alignment maps.

        Returns:
            h_G: (batch_size, hidden_dim) — graph-level embeddings.
        """
        if token_to_word_maps is not None:
            # Aggregate BERT subword embeddings → word-level features
            node_features_list = []
            for i, word_map in enumerate(token_to_word_maps):
                sent_hidden = bert_hidden[i]  # (seq_len, hidden_dim)
                num_words = max(word_map) + 1 if word_map else 0
                word_feats = torch.zeros(
                    num_words,
                    sent_hidden.size(-1),
                    device=sent_hidden.device,
                    dtype=sent_hidden.dtype,
                )
                word_counts = torch.zeros(num_words, device=sent_hidden.device)

                for subword_idx, word_idx in enumerate(word_map):
                    if word_idx >= 0:  # skip special tokens mapped to -1
                        word_feats[word_idx] += sent_hidden[subword_idx]
                        word_counts[word_idx] += 1

                # Mean-pool subwords per word
                word_counts = word_counts.clamp(min=1)
                word_feats = word_feats / word_counts.unsqueeze(-1)
                node_features_list.append(word_feats)

            # Replace graph_batch.x with aggregated word features
            graph_batch.x = torch.cat(node_features_list, dim=0)

        return self.gnn(graph_batch)

    @torch.no_grad()
    def encode_sentences(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        """Encode sentences using only the BERT branch (inference mode).

        This is the method used after training — the GNN branch is not needed
        because BERT has already internalized syntactic information.

        Returns:
            Sentence embeddings of shape (batch_size, hidden_dim).
        """
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sent_emb=True,
            **kwargs,
        )
        return output.h_bert
