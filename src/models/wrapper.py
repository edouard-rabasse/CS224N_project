"""
wrapper.py — SyntaxBertModel: Wraps SimCSE's BertForCL + SyntaxGNNEncoder

This module is the core architectural contribution. It combines:
  1. BERT branch (h_B): SimCSE's contrastive learning encoder
  2. GNN branch (h_G): Syntactic dependency graph encoder

During training, both branches produce sentence embeddings that are aligned
via a contrastive loss. At inference time, only the BERT branch is used —
the GNN branch is discarded, since BERT has internalized syntactic structure.

Key design choices:
  - Training forward accepts SimCSE format: input_ids of shape (bs, 2, seq_len).
    BERT runs once on the flattened (bs*2, seq_len) input; the SimCSE contrastive
    loss is computed from the two pooled views (z1, z2) internally.
  - Node features for the GNN come from BERT's last hidden states of view 1,
    aligned to the syntactic parser's word-level tokenization.
  - An optional projection head maps both h_B and h_G to a shared space
    before computing the alignment loss.
  - Supports stop-gradient and freeze modes for ablation experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch

from src.models.gnn_encoder import SyntaxGNNEncoder


@dataclass
class SyntaxBertOutput:
    """Output of the SyntaxBertModel forward pass."""

    h_bert: Tensor  # (batch_size, hidden_dim) — BERT sentence embedding (view 1)
    h_gnn: Optional[Tensor] = None  # (batch_size, hidden_dim) — GNN graph embedding
    h_bert_proj: Optional[Tensor] = None  # (batch_size, proj_dim) — projected BERT embedding
    h_gnn_proj: Optional[Tensor] = None  # (batch_size, proj_dim) — projected GNN embedding
    bert_hidden_states: Optional[Tensor] = None  # (batch_size, seq_len, hidden_dim) — view 1 hidden states
    simcse_loss: Optional[Tensor] = None  # scalar — SimCSE NT-Xent loss (z1 vs z2)


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
        bert_model: A SimCSE BertForCL (or RobertaForCL) instance, or a
            compatible wrapper (e.g. DebugBertForCL for testing).
        gnn_config: Dict with GNN hyperparameters (from Hydra config).
        alignment_config: Dict with alignment hyperparameters.
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
        # True when bert_model is SimCSE's BertForCL (has .bert sub-module + custom .pooler)
        # False when it is a plain HuggingFace BertModel
        self._is_simcse = hasattr(bert_model, "bert") and hasattr(bert_model, "pooler")

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

        Inference mode (``sent_emb=True`` or ``graph_batch is None``):
            - ``input_ids`` shape: ``(bs, seq_len)``
            - Only the BERT branch runs; GNN is discarded.
            - Returns ``h_bert`` only (no loss, no GNN output).

        Training mode (``graph_batch`` is provided):
            - ``input_ids`` shape: ``(bs, 2, seq_len)`` — SimCSE format.
              The same tokens are passed twice; different dropout masks create
              two views ``z1, z2`` for the SimCSE NT-Xent loss.
            - BERT runs once on the flattened ``(bs*2, seq_len)`` input.
            - View-1 hidden states ``[0::2]`` feed the GNN branch.
            - ``simcse_loss`` is computed as NT-Xent between ``z1`` and ``z2``.

        Args:
            input_ids: Token IDs, shape ``(bs, seq_len)`` or ``(bs, 2, seq_len)``.
            attention_mask: Attention mask, same shape as ``input_ids``.
            graph_batch: PyG ``Batch`` of dependency graphs (``bs`` graphs).
                If ``None``, the GNN branch is skipped (inference mode).
            token_to_word_maps: List of ``bs`` alignment maps from
                ``StanzaSyntaxParser.align_subwords()``. Each map has length
                ``seq_len``, with values being Stanza word indices (0-based)
                or -1 for special tokens.
            sent_emb: If ``True``, force inference mode regardless of other args.
            **bert_kwargs: Extra kwargs forwarded to the underlying BERT model.

        Returns:
            :class:`SyntaxBertOutput` with ``h_bert``, ``h_gnn``,
            projection embeddings, view-1 hidden states, and ``simcse_loss``.
        """
        # ---- Inference mode ------------------------------------------------
        if sent_emb or graph_batch is None:
            if self._is_simcse:
                # SimCSE BertForCL: use its sentemb_forward path
                bert_out = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sent_emb=True,
                    **bert_kwargs,
                )
                return SyntaxBertOutput(h_bert=bert_out.pooler_output)
            else:
                # Plain BertModel: forward returns BaseModelOutputWithPooling
                bert_out = self.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                    **bert_kwargs,
                )
                h = bert_out.pooler_output if bert_out.pooler_output is not None else bert_out.last_hidden_state[:, 0]
                return SyntaxBertOutput(h_bert=h)

        # ---- Training mode -------------------------------------------------
        # input_ids is (bs, 2, seq_len) — SimCSE format
        if input_ids.dim() == 3:
            bs, num_sent, seq_len = input_ids.shape
            flat_input_ids = input_ids.view(bs * num_sent, seq_len)
            flat_attention_mask = attention_mask.view(bs * num_sent, seq_len)
        else:
            # Fallback: single-view (bs, seq_len) — used for unit tests / debug
            bs, seq_len = input_ids.shape
            num_sent = 1
            flat_input_ids = input_ids
            flat_attention_mask = attention_mask

        # Run BERT once on the flattened input
        if self._is_simcse:
            # SimCSE BertForCL: call the inner .bert, then SimCSE's custom pooler
            needs_hidden_states = (
                getattr(getattr(self.bert, "model_args", None), "pooler_type", "cls")
                in ("avg_top2", "avg_first_last")
            )
            bert_outputs = self.bert.bert(
                flat_input_ids,
                attention_mask=flat_attention_mask,
                output_hidden_states=needs_hidden_states,
                return_dict=True,
                **bert_kwargs,
            )
            pooler_output = self.bert.pooler(flat_attention_mask, bert_outputs)
            pooler_output = pooler_output.view(bs, num_sent, -1)
            if getattr(self.bert, "pooler_type", "") == "cls":
                pooler_output = self.bert.mlp(pooler_output)
        else:
            # Plain BertModel: call it directly
            bert_outputs = self.bert(
                flat_input_ids,
                attention_mask=flat_attention_mask,
                return_dict=True,
                **bert_kwargs,
            )
            p = bert_outputs.pooler_output
            if p is None:
                p = bert_outputs.last_hidden_state[:, 0]
            pooler_output = p.view(bs, num_sent, -1)

        # z1 = view 1 sentence embeddings → this is h_B
        z1 = pooler_output[:, 0]  # (bs, hidden_dim)
        h_bert = z1

        # ---- SimCSE contrastive loss (NT-Xent between z1 and z2) ----------
        simcse_loss: Optional[Tensor] = None
        if num_sent == 2:
            z2 = pooler_output[:, 1]  # (bs, hidden_dim)
            temp = getattr(getattr(self.bert, "model_args", None), "temp", 0.05)
            z1_n = F.normalize(z1, dim=-1)
            z2_n = F.normalize(z2, dim=-1)
            # cos_sim[i, j] = cos(z1_i, z2_j) / temp
            cos_sim = (z1_n @ z2_n.T) / temp  # (bs, bs)
            labels = torch.arange(bs, device=input_ids.device)
            simcse_loss = F.cross_entropy(cos_sim, labels)

        # ---- GNN branch ------------------------------------------------
        # Use view-1 hidden states as GNN node features
        # last_hidden_state shape: (bs * num_sent, seq_len, hidden_dim)
        # Interleaved order: [sent0_view1, sent0_view2, sent1_view1, sent1_view2, ...]
        last_hidden_flat = bert_outputs.last_hidden_state  # (bs * num_sent, seq_len, D)
        last_hidden_v1 = last_hidden_flat[0::num_sent]     # (bs, seq_len, D) — view 1

        h_gnn = self._compute_gnn_embeddings(last_hidden_v1, graph_batch, token_to_word_maps)

        # Stop-gradient on GNN branch if configured
        if self.stop_grad_gnn:
            h_gnn = h_gnn.detach()

        # ---- Projections for alignment loss ----------------------------
        h_bert_proj = self.bert_projector(h_bert) if self.bert_projector is not None else None
        h_gnn_proj = self.gnn_projector(h_gnn) if self.gnn_projector is not None else None

        return SyntaxBertOutput(
            h_bert=h_bert,
            h_gnn=h_gnn,
            h_bert_proj=h_bert_proj,
            h_gnn_proj=h_gnn_proj,
            bert_hidden_states=last_hidden_v1,
            simcse_loss=simcse_loss,
        )

    def _compute_gnn_embeddings(
        self,
        bert_hidden: Tensor,
        graph_batch: Batch,
        token_to_word_maps: Optional[list[list[int]]],
    ) -> Tensor:
        """Extract per-word BERT features, set as GNN node features, and encode.

        If ``token_to_word_maps`` is provided, aggregates subword embeddings by
        mean-pooling per word. Otherwise, assumes ``graph_batch.x`` is already set.

        Args:
            bert_hidden: ``(bs, seq_len, hidden_dim)`` — BERT last hidden states (view 1).
            graph_batch: PyG ``Batch`` with ``edge_index`` and ``batch`` vectors.
            token_to_word_maps: Subword → word alignment maps (length = bs).

        Returns:
            h_G: ``(bs, hidden_dim)`` — graph-level embeddings.
        """
        if token_to_word_maps is not None:
            node_features_list = []
            # graph_batch.ptr is the cumulative node-count array [0, n0, n0+n1, ...]
            # Use it so the feature matrix always matches edge_index dimensions,
            # even when BERT truncated the sentence and the parse has more words.
            ptr = graph_batch.ptr  # (bs+1,)
            for i, word_map in enumerate(token_to_word_maps):
                sent_hidden = bert_hidden[i]  # (seq_len, hidden_dim)
                # True node count for this graph (from the parse, not the alignment)
                num_words = max(int((ptr[i + 1] - ptr[i]).item()), 1)
                word_feats = torch.zeros(
                    num_words,
                    sent_hidden.size(-1),
                    device=sent_hidden.device,
                    dtype=sent_hidden.dtype,
                )
                word_counts = torch.zeros(num_words, device=sent_hidden.device)

                for subword_idx, word_idx in enumerate(word_map):
                    # Guard: skip indices beyond the parse graph (truncated sentences)
                    if 0 <= word_idx < num_words:
                        word_feats[word_idx] += sent_hidden[subword_idx]
                        word_counts[word_idx] += 1

                # Mean-pool subwords per word; words with no BERT coverage keep zeros
                word_feats = word_feats / word_counts.clamp(min=1).unsqueeze(-1)
                node_features_list.append(word_feats)

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
            Sentence embeddings of shape ``(batch_size, hidden_dim)``.
        """
        output = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            sent_emb=True,
            **kwargs,
        )
        return output.h_bert
