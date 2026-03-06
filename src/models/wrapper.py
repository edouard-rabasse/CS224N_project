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

import json
from dataclasses import dataclass
from pathlib import Path
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
        self.detach_bert_for_gnn = alignment_config.get("detach_bert_for_gnn", False)
        self.freeze_gnn = gnn_config.get("freeze", False)

        if self.freeze_gnn:
            self._freeze_gnn()

        # ---- Partial BERT freezing ----
        # num_unfrozen_layers controls how many encoder layers are trainable:
        #   None  → fine-tune ALL layers (full fine-tuning, default)
        #   0     → freeze ALL encoder layers + embeddings, only pooler/MLP head trainable
        #   N > 0 → freeze layers 0..(L-N-1), fine-tune last N layers + pooler
        bert_config = getattr(bert_model, "config", None)
        self._num_bert_layers = getattr(bert_config, "num_hidden_layers", 12)
        num_unfrozen = alignment_config.get("num_unfrozen_layers", None)
        if num_unfrozen is None:
            # Also check gnn_config for backwards compat
            num_unfrozen = gnn_config.get("num_unfrozen_layers", None)
        self.num_unfrozen_layers = num_unfrozen
        if num_unfrozen is not None:
            self._partial_freeze_bert(num_unfrozen)

    def _freeze_gnn(self) -> None:
        """Freeze all GNN parameters."""
        for param in self.gnn.parameters():
            param.requires_grad = False

    def _freeze_bert(self) -> None:
        """Freeze all BERT parameters (used in phase 1 of freeze-then-align)."""
        for param in self.bert.parameters():
            param.requires_grad = False

    def _unfreeze_bert(self) -> None:
        """Unfreeze BERT parameters, respecting partial freeze if configured.

        If ``num_unfrozen_layers`` is set (including 0), only the specified
        layers + pooler are unfrozen. Otherwise all parameters are unfrozen.
        """
        if self.num_unfrozen_layers is not None:
            self._partial_freeze_bert(self.num_unfrozen_layers)
        else:
            for param in self.bert.parameters():
                param.requires_grad = True

    def _partial_freeze_bert(self, num_unfrozen: int) -> None:
        """Freeze all BERT layers except the last ``num_unfrozen`` encoder layers.

        If ``num_unfrozen == 0``, ALL encoder layers + embeddings are frozen;
        only the pooler and SimCSE MLP head remain trainable.

        Frozen: embeddings + encoder layers 0 .. (L - num_unfrozen - 1).
        Trainable: encoder layers (L - num_unfrozen) .. (L-1) + pooler.

        Works with both SimCSE's BertForCL (which wraps .bert) and plain
        HuggingFace BertModel.

        Args:
            num_unfrozen: Number of top encoder layers to keep trainable.
        """
        # First freeze everything
        for param in self.bert.parameters():
            param.requires_grad = False

        # Resolve the inner encoder (SimCSE wraps as self.bert.bert)
        inner_bert = getattr(self.bert, "bert", self.bert)
        encoder_layers = None
        if hasattr(inner_bert, "encoder") and hasattr(inner_bert.encoder, "layer"):
            encoder_layers = inner_bert.encoder.layer
        elif hasattr(inner_bert, "layers"):
            encoder_layers = inner_bert.layers

        if encoder_layers is None:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "Could not find encoder layers for partial freeze — unfreezing all BERT params."
            )
            for param in self.bert.parameters():
                param.requires_grad = True
            return

        total_layers = len(encoder_layers)
        first_unfrozen = max(total_layers - num_unfrozen, 0)

        # Unfreeze last N layers
        for i in range(first_unfrozen, total_layers):
            for param in encoder_layers[i].parameters():
                param.requires_grad = True

        # Unfreeze pooler (always trainable when BERT is active)
        if hasattr(inner_bert, "pooler") and inner_bert.pooler is not None:
            for param in inner_bert.pooler.parameters():
                param.requires_grad = True

        # Unfreeze SimCSE's custom pooler + MLP if present
        if self._is_simcse:
            if hasattr(self.bert, "pooler"):
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
            if hasattr(self.bert, "mlp"):
                for param in self.bert.mlp.parameters():
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

        gnn_input = last_hidden_v1.detach() if self.detach_bert_for_gnn else last_hidden_v1
        h_gnn = self._compute_gnn_embeddings(gnn_input, graph_batch, token_to_word_maps)

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

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str, tokenizer=None) -> None:
        """Save both BERT and GNN weights to ``output_dir``.

        Saves:
          - ``bert_weights/``          — HuggingFace-format BERT weights
            (loadable with ``AutoModel.from_pretrained``)
          - ``gnn_state.pt``           — raw GNN + projection-head state dict
          - ``model_config.json``      — GNN / alignment hyper-parameters

        Args:
            output_dir: Directory to write the checkpoint into (created if absent).
            tokenizer: Optional tokenizer to save alongside the weights.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # BERT branch: save in HuggingFace format so it can be loaded standalone.
        # When wrapping SimCSE's BertForCL, save the *inner* BertModel directly
        # (self.bert.bert) so that BertModel.from_pretrained() loads without any
        # key-prefix mismatch.  Saving the outer BertForCL would prefix all
        # encoder keys with "bert." and they would be silently ignored on reload.
        bert_dir = out / "bert_weights"
        bert_to_save = (
            self.bert.bert
            if self._is_simcse and hasattr(self.bert, "bert")
            else self.bert
        )
        if hasattr(bert_to_save, "save_pretrained"):
            bert_to_save.save_pretrained(str(bert_dir))
        else:
            bert_dir.mkdir(parents=True, exist_ok=True)
            torch.save(bert_to_save.state_dict(), bert_dir / "pytorch_model.bin")

        # Save tokenizer alongside BERT weights so eval can find it
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(bert_dir))
        else:
            print("Warning: tokenizer not saved with checkpoint; make sure to provide it during evaluation for consistent tokenization.")

        # GNN + projection heads: save as a plain state dict
        gnn_state: dict = {"gnn": self.gnn.state_dict()}
        if self.bert_projector is not None:
            gnn_state["bert_projector"] = self.bert_projector.state_dict()
        if self.gnn_projector is not None:
            gnn_state["gnn_projector"] = self.gnn_projector.state_dict()
        torch.save(gnn_state, out / "gnn_state.pt")

        # Store GNN / alignment hyper-parameters for reconstruction
        cfg = {
            "hidden_dim": self.hidden_dim,
            "stop_grad_gnn": self.stop_grad_gnn,
            "freeze_gnn": self.freeze_gnn,
            "has_bert_projector": self.bert_projector is not None,
            "has_gnn_projector": self.gnn_projector is not None,
        }
        with open(out / "model_config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def save_bert_only(self, output_dir: str, tokenizer=None) -> None:
        """Save only the BERT branch in HuggingFace format.

        Use this to produce an inference checkpoint that is directly
        loadable as a standard sentence embedding model — no GNN code needed.

        Args:
            output_dir: Destination directory (created if absent).
            tokenizer: Optional tokenizer to save alongside the weights.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        bert_to_save = (
            self.bert.bert
            if self._is_simcse and hasattr(self.bert, "bert")
            else self.bert
        )
        if hasattr(bert_to_save, "save_pretrained"):
            bert_to_save.save_pretrained(str(out))
        else:
            torch.save(bert_to_save.state_dict(), out / "pytorch_model.bin")

        # Save tokenizer so the checkpoint is self-contained for evaluation
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(str(out))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        device: Optional[torch.device] = None,
    ) -> "SyntaxBertModel":
        """Load a SyntaxBertModel from a checkpoint saved by ``save_checkpoint``.

        Reconstructs both BERT and GNN branches. Requires ``bert_weights/``,
        ``gnn_state.pt``, and ``model_config.json`` to be present.

        Args:
            checkpoint_dir: Directory produced by ``save_checkpoint``.
            device: Target device. Defaults to CPU.

        Returns:
            Fully restored ``SyntaxBertModel`` in eval mode.
        """
        from transformers import BertModel

        ckpt = Path(checkpoint_dir)
        cfg_path = ckpt / "model_config.json"
        gnn_path = ckpt / "gnn_state.pt"
        bert_dir = ckpt / "bert_weights"

        with open(cfg_path) as f:
            saved_cfg: dict = json.load(f)

        hidden_dim: int = saved_cfg["hidden_dim"]

        # Reconstruct BERT
        # add_pooling_layer=False: SimCSE's inner BertModel has no pooler;
        # inference uses last_hidden_state[:, 0] (cls_before_pooler style).
        if bert_dir.exists():
            bert_model = BertModel.from_pretrained(str(bert_dir), add_pooling_layer=False)
        else:
            raise FileNotFoundError(f"bert_weights/ not found in {ckpt}")

        # Reconstruct GNN config from saved hidden dim
        gnn_config: dict = {
            "conv_type": "gat",       # defaults — overridden by loaded weights
            "num_layers": 2,
            "hidden_dim": hidden_dim,
            "heads": 4,
            "dropout": 0.1,
            "pooling": "mean",
        }
        proj_dim = hidden_dim // 3 if saved_cfg.get("has_bert_projector", False) else 0
        alignment_config: dict = {
            "projector_dim": proj_dim,
            "stop_grad_gnn": saved_cfg.get("stop_grad_gnn", False),
        }

        model = cls(
            bert_model=bert_model,
            gnn_config=gnn_config,
            alignment_config=alignment_config,
        )

        # Restore GNN + projector weights
        if gnn_path.exists():
            gnn_state = torch.load(gnn_path, map_location="cpu", weights_only=True)
            model.gnn.load_state_dict(gnn_state["gnn"])
            if "bert_projector" in gnn_state and model.bert_projector is not None:
                model.bert_projector.load_state_dict(gnn_state["bert_projector"])
            if "gnn_projector" in gnn_state and model.gnn_projector is not None:
                model.gnn_projector.load_state_dict(gnn_state["gnn_projector"])

        model.eval()
        if device is not None:
            model = model.to(device)
        return model
