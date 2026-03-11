"""
gnn_encoder.py — Syntax-aware Graph Neural Network Encoder

Encodes dependency parse trees into fixed-size sentence embeddings using
graph convolution layers (GAT, GCN, or Graph Transformer) from PyTorch Geometric.

Node features are BERT token embeddings aligned to the syntactic parser's
word-level tokenization. Edges represent syntactic dependency arcs.

Architecture:
    Input: PyG Batch with x = (total_nodes, hidden_dim), edge_index = (2, total_edges)
    → N layers of GATConv, GCNConv, or TransformerConv with ReLU + Dropout
    → Graph-level readout (mean / max / cls_node pooling)
    Output: h_G = (batch_size, hidden_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, GCNConv, TransformerConv, global_max_pool, global_mean_pool


class SyntaxGNNEncoder(nn.Module):
    """Graph neural network encoder for syntactic dependency trees.

    Args:
        in_dim: Input feature dimension (must match BERT hidden size, typically 768).
        hidden_dim: Hidden dimension of GNN layers.
        num_layers: Number of message-passing layers.
        conv_type: Type of graph convolution — "gat", "gcn", or "gt" (Graph Transformer).
        heads: Number of attention heads for GAT / GT (ignored if conv_type="gcn").
        dropout: Dropout probability after each layer.
        pooling: Graph-level readout strategy — "mean", "max", or "cls_node".
        gt_beta: Enable gated skip connection inside each TransformerConv layer (GT only).
        use_independent_embeddings: If True, the GNN uses its own word embedding
            table instead of receiving BERT hidden states as node features.
            This decouples the GNN input from BERT, making alignment non-trivial.
        vocab_size: Vocabulary size for independent embeddings (default: BERT vocab).
    """

    VALID_CONV_TYPES = ("gat", "gcn", "gt")
    VALID_POOLING = ("mean", "max", "cls_node")

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 768,
        num_layers: int = 2,
        conv_type: str = "gat",
        heads: int = 4,
        dropout: float = 0.1,
        pooling: str = "mean",
        gt_beta: bool = True,
        use_independent_embeddings: bool = False,
        vocab_size: int = 30522,
    ) -> None:
        super().__init__()

        if conv_type not in self.VALID_CONV_TYPES:
            raise ValueError(f"conv_type must be one of {self.VALID_CONV_TYPES}, got '{conv_type}'")
        if pooling not in self.VALID_POOLING:
            raise ValueError(f"pooling must be one of {self.VALID_POOLING}, got '{pooling}'")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.conv_type = conv_type
        self.pooling = pooling
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            layer_in = in_dim if i == 0 else hidden_dim

            if conv_type == "gat":
                # GAT: each head outputs hidden_dim // heads features, concatenated = hidden_dim
                head_dim = hidden_dim // heads
                conv = GATConv(
                    in_channels=layer_in,
                    out_channels=head_dim,
                    heads=heads,
                    concat=True,  # concatenate multi-head outputs
                    dropout=dropout,
                    add_self_loops=True,
                )
            elif conv_type == "gt":
                # Graph Transformer: full Q/K/V attention over graph neighbourhoods.
                # Each head outputs hidden_dim // heads features, concat → hidden_dim.
                # beta=True enables a learned gating between the skip connection and
                # the aggregated neighbourhood, analogous to a transformer residual.
                head_dim = hidden_dim // heads
                conv = TransformerConv(
                    in_channels=layer_in,
                    out_channels=head_dim,
                    heads=heads,
                    concat=True,
                    beta=gt_beta,
                    dropout=dropout,
                    edge_dim=None,
                )
            else:
                # GCN: simple degree-normalized convolution
                conv = GCNConv(
                    in_channels=layer_in,
                    out_channels=hidden_dim,
                    add_self_loops=True,
                    normalize=True,
                )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Output projection to ensure consistent output dimension
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # ---- Independent word embeddings (Axe B) ----
        self.use_independent_embeddings = use_independent_embeddings
        if use_independent_embeddings:
            self.word_embeddings = nn.Embedding(vocab_size, in_dim)
            # Small projection to allow the embedding space to diverge from BERT
            self.embed_layer_norm = nn.LayerNorm(in_dim)

    def init_from_bert_embeddings(self, bert_word_embeddings: nn.Embedding) -> None:
        """Warm-start the independent embedding table from BERT's word embeddings.

        This gives the GNN a meaningful starting point instead of random features,
        while allowing the embeddings to diverge during training.

        Args:
            bert_word_embeddings: BERT's ``word_embeddings`` layer.
        """
        if not self.use_independent_embeddings:
            return
        with torch.no_grad():
            # Copy BERT weights; handle size mismatch gracefully
            src_weight = bert_word_embeddings.weight
            tgt_weight = self.word_embeddings.weight
            n = min(src_weight.size(0), tgt_weight.size(0))
            d = min(src_weight.size(1), tgt_weight.size(1))
            tgt_weight[:n, :d] = src_weight[:n, :d]

    def compute_independent_features(
        self,
        word_token_ids: Tensor,
    ) -> Tensor:
        """Compute node features from the GNN's own embedding table.

        Args:
            word_token_ids: (total_nodes,) — BERT vocab token IDs for each word node.

        Returns:
            Node feature tensor of shape (total_nodes, in_dim).
        """
        x = self.word_embeddings(word_token_ids)
        x = self.embed_layer_norm(x)
        return x

    def forward(self, batch: Batch) -> Tensor:
        """Encode a batch of dependency graphs into sentence embeddings.

        Args:
            batch: PyG Batch object containing:
                - x: Node features of shape (total_nodes, in_dim)
                - edge_index: Edge connectivity of shape (2, total_edges)
                - batch: Batch assignment vector of shape (total_nodes,)

        Returns:
            h_G: Graph-level embeddings of shape (batch_size, hidden_dim)
        """
        x = batch.x
        edge_index = batch.edge_index
        batch_vec = batch.batch

        # Multi-head conv types concatenate heads → output size is out_channels * heads
        _multi_head = self.conv_type in ("gat", "gt")

        # Message passing layers
        for conv, norm in zip(self.convs, self.norms):
            expected_out = conv.out_channels * (conv.heads if _multi_head else 1)
            x_res = x if x.size(-1) == expected_out else None
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Residual connection when dimensions match
            if x_res is not None:
                x = x + x_res

        # Graph-level readout
        if self.pooling == "mean":
            h_G = global_mean_pool(x, batch_vec)
        elif self.pooling == "max":
            h_G = global_max_pool(x, batch_vec)
        elif self.pooling == "cls_node":
            # Use the ROOT node (index 0 in each graph) as the graph representation.
            # In dependency trees, ROOT is typically the first node.
            h_G = self._cls_node_pool(x, batch_vec)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        h_G = self.out_proj(h_G)
        return h_G

    @staticmethod
    def _cls_node_pool(x: Tensor, batch: Tensor) -> Tensor:
        """Extract the first node (ROOT) from each graph in the batch."""
        # Find indices where a new graph starts
        batch_size = int(batch.max().item()) + 1
        # Scatter: for each graph, get the index of its first node
        first_indices = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        seen = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        for i in range(x.size(0)):
            graph_id = batch[i].item()
            if not seen[graph_id]:
                first_indices[graph_id] = i
                seen[graph_id] = True
        return x[first_indices]
