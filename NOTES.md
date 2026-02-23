# Session Notes — GNN-Syntax-BERT Project

> Document tracking all design decisions, research findings, and implementation progress.
> Project: *Transferring Syntactic Structure into Sentence Embeddings using GNNs*
> Course: CS224N — Stanford University

---

## Session 1 — February 22, 2026: Project Initialization

### Objective
Scaffold the complete research project: directory structure, configs, core modules, and scripts.

### Scientific Context

The project aims to improve BERT's syntactic understanding through contrastive alignment with a GNN that encodes dependency parse trees. Key difference from prior work (SynCSE): we align representations **during training** so that BERT internalizes syntax, then **only BERT** is used at inference. The GNN is discarded after training.

**Architecture**:
- Branch BERT (h_B): Standard sentence encoder, pooled via CLS or mean pooling.
- Branch GNN (h_G): Encodes syntactic dependency graphs. Node features initialized from BERT token embeddings.
- Alignment: NT-Xent contrastive loss where (h_B, h_G) for the same sentence form positive pairs.
- Combined loss: $L = L_\text{SimCSE} + \lambda L_\text{align} + \mu L_\text{GNN}$

### Design Decisions

#### 1. SimCSE Integration → Git Submodule + Editable Install
**Decision**: SimCSE is added as a git submodule at `./SimCSE/` and installed with `uv pip install -e ./SimCSE`.

**Rationale**: 
- Keeps upstream intact for future `git pull` updates.
- SimCSE's `setup.py` has pinned deps (`scipy<1.6`, `numpy<1.20`) that conflict with modern PyTorch Geometric. The `setup_env.sh` script patches these via `sed`.
- SimCSE can still be imported as `from simcse.models import BertForCL`.

**Alternatives considered**:
- *Vendoring* (copying SimCSE files into `src/`): Rejected — harder to track upstream changes and unclear licensing implications.
- *Full fork*: Rejected — overly heavy; we only extend SimCSE, not rewrite it.

#### 2. GNN Backend → PyTorch Geometric (PyG)
**Decision**: Use PyG with both `GATConv` and `GCNConv` available, selectable via Hydra config.

**Rationale**:
- More mature ecosystem, wider community adoption, better documentation.
- `torch_geometric.data.Batch` handles variable-size graph batching natively.
- `GATConv` allows attention-weighted aggregation (can learn which dependency edges matter more).
- `GCNConv` serves as a simpler baseline for ablations.

**Alternative considered**:
- *DGL*: Easier pip install but narrower ecosystem. Rejected.

#### 3. GNN Conv Layer → Both GAT and GCN (Configurable)
**Decision**: Both are implemented in `SyntaxGNNEncoder`, selected via `configs/model/gnn.yaml → conv_type: gat|gcn`.

**Rationale**: Key ablation axis — we need to measure whether attention over dependency edges adds value over simple degree-normalized propagation.

#### 4. Hydra Strategy → Compose API (Not `@hydra.main()`)
**Decision**: Use `hydra.initialize_config_dir()` + `hydra.compose()` instead of `@hydra.main()`.

**Rationale**:
- SimCSE uses `HfArgumentParser` (from HuggingFace `transformers`), which internally reads `sys.argv`.
- Hydra's `@hydra.main()` also wants to own `sys.argv` → conflict.
- The Compose API lets us load Hydra configs programmatically, then manually construct SimCSE's `ModelArguments`, `DataTrainingArguments`, and `OurTrainingArguments` dataclasses.
- This approach is documented by Hydra as the recommended pattern when integrating with other argument parsers.

#### 5. Subword → Word Alignment → Greedy Char-Span + Mean-Pool
**Decision**: Align BERT WordPiece tokens to Stanza words via character-position tracking, then mean-pool subword embeddings per word.

**Rationale**:
- Stanza tokenizes at the word level (e.g., `["unbelievable", "cat"]`), while BERT uses subwords (e.g., `["un", "##bel", "##ie", "##va", "##ble", "cat"]`).
- We track consumed characters: each subword's cleaned text consumes characters from the current Stanza word. When the word is fully consumed, advance to the next.
- Mean-pooling (vs. first-subword) preserves information from all subwords.
- Edge case: Stanza and BERT may disagree on tokenization boundaries. The alignment falls back gracefully with a `-1` (unmapped) index.

#### 6. Project Manager → uv (Exclusively)
**Decision**: Use `uv` for all dependency management (`uv sync`, `uv run`). No `pip`, no `conda`.

**Rationale**: Fast, deterministic resolution, lockfile-based, modern standard for Python packaging.

### Research: SimCSE Codebase Analysis

Key findings from analyzing `princeton-nlp/SimCSE`:

| Component | Details |
|---|---|
| **Model classes** | `BertForCL(BertPreTrainedModel)`, `RobertaForCL(RobertaPreTrainedModel)` in `simcse/models.py` |
| **Training** | `CLTrainer(Trainer)` in `simcse/trainers.py` — heavily forked HF Trainer (~500 lines) |
| **Contrastive loss** | Inside `cl_forward()`: cosine sim matrix / temp → CrossEntropy with diagonal labels |
| **Data loading** | HF `datasets.load_dataset()` — `.txt` for unsupervised, `.csv` for supervised |
| **Pooling** | 5 options: `cls`, `cls_before_pooler`, `avg`, `avg_top2`, `avg_first_last` |
| **Evaluation** | SentEval integration in `CLTrainer.evaluate()` and standalone `evaluation.py` |
| **Pinned deps** | `scipy>=1.5.4,<1.6`, `numpy>=1.19.5,<1.20` — must be relaxed |
| **HF dataset** | `princeton-nlp/datasets-for-simcse` containing `wiki1m_for_simcse.txt` (~120MB, 1M lines) |

### Technical Constraints Identified

| Constraint | Severity | Mitigation |
|---|---|---|
| SimCSE's pinned scipy/numpy | Medium | Patch `setup.py` via `sed` in `setup_env.sh` |
| SimCSE uses old `transformers` imports (`from transformers.file_utils import ...`) | Medium | May need transformers ~4.36+ with compatibility shims |
| `HfArgumentParser` vs Hydra `sys.argv` conflict | Low | Compose API (implemented) |
| Stanza word-level vs BERT subword tokenization mismatch | Medium | Greedy char-span alignment + mean-pool (implemented) |
| SimCSE's `CLTrainer` is a full fork of old HF `Trainer` | Medium | Don't subclass it; build standalone `SyntaxCLTrainer` that uses modern `transformers.Trainer` |
| SimCSE's `setup.py` (not pyproject.toml) | None | Works fine with `uv pip install -e` |

### Files Created

| File | Description | Status |
|---|---|---|
| `pyproject.toml` | uv project config with all dependencies | ✅ Complete |
| `.gitignore` | Ignores data/, outputs/, .venv/, etc. | ✅ Complete |
| `configs/config.yaml` | Hydra root config | ✅ Complete |
| `configs/model/bert.yaml` | BERT encoder params | ✅ Complete |
| `configs/model/gnn.yaml` | GNN encoder params (GAT/GCN) | ✅ Complete |
| `configs/model/alignment.yaml` | Alignment loss params (λ, μ, τ) | ✅ Complete |
| `configs/experiment/stop_grad.yaml` | Stop-gradient strategy | ✅ Complete |
| `configs/experiment/freeze_then_align.yaml` | Two-phase training strategy | ✅ Complete |
| `configs/experiment/multi_loss.yaml` | Joint multi-loss strategy | ✅ Complete |
| `scripts/setup_env.sh` | Full env setup (submodule, patch, uv, stanza) | ✅ Complete |
| `scripts/download_wiki.py` | Download wiki1m from HuggingFace Hub | ✅ Complete |
| `src/models/gnn_encoder.py` | `SyntaxGNNEncoder` — GAT/GCN with readout | ✅ Complete |
| `src/models/wrapper.py` | `SyntaxBertModel` — BERT + GNN wrapper | ✅ Complete |
| `src/alignment/losses.py` | `AlignmentLoss`, `GNNContrastiveLoss`, `CombinedLoss` | ✅ Complete |
| `src/processing/syntax_parser.py` | `StanzaSyntaxParser` — parsing + alignment + caching | ✅ Complete |
| `src/train.py` | Main entry point with Hydra Compose API bridge | ✅ Complete (training loop skeleton) |
| `tests/test_gnn_encoder.py` | Smoke tests for GNN, losses, alignment | ✅ Complete |
| `README.md` | Full project documentation | ✅ Complete |
| `TODO.md` | Remaining work items | ✅ Complete |
| `NOTES.md` | This file | ✅ Complete |

### Remaining Work (see TODO.md)

The major remaining piece is the **training loop integration**:
1. Custom `DataCollator` that loads PyG graphs alongside tokenized text
2. Proper `compute_loss()` override in a HuggingFace `Trainer` subclass
3. Freeze-then-align phase transitions
4. STS evaluation with the wrapper model
5. Multi-GPU support for PyG batch tensors

### Architecture Diagram

```
Training Pipeline:
                                                 
  wiki1m_for_simcse.txt ──► Stanza Parser ──► parsed_graphs/ (.jsonl)
          │                                        │
          ▼                                        ▼
  BERT Tokenizer              Parse Result → PyG Data (edge_index, num_nodes)
          │                                        │
          ▼                                        ▼
  ┌───────────────┐                    ┌───────────────────┐
  │ BertForCL     │ hidden states ───► │ Subword→Word      │
  │ (SimCSE)      │                    │ Alignment + Pool   │
  │               │                    └────────┬──────────┘
  │  ┌─────────┐  │                             │
  │  │ Pooler  │  │                    ┌────────▼──────────┐
  │  └────┬────┘  │                    │ SyntaxGNNEncoder  │
  │       │       │                    │ (GAT/GCN layers)  │
  └───────┼───────┘                    │ + global_pool     │
          │                            └────────┬──────────┘
          ▼                                     ▼
        h_B                                   h_G
          │                                     │
          ├──────────► AlignmentLoss ◄──────────┤
          │            (NT-Xent)                │
          ▼                                     ▼
      L_SimCSE                              L_GNN (aux)
          │                                     │
          └──────────┬──────────────────────────┘
                     ▼
          L = L_SimCSE + λ·L_align + μ·L_GNN
```

### References

- Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP.
- Zhang, J., et al. (2023). SynCSE: Syntax-enhanced Contrastive Learning for Sentence Embedding.
- Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
- Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. ICLR.

---

*Last updated: February 22, 2026*
