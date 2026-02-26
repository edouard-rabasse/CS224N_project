# GEMINI.md — AI Agent Instructions

> **This file is the primary reference for any AI coding agent working on this project.**
> Read this ENTIRELY before making any changes.

---

## 1. What This Project Is

**Title**: *Transferring Syntactic Structure into Sentence Embeddings using GNNs*

**Course**: CS224N (NLP with Deep Learning) — Stanford University

**One-sentence summary**: We train BERT alongside a Graph Neural Network that encodes dependency parse trees, using contrastive alignment so BERT **internalizes syntactic structure** — then discard the GNN at inference.

### Core Hypothesis

Standard BERT embeddings are weak at capturing syntactic structure (e.g., dependency relations, tree depth). By aligning BERT's sentence representations with a GNN that explicitly encodes dependency parse trees during training, BERT can learn to encode syntax **without needing the GNN at inference time**.

### How It Differs From Prior Work

| Approach         | When GNN/Syntax is Used               | Our Advantage                           |
| ---------------- | ------------------------------------- | --------------------------------------- |
| **SynCSE** | At inference (fuses embeddings)       | Inference cost, architecture dependency |
| **Ours**   | Training only (contrastive alignment) | At inference: pure BERT, no GNN needed  |

### Architecture Overview

```
Training:
  Sentence → BERT encoder → h_B (sentence embedding)
                                ↕  contrastive alignment loss
  Sentence → Stanza parser → dependency graph → GNN encoder → h_G (graph embedding)

  Loss = L_SimCSE + λ·L_align(h_B, h_G) + μ·L_GNN

Inference (GNN discarded):
  Sentence → BERT encoder → h_B → downstream tasks (STS, classification, etc.)
```

The GNN's node features come from BERT's own hidden states (aligned from subword to word level), so the GNN acts as a "syntactic lens" on BERT's representations.

---

## 2. Project Structure

```
project/
├── pyproject.toml              # uv-managed dependencies
├── SimCSE/                     # Git submodule: princeton-nlp/SimCSE
├── configs/                    # Hydra YAML configs
│   ├── config.yaml             # Root config (defaults + training + data)
│   ├── model/
│   │   ├── bert.yaml           # BERT params → maps to SimCSE's ModelArguments
│   │   ├── gnn.yaml            # GNN params: conv_type (gat|gcn), layers, heads, pooling
│   │   └── alignment.yaml      # Loss weights: lambda_align, mu_gnn, temperature
│   └── experiment/
│       ├── stop_grad.yaml      # Freeze GNN, align BERT only
│       ├── freeze_then_align.yaml  # Phase 1: warm up GNN; Phase 2: joint training
│       └── multi_loss.yaml     # Joint training from epoch 0 (default)
├── run_pipeline.sh             # Automated pipeline: train → validate → evaluate → plot
├── scripts/
│   ├── setup_env.sh            # Environment setup (uv sync + SimCSE patch + Stanza)
│   ├── download_wiki.py        # Download wiki1m_for_simcse.txt from HuggingFace
│   ├── debug_train.py          # Quick end-to-end pipeline test (no deps needed)
│   ├── evaluation.py           # STS + syntactic probing evaluation (to be written)
│   └── plot_results.py         # Chart generation from eval JSON (to be written)
├── src/
│   ├── train.py                # Main entry: Hydra Compose API → SimCSE bridge → training
│   ├── models/
│   │   ├── gnn_encoder.py      # SyntaxGNNEncoder (GAT/GCN + pooling)
│   │   └── wrapper.py          # SyntaxBertModel (BERT + GNN + projections)
│   ├── alignment/
│   │   └── losses.py           # AlignmentLoss, GNNContrastiveLoss, CombinedLoss
│   └── processing/
│       └── syntax_parser.py    # StanzaSyntaxParser: parsing + subword alignment
├── tests/
│   └── test_gnn_encoder.py     # Smoke tests for GNN, losses, alignment
├── README.md                   # User-facing documentation
├── TODO.md                     # Remaining implementation work
└── NOTES.md                    # Design decisions and research log
```

---

## 3. Key Modules — What Each File Does

### `src/models/gnn_encoder.py` — `SyntaxGNNEncoder`

- **Input**: PyG `Batch` with `x` (node features from BERT), `edge_index` (dependency edges), `batch` (graph assignment)
- **Architecture**: N layers of `GATConv` or `GCNConv` (configurable) + LayerNorm + ReLU + residual connections
- **Output**: `h_G` of shape `(batch_size, hidden_dim)` via `global_mean_pool` / `global_max_pool` / CLS-node pooling
- **Config**: `configs/model/gnn.yaml`

### `src/models/wrapper.py` — `SyntaxBertModel`

- Wraps `BertForCL` (SimCSE) + `SyntaxGNNEncoder`
- **Training forward**: Runs BERT → extracts hidden states → aligns subwords to words → sets as GNN node features → runs GNN → returns both `h_B` and `h_G` (+ optional projections)
- **Inference forward** (`sent_emb=True`): Runs BERT only, returns `h_B`, `h_G=None`
- Handles `stop_grad` (detaches `h_G` before alignment loss) and `freeze` (freezes GNN/BERT parameters)
- **IMPORTANT**: The current `forward()` bypasses SimCSE's `cl_forward()` — see TODO.md for the integration plan

### `src/alignment/losses.py`

- `AlignmentLoss`: Symmetric NT-Xent between `h_B` and `h_G`. Positive pairs = same sentence. Temperature-scaled.
- `GNNContrastiveLoss`: Auxiliary contrastive loss for GNN (between two graph views). Includes `drop_edges()` augmentation utility.
- `CombinedLoss`: `L = L_SimCSE + λ·L_align + μ·L_GNN`. Returns dict with each loss component.

### `src/processing/syntax_parser.py` — `StanzaSyntaxParser`

- `parse_sentence(str) → dict`: Runs Stanza depparse, returns tokens, edges, deprels, POS tags
- `to_pyg_data(parse_result) → Data`: Converts parse to PyG Data object
- `align_subwords(stanza_tokens, bert_tokens) → list[int]`: Maps each BERT subword to Stanza word index (-1 for special tokens). Uses greedy character-span matching.
- `aggregate_subword_embeddings(bert_hidden, alignment, num_words) → Tensor`: Mean-pools BERT subwords per word.
- `batch_parse(sentences, output_dir)`: Parses many sentences, saves as `.jsonl` chunk files.

### `src/train.py`

- `load_hydra_config()`: Uses Hydra Compose API (NOT `@hydra.main()` — incompatible with SimCSE's HfArgumentParser)
- `config_to_simcse_args()`: Converts Hydra DictConfig → SimCSE's `ModelArguments`, `DataTrainingArguments`, `TrainingArguments`
- `SyntaxCLTrainer`: Training orchestrator with separate optimizer groups for BERT/GNN. Skeleton for standard and freeze-then-align strategies.

### `scripts/debug_train.py`

- Self-contained debug script that tests the full pipeline on CPU with toy data
- Does NOT require SimCSE, Stanza, or any downloaded data
- Uses `DebugBertForCL` wrapper that mimics SimCSE's `BertForCL` interface
- Validates: forward shapes, backward gradients, inference mode, NaN checks

---

## 4. Technical Constraints You MUST Follow

### Package Management

- **Use `uv` exclusively.** No `pip install`, no `conda`. All commands: `uv sync`, `uv run python ...`, `uv add <package>`.
- Dependencies are in `pyproject.toml`.
- SimCSE is an editable install from the `./SimCSE` submodule via `[tool.uv.sources]`.

### SimCSE Integration

- SimCSE is a **git submodule** at `./SimCSE/`. Do NOT modify files inside `SimCSE/` directly.
- SimCSE's `setup.py` has pinned `scipy<1.6` and `numpy<1.20` — patched by `scripts/setup_env.sh`.
- Import SimCSE as: `from simcse.models import BertForCL` or `from simcse.trainers import CLTrainer`.
- SimCSE's `BertForCL.forward()` has two modes:
  - `sent_emb=True` → inference (returns `BaseModelOutputWithPoolingAndCrossAttentions`)
  - `sent_emb=False` → training, expects input shape `(batch_size, num_sent, seq_len)` where `num_sent=2` for unsupervised

### Hydra Configuration

- Use **Compose API** (`hydra.initialize_config_dir()` + `hydra.compose()`), NOT `@hydra.main()`.
- Reason: `@hydra.main()` captures `sys.argv`, conflicting with HuggingFace's `HfArgumentParser`.
- Config root: `configs/config.yaml`. Experiment overrides: `experiment=stop_grad`, etc.

### Graph Neural Networks

- Use **PyTorch Geometric (PyG)**, not DGL.
- Both `GATConv` and `GCNConv` are implemented, selected via config `conv_type: gat|gcn`.
- Graph batching: `torch_geometric.data.Batch.from_data_list([...])`.
- Node features come from BERT hidden states, aligned via `StanzaSyntaxParser.align_subwords()`.

### Subword → Word Alignment

- Critical bridge between BERT (WordPiece subwords) and Stanza (word-level tokens).
- Strategy: greedy char-span matching → mean-pool subwords per word.
- Special tokens ([CLS], [SEP], [PAD]) map to `-1`.
- Implementation: `StanzaSyntaxParser.align_subwords()` and `aggregate_subword_embeddings()`.

---

## 5. Three Training Strategies

| Strategy              | Config                           | Description                                                                                          |
| --------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------- |
| `multi_loss`        | `experiment=multi_loss`        | Joint training from epoch 0.`L = L_SimCSE + λ·L_align + μ·L_GNN`                               |
| `stop_grad`         | `experiment=stop_grad`         | GNN is frozen. Gradients from `L_align` only update BERT. `h_G` is detached.                     |
| `freeze_then_align` | `experiment=freeze_then_align` | Phase 1: train GNN only (BERT frozen,`L_GNN`). Phase 2: joint training (BERT unfrozen, full loss). |

Each strategy is a Hydra YAML override in `configs/experiment/`. The strategy selection is handled in `SyntaxCLTrainer.train()`.

---

## 6. What Is Already Done

| Component                   | Status  | Notes                                                    |
| --------------------------- | ------- | -------------------------------------------------------- |
| Project structure & configs | ✅ Done | Hydra configs for all 3 strategies                       |
| `SyntaxGNNEncoder`        | ✅ Done | GAT + GCN, 3 pooling modes, residual connections         |
| `SyntaxBertModel` wrapper | ✅ Done | Forward pass, freeze/unfreeze, projection heads          |
| Loss functions              | ✅ Done | AlignmentLoss, GNNContrastiveLoss, CombinedLoss          |
| Syntax parser + alignment   | ✅ Done | Stanza pipeline, subword alignment, batch caching        |
| Debug training script       | ✅ Done | End-to-end test on CPU with toy data                     |
| Unit tests                  | ✅ Done | GNN shapes, losses, gradient flow, alignment             |
| `train.py` skeleton       | ✅ Done | Hydra → SimCSE bridge, optimizer groups, config loading |

---

## 7. What Remains (see TODO.md)

### Critical (must do before training)

1. **Graph-aware DataCollator**: A custom collator that loads pre-parsed dependency graphs alongside tokenized text, builds `PyG.Batch`, and returns both to the training loop.
2. **SimCSE `cl_forward()` integration**: Currently `wrapper.py` bypasses `cl_forward()`. Need to either hook into it or refactor so that `L_SimCSE` is computed via SimCSE's original logic AND hidden states are extracted for the GNN.
3. **Freeze-then-align phase transitions**: Epoch callback to switch phases mid-training.

### Important (for experiments)

4. STS evaluation pipeline (adapt `SimCSE/evaluation.py`)
5. Ablation Hydra configs (GCN vs GAT, layer count, λ/μ sweep)
6. Syntactic probing benchmarks
7. WandB logging of loss components

### Nice-to-have

8. Multi-GPU / DDP support for PyG batches
9. Multiprocessing for Stanza parsing (1M sentences)
10. Pre-tokenization caching

---

## 8. How to Run Things

```bash
bash activate_env.sh  # sets up uv environment

activate_env # activate the uv shell

# You might need to download data, look at what setup_env.sh does but do not dowload

# Quick pipeline test (no data/SimCSE needed)
uv run python scripts/debug_train.py
uv run python scripts/debug_train.py --conv-type gcn --steps 5

# Download data
uv run python scripts/download_wiki.py

# Parse syntax (slow — hours for 1M sentences)
uv run python -m src.processing.syntax_parser \
    --input data/wiki1m_for_simcse.txt \
    --output data/parsed_graphs/ \
    --max-sentences 1000  # for quick test

# Train
uv run python src/train.py --config-name config experiment=multi_loss

# Run tests
uv run pytest tests/ -v

# ── Full automated pipeline (train → evaluate → plot) ──────────────────────
# Default run (multi_loss, all defaults):
bash run_pipeline.sh

# Override any hyperparameter via env var:
EXPERIMENT=stop_grad BATCH_SIZE=32 LR=1e-5 LAMBDA_ALIGN=0.05 bash run_pipeline.sh

# Quick smoke test (small data, 1 epoch):
MAX_TRAIN_SAMPLES=100 BATCH_SIZE=8 NUM_EPOCHS=1 bash run_pipeline.sh

# The pipeline writes timestamped logs to logs/pipeline_<YYYYMMDD_HHMMSS>.log
# Evaluation JSON  → eval_results/eval_<EXPERIMENT>_<TIMESTAMP>.json
# Charts           → plots/
```

### `run_pipeline.sh` — Environment variable reference

| Variable | Default | Description |
| --- | --- | --- |
| `EXPERIMENT` | `multi_loss` | Hydra experiment config (`multi_loss` / `stop_grad` / `freeze_then_align`) |
| `BATCH_SIZE` | `64` | Per-device training batch size |
| `LR` | `3e-5` | BERT learning rate |
| `GNN_LR` | `1e-4` | GNN learning rate (separate optimizer group) |
| `LAMBDA_ALIGN` | `0.1` | Alignment loss weight λ |
| `MU_GNN` | `0.05` | GNN contrastive loss weight μ |
| `NUM_EPOCHS` | `3` | Number of training epochs |
| `OUTPUT_DIR` | `outputs/<EXPERIMENT>` | Where checkpoints are saved |
| `EVAL_RESULTS_DIR` | `eval_results` | Directory for evaluation JSON output |
| `PLOTS_DIR` | `plots` | Directory for generated charts |
| `LOG_FILE` | `logs/pipeline_<timestamp>.log` | Full pipeline log path |

The script sets `UV_PROJECT_ENVIRONMENT=/Data/edouard.rabasse/venvs/CS224N_project` (replicating `activate_env.sh`) and uses `uv run python` throughout — no manual venv activation needed.

---

## 9. Coding Standards

- **Python**: 3.10+, type hints everywhere, docstrings on all public methods.
- **Formatting**: `ruff` (line length 120, configured in `pyproject.toml`).
- **Imports**: Use absolute imports from project root (`from src.models.gnn_encoder import ...`).
- **No empty files**: Every `.py` file must have real code or at minimum a descriptive module docstring. Never create placeholder/stub files.
- **Config-driven**: All hyperparameters go in Hydra YAML configs, not hardcoded. Use `OmegaConf.to_container(cfg, resolve=True)` to convert to plain dicts.
- **Testing**: Add tests for new modules in `tests/`. Run with `uv run pytest tests/ -v`.

---

## 10. Key Design Decisions (Rationale)

These decisions were made deliberately. Do NOT change them without discussion:

1. **SimCSE as submodule** (not vendored, not forked): Keeps upstream clean, allows `git pull` updates.
2. **Hydra Compose API** (not `@hydra.main()`): Required because SimCSE uses `HfArgumentParser` which also reads `sys.argv`.
3. **PyG** (not DGL): Wider ecosystem, better batching API, more community support.
4. **Mean-pool subword alignment** (not first-subword): Preserves information from all subwords.
5. **Separate LR for GNN** (higher than BERT): GNN learns from scratch while BERT is fine-tuned — different learning rate scales.
6. **Projection heads for alignment**: Following SimCLR/BYOL pattern — prevents alignment loss from collapsing the representation space.
7. **GNN discarded at inference**: The whole point — BERT should internalize syntax during training so it doesn't need the GNN later.

---

## 11. Common Pitfalls

- **SimCSE input shape**: `cl_forward()` expects `(batch_size, num_sent, seq_len)` NOT `(batch_size, seq_len)`. The `num_sent` dimension is 2 for unsupervised (same sentence passed twice with different dropout masks).
- **Stanza tokenization ≠ BERT tokenization**: Always use `align_subwords()` to bridge them. Never assume 1:1 mapping.
- **PyG `Batch.batch`**: This is the node-to-graph assignment vector, NOT the batch size. Don't confuse it with the batch dimension.
- **SimCSE's `Pooler` needs `output_hidden_states=True`** for `avg_top2` and `avg_first_last` pooling modes.
- **`transformers.file_utils` is deprecated**: SimCSE imports from it. Modern transformers moved these to `transformers.utils`. This may cause import errors with transformers ≥4.30.

---

*Last updated: February 26, 2026*