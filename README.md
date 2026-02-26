# Transferring Syntactic Structure into Sentence Embeddings using GNNs

> **CS224N Project — Stanford University**

## Abstract

This project improves BERT's syntactic understanding through contrastive alignment with a Graph Neural Network that encodes dependency parse trees. Unlike SynCSE (which fuses embeddings at inference), our approach uses alignment during training so that BERT **internalizes** syntactic structure — at inference time, only BERT is needed.

**Core idea**: Train BERT alongside a GNN that encodes dependency trees. A contrastive loss aligns their representations, forcing BERT to internalize syntactic information. Then discard the GNN at inference.

## Architecture

```
                    ┌──────────────────────┐
                    │   Raw Sentence       │
                    │ "The cat sat on mat" │
                    └─────────┬────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
    ┌─────────────────┐             ┌──────────────────┐
    │   BERT Encoder  │             │  Stanza Parser   │
    │ (bert-base)     │             │  (dep. tree)     │
    └────────┬────────┘             └────────┬─────────┘
             │                               │
             ▼                               ▼
    ┌─────────────────┐             ┌──────────────────┐
    │  h_B (pooled)   │             │  GNN Encoder     │
    │  (768-dim)      │◄── align ──►│  GAT/GCN (PyG)   │
    └────────┬────────┘             └────────┬─────────┘
             │                               │
             ▼                               ▼
         L_SimCSE                     h_G (768-dim)
             │                               │
             └───────────┬───────────────────┘
                         ▼
              L = L_SimCSE + λ·L_align + μ·L_GNN
```

**At inference**: Only the BERT branch is used. The GNN is discarded.

## Quick Start

### 1. Clone & Setup

```bash
# Clone with SimCSE submodule
git clone --recurse-submodules <your-repo-url>
cd project

# Full automated setup (uv, SimCSE patching, Stanza models)
bash scripts/setup_env.sh
```

Or manually:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Initialize submodule
git submodule update --init --recursive

# Install all dependencies
uv sync

# Download Stanza English model
uv run python -c "import stanza; stanza.download('en', processors='tokenize,pos,lemma,depparse')"
```

### 2. Download Data

```bash
# Download Wikipedia 1M sentences (~120 MB)
uv run python scripts/download_wiki.py
```

### 3. Parse Syntactic Dependencies

```bash
# Parse all 1M sentences into dependency graphs (~2-4 hours on CPU, ~30 min on GPU)
uv run python -m src.processing.syntax_parser \
    --input data/wiki1m_for_simcse.txt \
    --output data/parsed_graphs/ \
    --chunk-size 10000

# For a quick test with fewer sentences:
uv run python -m src.processing.syntax_parser \
    --input data/wiki1m_for_simcse.txt \
    --output data/parsed_graphs/ \
    --max-sentences 1000 \
    --gpu
```

### 4. Train

```bash
# Default: multi-loss joint training strategy
uv run python src/train.py --config-name config

# Stop-gradient strategy (GNN frozen, only BERT learns from alignment)
uv run python src/train.py --config-name config experiment=stop_grad

# Freeze-then-align (2-phase: warm up GNN, then align BERT)
uv run python src/train.py --config-name config experiment=freeze_then_align

# Override any parameter via Hydra
uv run python src/train.py --config-name config \
    experiment=multi_loss \
    training.learning_rate=5e-5 \
    model.gnn.conv_type=gcn \
    model.gnn.num_layers=3 \
    model.alignment.lambda_align=0.2
```

### 5. Download SentEval Data

SentEval evaluation datasets are not included in the repo and must be downloaded separately:

```bash
bash scripts/download_senteval_data.sh
```

This downloads all STS and transfer task datasets (STS12-16, STSBenchmark, SICKRelatedness, MR, CR, MPQA, SUBJ, SST2, TREC, MRPC) into `SimCSE/SentEval/data/downstream/`.

> **Note**: Requires the SimCSE submodule to be initialized first (`git submodule update --init --recursive`).

### 6. Evaluate

```bash
# Evaluate on STS benchmarks (dev mode, fast)
uv run python -m src.evaluation \
    --model_name_or_path outputs/multi_loss/ \
    --pooler cls_before_pooler \
    --mode dev \
    --task_set sts

# Full test evaluation
uv run python -m src.evaluation \
    --model_name_or_path outputs/multi_loss/ \
    --pooler cls_before_pooler \
    --mode test \
    --task_set full
```

## Project Structure

```
.
├── pyproject.toml              # uv project config & dependencies
├── .gitignore
├── .gitmodules                 # SimCSE as git submodule
├── README.md                   # ← You are here
├── TODO.md                     # Remaining work items
├── SimCSE/                     # Git submodule (princeton-nlp/SimCSE)
│
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Root config (training, data, defaults)
│   ├── model/
│   │   ├── bert.yaml           # BERT params (model, pooler, temp)
│   │   ├── gnn.yaml            # GNN params (conv_type, layers, heads)
│   │   └── alignment.yaml      # Alignment params (λ, μ, temperature)
│   └── experiment/
│       ├── stop_grad.yaml      # Strategy: frozen GNN, align BERT only
│       ├── freeze_then_align.yaml  # Strategy: 2-phase training
│       └── multi_loss.yaml     # Strategy: joint L_SimCSE + λL_align + μL_GNN
│
├── data/                       # Datasets (gitignored)
│   ├── wiki1m_for_simcse.txt   # 1M Wikipedia sentences
│   └── parsed_graphs/          # Cached dependency parses (.jsonl)
│
├── scripts/
│   ├── setup_env.sh            # Full env setup (uv + SimCSE + Stanza)
│   └── download_wiki.py        # Download wiki1m from HuggingFace
│
├── src/
│   ├── train.py                # Main entry — Hydra Compose API → training
│   ├── alignment/
│   │   └── losses.py           # AlignmentLoss (NT-Xent), CombinedLoss
│   ├── models/
│   │   ├── gnn_encoder.py      # SyntaxGNNEncoder (GAT/GCN + readout)
│   │   └── wrapper.py          # SyntaxBertModel (BERT + GNN wrapper)
│   └── processing/
│       └── syntax_parser.py    # Stanza pipeline → PyG Data objects
│
└── tests/
    └── test_gnn_encoder.py     # GNN, loss, and alignment smoke tests
```

## Training Strategies

| Strategy | Description | Loss |
|---|---|---|
| `multi_loss` | Joint training from epoch 0 | $L = L_\text{SimCSE} + \lambda L_\text{align} + \mu L_\text{GNN}$ |
| `stop_grad` | GNN frozen, alignment updates BERT only | $L = L_\text{SimCSE} + \lambda L_\text{align}(h_B, \text{sg}(h_G))$ |
| `freeze_then_align` | Phase 1: GNN warmup; Phase 2: joint | Phase 1: $L_\text{GNN}$, Phase 2: full combined |

## Configuration System

All hyperparameters are managed via [Hydra](https://hydra.cc/) YAML configs. The root config at `configs/config.yaml` composes from model and experiment configs:

```yaml
defaults:
  - model/bert: bert
  - model/gnn: gnn
  - model/alignment: alignment
  - experiment: multi_loss
```

Override any parameter from the command line:
```bash
uv run python src/train.py model.gnn.conv_type=gcn training.num_train_epochs=5
```

## Key Design Decisions

1. **Hydra Compose API** — SimCSE uses `HfArgumentParser` which conflicts with `@hydra.main()`. We use `hydra.compose()` and manually construct SimCSE's dataclasses.

2. **Subword → Word Alignment** — BERT's WordPiece tokenization differs from Stanza's word tokenization. We align via greedy character-span matching and mean-pool subword embeddings per word for GNN node features.

3. **SimCSE as Submodule** — Keeps upstream intact for future pulls. Version-pinned deps are patched via `setup_env.sh`.

4. **Inference = BERT only** — After training, the GNN branch is discarded. The hypothesis is that BERT has internalized syntactic structure through alignment.

## References

- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) (Gao et al., 2021)
- [SynCSE: Syntax-enhanced Contrastive Learning for Sentence Embedding](https://arxiv.org/abs/2301.02507)
- [Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al., 2018)
- [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (Kipf & Welling, 2017)
