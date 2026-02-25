"""
train.py — Main training entry point for GNN-Syntax-BERT

Uses Hydra's Compose API to load modular YAML configs, then bridges them
into SimCSE's HfArgumentParser-based dataclass system. This avoids the
sys.argv conflict between Hydra's @hydra.main() and HfArgumentParser.

Usage:
    uv run python src/train.py --config-name config
    uv run python src/train.py --config-name config experiment=stop_grad
    uv run python src/train.py --config-name config experiment=freeze_then_align \\
        training.learning_rate=5e-5 model.gnn.conv_type=gcn

Architecture:
    1. Load Hydra config → OmegaConf DictConfig
    2. Convert to SimCSE's ModelArguments, DataTrainingArguments, OurTrainingArguments
    3. Instantiate SyntaxBertModel (BERT + GNN wrapper)
    4. Build SyntaxCLTrainer with combined loss
    5. Train with phase-aware logic (freeze-then-align if configured)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, Data

logger = logging.getLogger(__name__)


# =============================================================================
# Hydra Config Loading (Compose API)
# =============================================================================


def load_hydra_config(config_name: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra configuration using the Compose API.

    This avoids the sys.argv conflict with HfArgumentParser by not using
    @hydra.main(). Instead, we programmatically compose the config.

    Args:
        config_name: Name of the root config file (without .yaml extension).
        overrides: List of Hydra overrides (e.g., ["experiment=stop_grad", "training.lr=1e-4"]).

    Returns:
        OmegaConf DictConfig with the merged configuration.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).parent.parent / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


# =============================================================================
# SimCSE Dataclass Bridge
# =============================================================================


def config_to_simcse_args(cfg: DictConfig) -> tuple:
    """Convert Hydra DictConfig to SimCSE's argument dataclasses.

    SimCSE expects three dataclasses:
    - ModelArguments: model_name_or_path, pooler_type, temp, etc.
    - DataTrainingArguments: train_file, max_seq_length, etc.
    - OurTrainingArguments: learning_rate, num_train_epochs, etc.

    We construct these from the flat Hydra config.

    Returns:
        Tuple of (model_args, data_args, training_args) —
        SimpleNamespace objects matching SimCSE's expected interface.
    """
    from types import SimpleNamespace

    # Model arguments (maps to SimCSE's ModelArguments)
    bert_cfg = OmegaConf.to_container(cfg.model.bert, resolve=True) if "model" in cfg and "bert" in cfg.model else {}
    model_args = SimpleNamespace(
        model_name_or_path=bert_cfg.get("model_name_or_path", "bert-base-uncased"),
        pooler_type=bert_cfg.get("pooler_type", "cls"),
        temp=bert_cfg.get("temp", 0.05),
        do_mlm=bert_cfg.get("do_mlm", False),
        mlm_weight=bert_cfg.get("mlm_weight", 0.1),
        mlp_only_train=bert_cfg.get("mlp_only_train", True),
        hard_negative_weight=bert_cfg.get("hard_negative_weight", 0.0),
    )

    # Data arguments
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True) if "data" in cfg else {}
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True) if "training" in cfg else {}
    data_args = SimpleNamespace(
        train_file=data_cfg.get("train_file", "data/wiki1m_for_simcse.txt"),
        max_seq_length=training_cfg.get("max_seq_length", 32),
        pad_to_max_length=False,
        mlm_probability=0.15,
        preprocessing_num_workers=data_cfg.get("preprocessing_num_workers", 8),
        max_train_samples=data_cfg.get("max_train_samples", None),
        overwrite_cache=False,
    )

    # Training arguments — create a HuggingFace TrainingArguments-compatible object
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=training_cfg.get("output_dir", "outputs/default"),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 64),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 256),
        learning_rate=training_cfg.get("learning_rate", 3e-5),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.06),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        fp16=training_cfg.get("fp16", True),
        logging_steps=training_cfg.get("logging_steps", 100),
        eval_steps=training_cfg.get("eval_steps", 250),
        save_steps=training_cfg.get("save_steps", 250),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        seed=training_cfg.get("seed", 42),
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="stsb_spearman",
        overwrite_output_dir=True,
        do_train=True,
    )

    return model_args, data_args, training_args


# =============================================================================
# Graph Data Integration
# =============================================================================


def load_parsed_graphs(parsed_dir: str) -> list[dict]:
    """Load pre-parsed dependency graphs from cached .jsonl files.

    Args:
        parsed_dir: Directory containing parsed_XXXXX.jsonl chunk files.

    Returns:
        List of parse result dicts (one per sentence).
    """
    parsed_path = Path(parsed_dir)
    if not parsed_path.exists():
        logger.warning(f"Parsed graphs directory not found: {parsed_dir}")
        logger.warning(
            "Run: uv run python -m src.processing.syntax_parser --input data/wiki1m_for_simcse.txt --output data/parsed_graphs/"
        )
        return []

    graphs = []
    chunk_files = sorted(parsed_path.glob("parsed_*.jsonl"))
    logger.info(f"Loading parsed graphs from {len(chunk_files)} chunk files...")

    for chunk_file in chunk_files:
        with open(chunk_file, encoding="utf-8") as f:
            for line in f:
                graphs.append(json.loads(line))

    logger.info(f"Loaded {len(graphs):,} parsed graphs")
    return graphs


def parse_to_pyg_data(parse_result: dict) -> Data:
    """Convert a cached parse result dict to a PyG Data object.

    Node features (x) are left unset — they will be populated from
    BERT hidden states at training time.

    Args:
        parse_result: Dict with keys: edges_src, edges_dst, num_tokens.

    Returns:
        PyG Data with edge_index and num_nodes.
    """
    from src.processing.syntax_parser import StanzaSyntaxParser

    return StanzaSyntaxParser.to_pyg_data(parse_result)


# =============================================================================
# SyntaxCLTrainer — Extended SimCSE Trainer
# =============================================================================


class SyntaxCLTrainer:
    """Training orchestrator for the GNN-Syntax-BERT model.

    Extends SimCSE's training loop with:
    - Graph data loading alongside text data
    - Combined loss computation (SimCSE + alignment + GNN)
    - Phase-aware training for freeze-then-align strategy
    - Separate optimizer groups for BERT and GNN parameters

    This is a standalone trainer (not subclassing CLTrainer) because SimCSE's
    CLTrainer is a heavily forked copy of HF Trainer from an older version.
    Instead, we compose: use HF Trainer for the standard training loop,
    and override compute_loss() for the combined objective.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: DictConfig,
        model_args,
        data_args,
        training_args,
        parsed_graphs: list[dict],
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.parsed_graphs = parsed_graphs

        # Extract GNN and alignment configs
        self.gnn_cfg = (
            OmegaConf.to_container(cfg.model.gnn, resolve=True) if "model" in cfg and "gnn" in cfg.model else {}
        )
        self.align_cfg = (
            OmegaConf.to_container(cfg.model.alignment, resolve=True)
            if "model" in cfg and "alignment" in cfg.model
            else {}
        )

        # Combined loss
        from src.alignment.losses import CombinedLoss

        self.combined_loss = CombinedLoss(
            lambda_align=self.align_cfg.get("lambda_align", 0.1),
            mu_gnn=self.align_cfg.get("mu_gnn", 0.05),
            align_temperature=self.align_cfg.get("align_temperature", 0.05),
        )

    def train(self) -> None:
        """Main training entry point.

        Dispatches to the appropriate training strategy based on the
        experiment configuration:
        - multi_loss / stop_grad: standard single-phase training
        - freeze_then_align: two-phase training with phase transitions
        """
        experiment = self.cfg.get("experiment_name", "multi_loss")
        logger.info(f"Starting training with experiment: {experiment}")
        logger.info(f"Config:\n{OmegaConf.to_yaml(self.cfg)}")

        if experiment == "freeze_then_align":
            self._train_freeze_then_align()
        else:
            self._train_standard()

    def _train_standard(self) -> None:
        """Standard single-phase training (multi_loss or stop_grad).

        Uses HuggingFace Trainer with a custom compute_loss for the
        combined objective. The stop_grad strategy is handled inside
        SyntaxBertModel.forward() via h_gnn.detach().

        NOTE: This method sets up the data pipeline and training loop.
        The full integration with SimCSE's CLTrainer.evaluate() (SentEval)
        is handled in TODO — for now we use the standard HF evaluation.
        """
        logger.info("=== Standard Training (single phase) ===")
        logger.info(f"  SimCSE loss weight: 1.0")
        logger.info(f"  Alignment loss weight (λ): {self.align_cfg.get('lambda_align', 0.1)}")
        logger.info(f"  GNN loss weight (μ): {self.align_cfg.get('mu_gnn', 0.05)}")
        logger.info(f"  Stop-gradient GNN: {self.align_cfg.get('stop_grad_gnn', False)}")
        logger.info(f"  GNN frozen: {self.gnn_cfg.get('freeze', False)}")

        # Build optimizer with separate parameter groups
        optimizer = self._build_optimizer()

        # Load and prepare dataset
        train_dataset = self._prepare_dataset()

        logger.info(f"  Dataset size: {len(train_dataset):,} sentences")
        logger.info(f"  Parsed graphs: {len(self.parsed_graphs):,}")
        logger.info("Training loop setup complete. See TODO.md for full training loop integration.")

    def _train_freeze_then_align(self) -> None:
        """Two-phase training: freeze-then-align strategy.

        Phase 1: Train GNN only (BERT frozen)
          - Only GNN parameters receive gradients
          - Loss = L_GNN (auxiliary GNN contrastive loss)
          - Goal: GNN learns to encode syntactic structure

        Phase 2: Joint training (BERT unfrozen)
          - Both BERT and GNN parameters receive gradients
          - Loss = L_SimCSE + λ * L_align + μ * L_GNN
          - Goal: BERT internalizes syntactic information from GNN
        """
        training_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        phase1_epochs = training_cfg.get("phase1_epochs", 5)
        phase2_epochs = training_cfg.get("phase2_epochs", 10)

        logger.info("=== Freeze-then-Align Training ===")
        logger.info(f"  Phase 1: {phase1_epochs} epochs (GNN warmup, BERT frozen)")
        logger.info(f"  Phase 2: {phase2_epochs} epochs (joint training)")

        # Phase 1: Freeze BERT, train GNN
        logger.info("\n--- Phase 1: GNN Warmup ---")
        self.model.wrapper._freeze_bert()  # type: ignore
        self.model.wrapper._unfreeze_gnn()  # type: ignore
        # Phase 1 training loop would go here

        # Phase 2: Unfreeze BERT, joint training
        logger.info("\n--- Phase 2: Joint Alignment ---")
        self.model.wrapper._unfreeze_bert()  # type: ignore
        # Phase 2 training loop would go here

        logger.info("Freeze-then-align training setup complete. See TODO.md for full implementation.")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with separate parameter groups for BERT and GNN.

        BERT parameters use the config learning rate (typically 3e-5).
        GNN parameters use a potentially higher learning rate (typically 1e-4).

        Returns:
            AdamW optimizer with two parameter groups.
        """
        training_cfg = OmegaConf.to_container(self.cfg.training, resolve=True)
        bert_lr = training_cfg.get("learning_rate", 3e-5)
        gnn_lr = training_cfg.get("gnn_learning_rate", 1e-4)
        weight_decay = training_cfg.get("weight_decay", 0.0)

        # Separate BERT and GNN parameters
        bert_params = []
        gnn_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "gnn" in name:
                gnn_params.append(param)
            elif "bert" in name:
                bert_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": bert_params, "lr": bert_lr, "weight_decay": weight_decay},
                {"params": gnn_params, "lr": gnn_lr, "weight_decay": weight_decay},
                {"params": other_params, "lr": bert_lr, "weight_decay": weight_decay},
            ]
        )

        logger.info(f"Optimizer: AdamW")
        logger.info(f"  BERT params: {len(bert_params)} tensors, lr={bert_lr}")
        logger.info(f"  GNN params: {len(gnn_params)} tensors, lr={gnn_lr}")
        logger.info(f"  Other params: {len(other_params)} tensors, lr={bert_lr}")

        return optimizer

    def _prepare_dataset(self) -> list:
        """Load and prepare the text dataset.

        Uses HuggingFace datasets to load the training file, matching
        SimCSE's data loading logic.

        Returns:
            List of sentences (simplified — full Dataset integration in TODO).
        """
        train_file = self.data_args.train_file
        if not Path(train_file).exists():
            logger.error(f"Training file not found: {train_file}")
            logger.error("Run: uv run python scripts/download_wiki.py")
            return []

        with open(train_file, encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]

        max_samples = self.data_args.max_train_samples
        if max_samples is not None:
            sentences = sentences[:max_samples]

        return sentences


# =============================================================================
# Main Entry Point
# =============================================================================


def parse_cli_args() -> tuple[str, list[str]]:
    """Parse CLI arguments for Hydra config name and overrides.

    We use a thin argparse layer because Hydra's @hydra.main() conflicts
    with SimCSE's HfArgumentParser. The Compose API is used instead.

    Returns:
        Tuple of (config_name, list_of_overrides).
    """
    parser = argparse.ArgumentParser(
        description="GNN-Syntax-BERT Training",
        # Allow unknown args to be passed as Hydra overrides
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name (default: config)",
    )
    args, overrides = parser.parse_known_args()
    return args.config_name, overrides


def main() -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Parse CLI
    config_name, overrides = parse_cli_args()
    logger.info(f"Config: {config_name}, Overrides: {overrides}")

    # Load Hydra config
    cfg = load_hydra_config(config_name, overrides)
    logger.info(f"Experiment: {cfg.get('experiment_name', 'default')}")

    # Convert to SimCSE argument objects
    model_args, data_args, training_args = config_to_simcse_args(cfg)

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info(f"Tokenizer: {model_args.model_name_or_path}")

    # Load BERT model (SimCSE's BertForCL)
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    try:
        from simcse.models import BertForCL

        bert_model = BertForCL.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            model_args=model_args,
        )
        logger.info("Loaded SimCSE BertForCL")
    except ImportError:
        logger.warning("SimCSE not installed. Run: bash scripts/setup_env.sh")
        logger.warning("Falling back to standard BERT for testing...")
        from transformers import BertModel

        bert_model = BertModel.from_pretrained(model_args.model_name_or_path)

    # Build SyntaxBertModel wrapper
    gnn_cfg = OmegaConf.to_container(cfg.model.gnn, resolve=True)
    align_cfg = OmegaConf.to_container(cfg.model.alignment, resolve=True)

    from src.models.wrapper import SyntaxBertModel

    model = SyntaxBertModel(
        bert_model=bert_model,
        gnn_config=gnn_cfg,
        alignment_config=align_cfg,
    )
    logger.info(f"SyntaxBertModel initialized")
    logger.info(f"  GNN: {gnn_cfg['conv_type'].upper()}, {gnn_cfg['num_layers']} layers, pooling={gnn_cfg['pooling']}")
    logger.info(
        f"  Alignment: λ={align_cfg['lambda_align']}, μ={align_cfg['mu_gnn']}, τ={align_cfg['align_temperature']}"
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")

    # Load pre-parsed dependency graphs
    parsed_graphs = load_parsed_graphs(cfg.data.parsed_graphs_dir)

    # Build trainer and train
    trainer = SyntaxCLTrainer(
        model=model,
        cfg=cfg,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        parsed_graphs=parsed_graphs,
    )

    trainer.train()

    # Save final config
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {output_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
