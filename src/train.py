"""
train.py — Main training entry point for GNN-Syntax-BERT.

Usage:
    uv run python src/train.py --config-name config
    uv run python src/train.py --config-name config experiment=stop_grad
    uv run python src/train.py --config-name config experiment=freeze_then_align \\
        training.learning_rate=5e-5 model.gnn.conv_type=gcn

Architecture:
    1. Load Hydra config → OmegaConf DictConfig          (src/config.py)
    2. Convert to SimCSE argument objects                 (src/config.py)
    3. Instantiate SyntaxBertModel (BERT + GNN)           (src/models/wrapper.py)
    4. Build SyntaxCLTrainer with combined loss           (src/trainer.py)
    5. Train with phase-aware logic                       (src/trainer.py)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def parse_cli_args() -> tuple[str, list[str]]:
    """Parse CLI arguments for Hydra config name and overrides.

    We use a thin argparse layer because Hydra's @hydra.main() conflicts
    with SimCSE's HfArgumentParser. The Compose API is used instead.

    Returns:
        Tuple of (config_name, list_of_overrides).
    """
    parser = argparse.ArgumentParser(description="GNN-Syntax-BERT Training")
    parser.add_argument(
        "--config-name",
        type=str,
        default="config",
        help="Hydra config name (default: config)",
    )
    args, overrides = parser.parse_known_args()
    return args.config_name, overrides


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.config import config_to_simcse_args, load_hydra_config
    from src.models.wrapper import SyntaxBertModel
    from src.processing.graph_loader import load_parsed_graphs
    from src.trainer import SyntaxCLTrainer

    config_name, overrides = parse_cli_args()
    logger.info(f"Config: {config_name}, Overrides: {overrides}")

    cfg = load_hydra_config(config_name, overrides)
    logger.info(f"Experiment: {cfg.get('experiment_name', 'default')}")

    model_args, data_args, training_args = config_to_simcse_args(cfg)

    # Load BERT (SimCSE's BertForCL if available, otherwise vanilla BERT)
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    logger.info(f"Tokenizer: {model_args.model_name_or_path}")

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

    gnn_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.model.gnn, resolve=True))
    align_cfg = cast(dict[str, Any], OmegaConf.to_container(cfg.model.alignment, resolve=True))

    model = SyntaxBertModel(
        bert_model=bert_model,
        gnn_config=gnn_cfg,
        alignment_config=align_cfg,
    )
    logger.info("SyntaxBertModel initialized")
    logger.info(f"  GNN: {gnn_cfg['conv_type'].upper()}, {gnn_cfg['num_layers']} layers, pooling={gnn_cfg['pooling']}")
    logger.info(f"  Alignment: λ={align_cfg['lambda_align']}, μ={align_cfg['mu_gnn']}, τ={align_cfg['align_temperature']}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")

    parsed_graphs = load_parsed_graphs(str(Path(__file__).parent.parent / cfg.data.parsed_graphs_dir))

    trainer = SyntaxCLTrainer(
        model=model,
        cfg=cfg,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        parsed_graphs=parsed_graphs,
    )
    trainer.train()

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Config saved to {output_dir / 'config.yaml'}")


if __name__ == "__main__":
    main()
