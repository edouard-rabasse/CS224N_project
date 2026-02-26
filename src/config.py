"""
config.py — Hydra config loading and SimCSE argument bridge.

Provides:
- load_hydra_config: loads YAML configs via Hydra's Compose API (avoids
  the sys.argv conflict with HfArgumentParser from @hydra.main())
- config_to_simcse_args: converts OmegaConf DictConfig into the three
  dataclass-like argument objects expected by SimCSE / HuggingFace Trainer
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from omegaconf import DictConfig, OmegaConf

# Absolute path to the project root (one level above src/)
_PROJECT_ROOT = Path(__file__).parent.parent


def _resolve(path: str) -> str:
    """Resolve a relative path against the project root."""
    p = Path(path)
    return str(p if p.is_absolute() else _PROJECT_ROOT / p)


def load_hydra_config(config_name: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """Load Hydra configuration using the Compose API.

    Args:
        config_name: Name of the root config file (without .yaml extension).
        overrides: Hydra overrides, e.g. ["experiment=stop_grad", "training.lr=1e-4"].

    Returns:
        OmegaConf DictConfig with the merged configuration.
    """
    from hydra import compose, initialize_config_dir

    config_dir = str(Path(__file__).parent.parent / "configs")

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])

    return cfg


def config_to_simcse_args(cfg: DictConfig) -> tuple:
    """Convert Hydra DictConfig to SimCSE's argument objects.

    SimCSE expects three argument objects:
    - ModelArguments: model_name_or_path, pooler_type, temp, etc.
    - DataTrainingArguments: train_file, max_seq_length, etc.
    - OurTrainingArguments: learning_rate, num_train_epochs, etc.

    Returns:
        Tuple of (model_args, data_args, training_args).
    """
    from transformers import TrainingArguments

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

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True) if "data" in cfg else {}
    training_cfg = OmegaConf.to_container(cfg.training, resolve=True) if "training" in cfg else {}
    data_args = SimpleNamespace(
        train_file=_resolve(data_cfg.get("train_file", "data/wiki1m_for_simcse.txt")),
        max_seq_length=training_cfg.get("max_seq_length", 32),
        pad_to_max_length=False,
        mlm_probability=0.15,
        preprocessing_num_workers=data_cfg.get("preprocessing_num_workers", 8),
        max_train_samples=data_cfg.get("max_train_samples", None),
        overwrite_cache=False,
    )

    training_args = TrainingArguments(
        output_dir=_resolve(training_cfg.get("output_dir", "outputs/default")),
        num_train_epochs=training_cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 64),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 256),
        learning_rate=training_cfg.get("learning_rate", 3e-5),
        warmup_steps=training_cfg.get("warmup_steps", 0),
        weight_decay=training_cfg.get("weight_decay", 0.0),
        fp16=training_cfg.get("fp16", True),
        logging_steps=training_cfg.get("logging_steps", 100),
        eval_steps=training_cfg.get("eval_steps", 250),
        save_steps=training_cfg.get("save_steps", 250),
        save_total_limit=training_cfg.get("save_total_limit", 2),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        seed=training_cfg.get("seed", 42),
        eval_strategy="no",
        remove_unused_columns=False,
        do_train=True,
    )

    return model_args, data_args, training_args
