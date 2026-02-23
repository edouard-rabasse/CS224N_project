#!/usr/bin/env python3
"""
download_wiki.py — Download the Wikipedia 1M sentences dataset for SimCSE.

Downloads `wiki1m_for_simcse.txt` from the HuggingFace Hub
(princeton-nlp/datasets-for-simcse) into the local `data/` directory.

Usage:
    uv run python scripts/download_wiki.py
    uv run python scripts/download_wiki.py --output data/wiki1m_for_simcse.txt
"""

import argparse
import os
from pathlib import Path


def download_wiki(output_path: str) -> None:
    """Download wiki1m_for_simcse.txt from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        line_count = sum(1 for _ in open(output_path, encoding="utf-8"))
        print(f"File already exists: {output_path} ({line_count:,} lines)")
        return

    print("Downloading wiki1m_for_simcse.txt from HuggingFace Hub...")
    print("  Repo: princeton-nlp/datasets-for-simcse")
    print(f"  Dest: {output_path}")

    downloaded_path = hf_hub_download(
        repo_id="princeton-nlp/datasets-for-simcse",
        filename="wiki1m_for_simcse.txt",
        repo_type="dataset",
        local_dir=str(output_path.parent),
        local_dir_use_symlinks=False,
    )

    # hf_hub_download may place the file in a subfolder; ensure it's at the expected path
    downloaded_path = Path(downloaded_path)
    if downloaded_path != output_path and downloaded_path.exists():
        downloaded_path.rename(output_path)

    # Verify
    if not output_path.exists():
        raise FileNotFoundError(f"Download failed — {output_path} not found")

    line_count = sum(1 for _ in open(output_path, encoding="utf-8"))
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Downloaded: {output_path}")
    print(f"    {line_count:,} lines, {size_mb:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wikipedia 1M dataset for SimCSE")
    parser.add_argument(
        "--output",
        type=str,
        default="data/wiki1m_for_simcse.txt",
        help="Output file path (default: data/wiki1m_for_simcse.txt)",
    )
    args = parser.parse_args()

    download_wiki(args.output)


if __name__ == "__main__":
    main()
