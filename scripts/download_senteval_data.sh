#!/usr/bin/env bash
# =============================================================================
# download_senteval_data.sh — Download SentEval evaluation datasets
# =============================================================================
# Usage: bash scripts/download_senteval_data.sh
#
# Downloads all STS and transfer task datasets required by SentEval:
#   STS12-16, STSBenchmark, SICKRelatedness, MR, CR, MPQA, SUBJ, SST2, TREC, MRPC
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SENTEVAL_DATA_DIR="$PROJECT_ROOT/SimCSE/SentEval/data/downstream"

echo "============================================="
echo "  SentEval Data Download"
echo "============================================="

# Check that SentEval submodule is initialized
if [ ! -f "$PROJECT_ROOT/SimCSE/SentEval/setup.py" ]; then
    echo "  ✗ SentEval not found. Run first:"
    echo "    git submodule update --init --recursive"
    exit 1
fi

if [ ! -f "$SENTEVAL_DATA_DIR/get_transfer_data.bash" ]; then
    echo "  ✗ get_transfer_data.bash not found at $SENTEVAL_DATA_DIR"
    echo "    Make sure the SentEval submodule is fully initialized."
    exit 1
fi

echo ""
echo "Downloading SentEval datasets into:"
echo "  $SENTEVAL_DATA_DIR"
echo ""

cd "$SENTEVAL_DATA_DIR"
bash get_transfer_data.bash

echo ""
echo "============================================="
echo "  SentEval data download complete!"
echo ""
echo "  You can now run evaluation:"
echo "    uv run python -m src.evaluation \\"
echo "        --model_name_or_path <your_model> \\"
echo "        --pooler cls_before_pooler \\"
echo "        --mode dev \\"
echo "        --task_set sts"
echo "============================================="
