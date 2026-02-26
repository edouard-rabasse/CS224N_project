#!/usr/bin/env bash
# =============================================================================
# setup_env.sh — Full project environment setup
# =============================================================================
# Usage: bash scripts/setup_env.sh
#
# This script:
#   1. Initializes the SimCSE git submodule
#   2. Patches SimCSE's setup.py to relax pinned dependency versions
#   3. Installs all dependencies via uv sync
#   4. Downloads the Stanza English model for dependency parsing
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "============================================="
echo "  GNN-Syntax-BERT — Environment Setup"
echo "============================================="

# ---- Step 1: SimCSE Submodule ----
echo ""
echo "[1/4] Initializing SimCSE submodule..."
cd "$PROJECT_ROOT"

if [ ! -f "SimCSE/setup.py" ]; then
    git submodule update --init --recursive
    echo "  ✓ SimCSE submodule initialized"
else
    echo "  ✓ SimCSE submodule already present"
fi

# ---- Step 2: Patch SimCSE's setup.py ----
echo ""
echo "[2/4] Patching SimCSE/setup.py to relax version constraints..."

SIMCSE_SETUP="$PROJECT_ROOT/SimCSE/setup.py"
if [ -f "$SIMCSE_SETUP" ]; then
    # Relax scipy pin: scipy>=1.5.4,<1.6 → scipy>=1.5.4
    sed -i.bak "s/'scipy>=1.5.4,<1.6'/'scipy>=1.5.4'/g" "$SIMCSE_SETUP"
    # Relax numpy pin: numpy>=1.19.5,<1.20 → numpy>=1.19.5
    sed -i.bak "s/'numpy>=1.19.5,<1.20'/'numpy>=1.19.5'/g" "$SIMCSE_SETUP"
    # Remove backup files
    rm -f "${SIMCSE_SETUP}.bak"
    echo "  ✓ Relaxed scipy and numpy version pins"
else
    echo "  ⚠ SimCSE/setup.py not found — skipping patch"
    echo "    Run: git submodule add https://github.com/princeton-nlp/SimCSE.git SimCSE"
fi

# ---- Step 3: Install dependencies ----
echo ""
echo "[3/4] Installing dependencies with uv..."

if ! command -v uv &> /dev/null; then
    echo "  ✗ uv not found. Install it first:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

cd "$PROJECT_ROOT"
uv sync
echo "  ✓ All dependencies installed"

# ---- Step 4: Download Stanza English model ----
echo ""
echo "[4/4] Downloading Stanza English model (tokenize, pos, lemma, depparse)..."

uv run python -c "
import stanza
stanza.download('en', processors='tokenize,pos,lemma,depparse', logging_level='WARN')
print('  ✓ Stanza English model downloaded')
"

# ---- Done ----
echo ""
echo "============================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Download data:   uv run python scripts/download_wiki.py"
echo "    2. Parse syntax:    uv run python -m src.processing.syntax_parser \\"
echo "                          --input data/wiki1m_for_simcse.txt \\"
echo "                          --output data/parsed_graphs/"
echo "    3. Train:           uv run python src/train.py experiment=multi_loss"
echo "============================================="
