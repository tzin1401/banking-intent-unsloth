#!/bin/bash
set -euo pipefail

echo "========================================"
echo "  Kaggle Run: Train + Package Artifacts"
echo "========================================"

echo ""
echo "[1/4] Installing dependencies ..."
pip install -q -r requirements.txt

echo ""
echo "[2/4] Preprocessing BANKING77 data ..."
python scripts/preprocess_data.py

echo ""
echo "[3/4] Fine-tuning model ..."
python scripts/train.py

echo ""
echo "[4/4] Packaging outputs into artifacts/ ..."
python scripts/package_artifacts.py

echo ""
echo "✅ Done. Check artifacts/LATEST.txt for newest run."
