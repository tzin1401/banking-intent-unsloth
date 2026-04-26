#!/bin/bash
set -euo pipefail

echo "========================================"
echo "  Kaggle Run: Train + Package Artifacts"
echo "========================================"

echo ""
echo "[1/3] Preprocessing BANKING77 data ..."
python scripts/preprocess_data.py

echo ""
echo "[2/3] Fine-tuning model ..."
python scripts/train.py

echo ""
echo "[3/3] Packaging outputs into artifacts/ ..."
python scripts/package_artifacts.py

echo ""
echo "✅ Done. Check artifacts/LATEST.txt for newest run."
