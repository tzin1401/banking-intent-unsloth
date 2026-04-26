#!/bin/bash
set -e

echo "========================================"
echo "  Banking Intent — Training Pipeline"
echo "========================================"

# Step 1: Install dependencies
echo ""
echo "[1/3] Installing dependencies ..."
pip install -q -r requirements.txt

# Step 2: Preprocess data
echo ""
echo "[2/3] Preprocessing BANKING77 data ..."
python scripts/preprocess_data.py

# Step 3: Train
echo ""
echo "[3/3] Fine-tuning model ..."
python scripts/train.py

echo ""
echo "========================================"
echo "  ✅ Training pipeline complete!"
echo "========================================"
