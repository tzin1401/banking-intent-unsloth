#!/bin/bash
set -e

echo "========================================"
echo "  Banking Intent — Inference"
echo "========================================"

CONFIG="${1:-configs/inference.yaml}"
python scripts/inference.py "$CONFIG"
