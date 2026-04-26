# Banking Intent Classification with Unsloth

Fine-tuning **Llama-3.2-1B-Instruct** (QLoRA 4-bit) on the **BANKING77** dataset for banking customer intent classification using **Unsloth**.

---

## 📁 Project Structure

```
banking-intent-unsloth/
├── scripts/
│   ├── preprocess_data.py    # Data preprocessing & sampling
│   ├── train.py              # Fine-tuning with Unsloth
│   └── inference.py          # Standalone inference (IntentClassification class)
├── configs/
│   ├── train.yaml            # Training hyperparameters
│   └── inference.yaml        # Inference settings
├── sample_data/
│   ├── train.csv             # Sampled & preprocessed training data
│   └── test.csv              # Sampled & preprocessed test data
├── train.sh                  # One-click training script
├── inference.sh              # One-click inference script
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Setup

### Requirements
- Python 3.10+
- CUDA-compatible GPU (T4 16GB recommended for training)
- ~4GB VRAM minimum for inference (RTX 3050+)

### Installation
```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

| | Source | Samples | Classes |
|---|---|---|---|
| **Full** | [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) | 13,083 | 77 intents |
| **Sampled Train** | Balanced subset | ~3,850 | 77 (50/class) |
| **Sampled Test** | Balanced subset | ~1,540 | 77 (20/class) |

**Preprocessing steps:**
1. Text normalization (lowercase, strip whitespace)
2. Label mapping (integer → intent name)
3. Balanced sampling per intent class
4. Format to Alpaca-style prompts (instruction / input / output)

---

## 🧬 Training

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Base Model** | `Llama-3.2-1B-Instruct` | 1B parameter LLM |
| **Quantization** | QLoRA 4-bit | Saves ~75% VRAM |
| **LoRA rank (r)** | 16 | Low-rank adaptation matrices |
| **LoRA alpha** | 16 | Scaling factor |
| **Batch size** | 8 | Per-device |
| **Gradient accumulation** | 4 | Effective batch = 32 |
| **Epochs** | 3 | Training iterations |
| **Learning rate** | 2e-4 | With linear scheduler |
| **Optimizer** | AdamW 8-bit | Memory-efficient |
| **Weight decay** | 0.01 | Regularization |
| **Max sequence length** | 512 | Token limit per sample |
| **Warmup steps** | 10 | LR warmup |

### Run Training

```bash
# Full pipeline (preprocess + train + evaluate)
bash train.sh

# Or step by step:
python scripts/preprocess_data.py
python scripts/train.py
```

> **Note:** Training is recommended on **Google Colab** (free T4 GPU) or **Kaggle**. Upload this project folder and run `train.sh`.

---

## 🖥️ Inference

### Python API
```python
from scripts.inference import IntentClassification

classifier = IntentClassification("configs/inference.yaml")
label = classifier("I am still waiting on my card?")
print(label)  # → "card_arrival"
```

### CLI
```bash
bash inference.sh
# or
python scripts/inference.py configs/inference.yaml
```

---

## 📈 Results

- **Test Accuracy:** _XX.XX%_ *(update after training)*

---

## 🎬 Video Demo

[🔗 Demo Video (Google Drive)](YOUR_GOOGLE_DRIVE_LINK_HERE)

The video demonstrates:
1. Running the inference script
2. Example input messages & predicted intents
3. Final accuracy on the test set
