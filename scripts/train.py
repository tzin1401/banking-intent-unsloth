"""
train.py - Fine-tune LLM for Banking Intent Classification with Unsloth
========================================================================
Steps:
  1. Load config (configs/train.yaml)
  2. Load base model + tokenizer (QLoRA 4-bit)
  3. Attach LoRA adapters
  4. Prepare dataset with Alpaca prompt template
  5. Train with SFTTrainer
  6. Save checkpoint (LoRA adapter)
  7. Evaluate on test set → accuracy + classification report
"""

import os
import sys
import yaml
import pandas as pd
from difflib import get_close_matches
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from sklearn.metrics import accuracy_score, classification_report

# Auto-detect trl version: SFTConfig (>= 0.8) vs TrainingArguments (< 0.8)
try:
    from trl import SFTConfig
    USE_SFT_CONFIG = True
except ImportError:
    from transformers import TrainingArguments
    USE_SFT_CONFIG = False

# ============================================================
# 1. Load config
# ============================================================
CONFIG_PATH = os.environ.get("TRAIN_CONFIG", "configs/train.yaml")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

print("=" * 55)
print("  Banking Intent — Training")
print("=" * 55)
print("\n📋 Hyperparameters:")
for k, v in config.items():
    print(f"   {k}: {v}")
print()

# ============================================================
# 2. Load model + tokenizer
# ============================================================
print("📦 Loading model:", config["model_name"])
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config["model_name"],
    max_seq_length=config["max_seq_length"],
    dtype=config.get("dtype"),
    load_in_4bit=config["load_in_4bit"],
)

# ============================================================
# 3. Attach LoRA adapters
# ============================================================
print("🔧 Adding LoRA adapters (r={}, alpha={}) ...".format(
    config["lora_r"], config["lora_alpha"]
))
model = FastLanguageModel.get_peft_model(
    model,
    r=config["lora_r"],
    target_modules=config["target_modules"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",   # saves ~30% VRAM
    random_state=config["seed"],
)

# ============================================================
# 4. Prepare dataset
# ============================================================
train_df = pd.read_csv(config["train_data_path"])
test_df  = pd.read_csv(config["test_data_path"])

# Load valid label list for prediction validation
labels_path = os.path.join(os.path.dirname(config["train_data_path"]), "labels.txt")
if os.path.exists(labels_path):
    with open(labels_path, "r") as f:
        VALID_LABELS = [line.strip() for line in f if line.strip()]
    print(f"   Valid labels: {len(VALID_LABELS)} loaded from {labels_path}")
else:
    # Fallback: extract from training data
    VALID_LABELS = sorted(train_df["output"].unique().tolist())
    print(f"   Valid labels: {len(VALID_LABELS)} (extracted from training data)")


def clean_prediction(raw_pred: str, valid_labels: list = VALID_LABELS) -> str:
    """Map raw model prediction to the closest valid label."""
    raw_pred = raw_pred.strip()
    # Exact match (case-insensitive)
    for label in valid_labels:
        if raw_pred.lower() == label.lower():
            return label
    # Fuzzy match
    lower_labels = [l.lower() for l in valid_labels]
    matches = get_close_matches(raw_pred.lower(), lower_labels, n=1, cutoff=0.6)
    if matches:
        idx = lower_labels.index(matches[0])
        return valid_labels[idx]
    return raw_pred  # fallback — will count as wrong

# Alpaca-style prompt template --------------------------------
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token


def formatting_func(examples):
    """Format each row into the full prompt string."""
    texts = []
    for instr, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(ALPACA_PROMPT.format(instr, inp, out) + EOS_TOKEN)
    return {"text": texts}


train_dataset = Dataset.from_pandas(train_df[["instruction", "input", "output"]])
train_dataset = train_dataset.map(formatting_func, batched=True)

print(f"\n📊 Train samples : {len(train_dataset)}")
print(f"   Test samples  : {len(test_df)}")
print(f"\n📝 Sample prompt (first 300 chars):\n{train_dataset[0]['text'][:300]}...\n")

# ============================================================
# 5. Trainer
# ============================================================
training_kwargs = dict(
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    num_train_epochs=config["num_train_epochs"],
    learning_rate=config["learning_rate"],
    optim=config["optimizer"],
    weight_decay=config["weight_decay"],
    lr_scheduler_type=config["lr_scheduler_type"],
    warmup_steps=config["warmup_steps"],
    max_grad_norm=config["max_grad_norm"],
    logging_steps=config["logging_steps"],
    output_dir=config["output_dir"],
    save_strategy=config["save_strategy"],
    seed=config["seed"],
    fp16=config.get("fp16", True),
    bf16=config.get("bf16", False),
    report_to="none",
)

if USE_SFT_CONFIG:
    training_kwargs["dataset_text_field"] = "text"
    training_kwargs["max_seq_length"] = config["max_seq_length"]
    training_kwargs["packing"] = False
    training_kwargs["dataset_num_proc"] = config.get("dataset_num_proc", 2)
    args = SFTConfig(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        args=args,
    )
else:
    args = TrainingArguments(**training_kwargs)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=config["max_seq_length"],
        dataset_num_proc=config.get("dataset_num_proc", 2),
        packing=False,
        args=args,
    )

# ============================================================
# 6. Train!
# ============================================================
print("🚀 Starting training ...\n")
stats = trainer.train()

print(f"\n✅ Training finished!")
print(f"   Runtime     : {stats.metrics['train_runtime']:.1f}s")
print(f"   Final loss  : {stats.metrics['train_loss']:.4f}")

# ============================================================
# 7. Save checkpoint
# ============================================================
os.makedirs(config["output_dir"], exist_ok=True)
model.save_pretrained(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print(f"\n💾 Model saved → {config['output_dir']}/")

# ============================================================
# 8. Evaluate on test set
# ============================================================
print("\n📊 Evaluating on test set ...")
FastLanguageModel.for_inference(model)

y_true, y_pred = [], []

for idx, row in test_df.iterrows():
    prompt = ALPACA_PROMPT.format(row["instruction"], row["input"], "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.get("eval_max_new_tokens", 32),
        temperature=config.get("eval_temperature", 0.0),
        do_sample=config.get("eval_do_sample", False),
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    raw_predicted = response.split("### Response:")[-1].strip().split("\n")[0].strip()
    predicted = clean_prediction(raw_predicted)

    y_true.append(row["output"])
    y_pred.append(predicted)

    if (idx + 1) % 200 == 0:
        acc_so_far = accuracy_score(y_true, y_pred)
        print(f"   [{idx+1}/{len(test_df)}]  running acc = {acc_so_far*100:.1f}%")

# --- Results ---
accuracy = accuracy_score(y_true, y_pred)
report   = classification_report(y_true, y_pred, zero_division=0)

print(f"\n{'=' * 55}")
print(f"  🎯 Test Accuracy: {accuracy * 100:.2f}%")
print(f"{'=' * 55}")
print(f"\n{report}")

# Save to file
results_path = os.path.join(config["output_dir"], "eval_results.txt")
with open(results_path, "w") as f:
    f.write(f"Test Accuracy: {accuracy * 100:.2f}%\n\n")
    f.write(f"Classification Report:\n{report}\n")
print(f"📄 Results saved → {results_path}")
