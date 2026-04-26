"""
Banking Intent Classification — Google Colab Training Notebook
================================================================
Copy-paste this entire file into a Google Colab notebook cell-by-cell.
Or upload it and run as a .py script.

Prerequisites: Enable GPU runtime in Colab
  Runtime → Change runtime type → T4 GPU
"""

# ============================================================
# CELL 1: Install dependencies
# ============================================================
# !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# !pip install --no-deps xformers trl peft accelerate bitsandbytes
# !pip install datasets pandas scikit-learn pyyaml

# ============================================================
# CELL 2: Imports
# ============================================================
import os
import pandas as pd
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.metrics import accuracy_score, classification_report

# ============================================================
# CELL 3: Configuration (all hyperparameters in one place)
# ============================================================
CONFIG = {
    # Model
    "model_name": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "max_seq_length": 512,
    "load_in_4bit": True,

    # LoRA
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "optimizer": "adamw_8bit",
    "weight_decay": 0.01,
    "lr_scheduler_type": "linear",
    "warmup_steps": 10,
    "max_grad_norm": 1.0,
    "logging_steps": 10,
    "seed": 42,

    # Save
    "output_dir": "./outputs",

    # Sampling
    "train_per_class": 50,
    "test_per_class": 20,
}

print("📋 Configuration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# ============================================================
# CELL 4: Load & preprocess BANKING77
# ============================================================
LABEL_NAMES = [
    "activate_my_card", "age_limit", "apple_pay_or_google_pay",
    "atm_support", "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "cancel_transfer", "card_about_to_expire",
    "card_acceptance", "card_arrival", "card_delivery_estimate",
    "card_linking", "card_not_working", "card_payment_fee_charged",
    "card_payment_not_recognised", "card_payment_wrong_exchange_rate",
    "card_swallowed", "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised", "change_pin", "compromised_card",
    "contactless_not_working", "country_support", "declined_card_payment",
    "declined_cash_withdrawal", "declined_transfer",
    "direct_debit_payment_not_recognised", "disposable_card_limits",
    "edit_personal_details", "exchange_charge", "exchange_rate",
    "exchange_via_app", "extra_charge_on_statement", "failed_transfer",
    "fiat_currency_support", "freeze_card", "getting_spare_card",
    "getting_virtual_card", "get_disposable_virtual_card",
    "get_physical_card", "lost_or_stolen_card", "lost_or_stolen_phone",
    "order_physical_card", "passcode_forgotten", "pending_card_payment",
    "pending_cash_withdrawal", "pending_top_up", "pending_transfer",
    "pin_blocked", "receiving_money", "Refund_not_showing_up",
    "request_refund", "reverted_card_payment?",
    "supported_cards_and_currencies", "terminate_account",
    "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_by_cash_or_cheque", "top_up_failed", "top_up_limits",
    "top_up_reverted", "topping_up_by_card", "transaction_charged_twice",
    "transfer_fee_charged", "transfer_into_account",
    "transfer_not_received_by_recipient", "transfer_timing",
    "unable_to_verify_identity", "verify_my_identity",
    "verify_source_of_funds", "verify_top_up",
    "virtual_card_not_working", "visa_or_mastercard",
    "why_verify_identity", "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

INSTRUCTION = (
    "You are a banking intent classifier. "
    "Given a customer's message, classify it into one of the banking "
    "intent categories. Only output the intent label, nothing else."
)

print("\n📥 Loading BANKING77 ...")
raw = load_dataset("PolyAI/banking77")

def build_df(split):
    return pd.DataFrame({
        "text": [r["text"].strip().lower() for r in split],
        "label": [r["label"] for r in split],
        "intent": [LABEL_NAMES[r["label"]] for r in split],
    })

train_full = build_df(raw["train"])
test_full  = build_df(raw["test"])

# Balanced sampling
train_df = (
    train_full.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=min(CONFIG["train_per_class"], len(g)), random_state=42))
    .reset_index(drop=True)
)
test_df = (
    test_full.groupby("label", group_keys=False)
    .apply(lambda g: g.sample(n=min(CONFIG["test_per_class"], len(g)), random_state=42))
    .reset_index(drop=True)
)

# Add SFT columns
train_df["instruction"] = INSTRUCTION
train_df["input"]       = train_df["text"]
train_df["output"]      = train_df["intent"]
test_df["instruction"]  = INSTRUCTION
test_df["input"]        = test_df["text"]
test_df["output"]       = test_df["intent"]

print(f"   Train: {len(train_df)} samples  |  Test: {len(test_df)} samples")

# Save CSVs
os.makedirs("sample_data", exist_ok=True)
train_df.to_csv("sample_data/train.csv", index=False)
test_df.to_csv("sample_data/test.csv", index=False)
print("   ✅ Saved sample_data/train.csv & test.csv")

# ============================================================
# CELL 5: Load model
# ============================================================
print("\n📦 Loading model ...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=CONFIG["model_name"],
    max_seq_length=CONFIG["max_seq_length"],
    dtype=None,
    load_in_4bit=CONFIG["load_in_4bit"],
)

# ============================================================
# CELL 6: Add LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=CONFIG["lora_r"],
    target_modules=CONFIG["target_modules"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=CONFIG["seed"],
)
print("🔧 LoRA adapters attached")

# ============================================================
# CELL 7: Format dataset
# ============================================================
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_func(examples):
    texts = []
    for instr, inp, out in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        texts.append(ALPACA_PROMPT.format(instr, inp, out) + EOS_TOKEN)
    return {"text": texts}

train_dataset = Dataset.from_pandas(train_df[["instruction", "input", "output"]])
train_dataset = train_dataset.map(formatting_func, batched=True)

print(f"📊 {len(train_dataset)} training samples formatted")
print(f"\n📝 Example:\n{train_dataset[0]['text'][:400]}...\n")

# ============================================================
# CELL 8: Train
# ============================================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=CONFIG["max_seq_length"],
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["num_train_epochs"],
        learning_rate=CONFIG["learning_rate"],
        optim=CONFIG["optimizer"],
        weight_decay=CONFIG["weight_decay"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_steps=CONFIG["warmup_steps"],
        max_grad_norm=CONFIG["max_grad_norm"],
        logging_steps=CONFIG["logging_steps"],
        output_dir=CONFIG["output_dir"],
        save_strategy="epoch",
        seed=CONFIG["seed"],
        fp16=True,
        report_to="none",
    ),
)

print("🚀 Training started ...\n")
stats = trainer.train()
print(f"\n✅ Done!  Loss: {stats.metrics['train_loss']:.4f}  "
      f"Time: {stats.metrics['train_runtime']:.0f}s")

# ============================================================
# CELL 9: Save checkpoint
# ============================================================
model.save_pretrained(CONFIG["output_dir"])
tokenizer.save_pretrained(CONFIG["output_dir"])
print(f"💾 Saved to {CONFIG['output_dir']}/")

# ============================================================
# CELL 10: Evaluate
# ============================================================
print("\n📊 Evaluating on test set ...")
FastLanguageModel.for_inference(model)

y_true, y_pred = [], []

for idx, row in test_df.iterrows():
    prompt = ALPACA_PROMPT.format(row["instruction"], row["input"], "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs, max_new_tokens=32, temperature=0.0, do_sample=False,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predicted = response.split("### Response:")[-1].strip().split("\n")[0].strip()

    y_true.append(row["output"])
    y_pred.append(predicted)

    if (idx + 1) % 200 == 0:
        print(f"   [{idx+1}/{len(test_df)}]  acc = {accuracy_score(y_true, y_pred)*100:.1f}%")

accuracy = accuracy_score(y_true, y_pred)
print(f"\n{'='*50}")
print(f"  🎯 TEST ACCURACY: {accuracy*100:.2f}%")
print(f"{'='*50}\n")
print(classification_report(y_true, y_pred, zero_division=0))

# ============================================================
# CELL 11: Test inference interactively
# ============================================================
print("\n🧪 Quick inference test:")
test_msgs = [
    "I am still waiting on my card?",
    "I want to change my pin",
    "Cancel my transfer please",
    "I lost my card",
]
for msg in test_msgs:
    prompt = ALPACA_PROMPT.format(INSTRUCTION, msg, "")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=32, temperature=0.0, do_sample=False)
    pred = tokenizer.decode(out[0], skip_special_tokens=True)
    pred = pred.split("### Response:")[-1].strip().split("\n")[0].strip()
    print(f"  \"{msg}\"  →  {pred}")

# ============================================================
# CELL 12: Download checkpoint
# ============================================================
# Uncomment to zip and download from Colab:
# !zip -r outputs.zip outputs/
# from google.colab import files
# files.download("outputs.zip")

print("\n✅ All done! Download the outputs/ folder for local inference.")
