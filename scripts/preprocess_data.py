"""
preprocess_data.py - Tiền xử lý dataset BANKING77
===================================================
- Load dataset từ HuggingFace
- Text normalization (lowercase, strip)
- Sample balanced subset (~50 samples/intent train, ~20 test)
- Format thành Alpaca-style (instruction / input / output)
- Lưu CSV ra sample_data/
"""

import os
import pandas as pd
from datasets import load_dataset

INSTRUCTION = (
    "You are a banking intent classifier. "
    "Given a customer's message, classify it into one of the banking intent categories. "
    "Only output the intent label, nothing else."
)


# ============================================================
# 2. Helpers
# ============================================================
def preprocess_text(text: str) -> str:
    """Basic text normalization."""
    return text.strip().lower()


def build_dataframe(split, label_names) -> pd.DataFrame:
    """Convert a HuggingFace dataset split into a DataFrame."""
    df = pd.DataFrame({
        "text": [preprocess_text(row["text"]) for row in split],
        "label": [row["label"] for row in split],
    })
    df["intent"] = df["label"].map(lambda idx: label_names[idx])
    return df


def sample_balanced(df: pd.DataFrame, n_per_class: int, seed: int = 42) -> pd.DataFrame:
    """Sample at most n_per_class rows per intent, keeping balance."""
    return (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=seed))
          .reset_index(drop=True)
    )


def add_sft_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add instruction / input / output columns for Alpaca-style SFT."""
    df = df.copy()
    df["instruction"] = INSTRUCTION
    df["input"] = df["text"]
    df["output"] = df["intent"]
    return df


# ============================================================
# 3. Main
# ============================================================
def main():
    print("=" * 55)
    print("  BANKING77 — Data Preprocessing")
    print("=" * 55)

    # --- Load --------------------------------------------------
    print("\n📥 Loading BANKING77 from HuggingFace ...")
    dataset = load_dataset("PolyAI/banking77")
    # Use official labels from dataset metadata to avoid manual label mismatch.
    label_names = dataset["train"].features["label"].names
    train_full = build_dataframe(dataset["train"], label_names)
    test_full  = build_dataframe(dataset["test"], label_names)
    print(f"   Full train : {len(train_full):>6,} rows  ({train_full['label'].nunique()} classes)")
    print(f"   Full test  : {len(test_full):>6,} rows  ({test_full['label'].nunique()} classes)")

    # --- Sample ------------------------------------------------
    TRAIN_PER_CLASS = int(os.environ.get("TRAIN_PER_CLASS", "50"))
    TEST_PER_CLASS  = int(os.environ.get("TEST_PER_CLASS", "20"))
    SAMPLE_SEED = int(os.environ.get("SAMPLE_SEED", "42"))

    train_sampled = sample_balanced(train_full, TRAIN_PER_CLASS, SAMPLE_SEED)
    test_sampled  = sample_balanced(test_full,  TEST_PER_CLASS, SAMPLE_SEED)
    print(f"\n✂️  Sampled train : {len(train_sampled):>6,} rows  ({TRAIN_PER_CLASS}/class)")
    print(f"   Sampled test  : {len(test_sampled):>6,} rows  ({TEST_PER_CLASS}/class)")
    print(f"   Sample seed   : {SAMPLE_SEED}")

    # --- Format for SFT ----------------------------------------
    train_sft = add_sft_columns(train_sampled)
    test_sft  = add_sft_columns(test_sampled)

    # --- Save --------------------------------------------------
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.csv")
    test_path  = os.path.join(out_dir, "test.csv")

    train_sft.to_csv(train_path, index=False)
    test_sft.to_csv(test_path,  index=False)

    print(f"\n💾 Saved  {train_path}  ({len(train_sft)} rows)")
    print(f"   Saved  {test_path}   ({len(test_sft)} rows)")

    # --- Preview -----------------------------------------------
    print("\n📝 Sample entry:")
    row = train_sft.iloc[0]
    print(f"   text       : {row['text']}")
    print(f"   intent     : {row['intent']}")
    print(f"   instruction: {row['instruction'][:60]}...")
    print()


if __name__ == "__main__":
    main()
