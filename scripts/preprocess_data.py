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

# ============================================================
# 1. BANKING77 label names (77 intents, index 0-76)
# ============================================================
LABEL_NAMES = [
    "activate_my_card",
    "age_limit",
    "apple_pay_or_google_pay",
    "atm_support",
    "automatic_top_up",
    "balance_not_updated_after_bank_transfer",
    "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed",
    "cancel_transfer",
    "card_about_to_expire",
    "card_acceptance",
    "card_arrival",
    "card_delivery_estimate",
    "card_linking",
    "card_not_working",
    "card_payment_fee_charged",
    "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate",
    "card_swallowed",
    "cash_withdrawal_charge",
    "cash_withdrawal_not_recognised",
    "change_pin",
    "compromised_card",
    "contactless_not_working",
    "country_support",
    "declined_card_payment",
    "declined_cash_withdrawal",
    "declined_transfer",
    "direct_debit_payment_not_recognised",
    "disposable_card_limits",
    "edit_personal_details",
    "exchange_charge",
    "exchange_rate",
    "exchange_via_app",
    "extra_charge_on_statement",
    "failed_transfer",
    "fiat_currency_support",
    "freeze_card",
    "getting_spare_card",
    "getting_virtual_card",
    "get_disposable_virtual_card",
    "get_physical_card",
    "lost_or_stolen_card",
    "lost_or_stolen_phone",
    "order_physical_card",
    "passcode_forgotten",
    "pending_card_payment",
    "pending_cash_withdrawal",
    "pending_top_up",
    "pending_transfer",
    "pin_blocked",
    "receiving_money",
    "Refund_not_showing_up",
    "request_refund",
    "reverted_card_payment?",
    "supported_cards_and_currencies",
    "terminate_account",
    "top_up_by_bank_transfer_charge",
    "top_up_by_card_charge",
    "top_up_by_cash_or_cheque",
    "top_up_failed",
    "top_up_limits",
    "top_up_reverted",
    "topping_up_by_card",
    "transaction_charged_twice",
    "transfer_fee_charged",
    "transfer_into_account",
    "transfer_not_received_by_recipient",
    "transfer_timing",
    "unable_to_verify_identity",
    "verify_my_identity",
    "verify_source_of_funds",
    "verify_top_up",
    "virtual_card_not_working",
    "visa_or_mastercard",
    "why_verify_identity",
    "wrong_amount_of_cash_received",
    "wrong_exchange_rate_for_cash_withdrawal",
]

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


def build_dataframe(split) -> pd.DataFrame:
    """Convert a HuggingFace dataset split into a DataFrame."""
    df = pd.DataFrame({
        "text": [preprocess_text(row["text"]) for row in split],
        "label": [row["label"] for row in split],
    })
    df["intent"] = df["label"].map(lambda idx: LABEL_NAMES[idx])
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
    train_full = build_dataframe(dataset["train"])
    test_full  = build_dataframe(dataset["test"])
    print(f"   Full train : {len(train_full):>6,} rows  ({train_full['label'].nunique()} classes)")
    print(f"   Full test  : {len(test_full):>6,} rows  ({test_full['label'].nunique()} classes)")

    # --- Sample ------------------------------------------------
    TRAIN_PER_CLASS = 50   # ~3,850 samples
    TEST_PER_CLASS  = 20   # ~1,540 samples

    train_sampled = sample_balanced(train_full, TRAIN_PER_CLASS)
    test_sampled  = sample_balanced(test_full,  TEST_PER_CLASS)
    print(f"\n✂️  Sampled train : {len(train_sampled):>6,} rows  ({TRAIN_PER_CLASS}/class)")
    print(f"   Sampled test  : {len(test_sampled):>6,} rows  ({TEST_PER_CLASS}/class)")

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
