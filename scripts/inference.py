"""
inference.py - Standalone Inference for Banking Intent Classification
=====================================================================
Required interface (per assignment spec):

    class IntentClassification:
        def __init__(self, model_path):  # load config, tokenizer, model
        def __call__(self, message):     # return predicted_label
"""

import sys
import yaml
from unsloth import FastLanguageModel


class IntentClassification:
    """
    Banking Intent Classifier using a fine-tuned LLM (Unsloth / LoRA).

    Usage::

        classifier = IntentClassification("configs/inference.yaml")
        label = classifier("I am still waiting on my card?")
        print(label)   # → "card_arrival"
    """

    # Prompt template — MUST match the one used during training
    _PROMPT = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    _INSTRUCTION = (
        "You are a banking intent classifier. "
        "Given a customer's message, classify it into one of the banking "
        "intent categories. Only output the intent label, nothing else."
    )

    # ------------------------------------------------------------------
    def __init__(self, model_path: str):
        """
        Load the configuration file, tokenizer, and model checkpoint.

        Parameters
        ----------
        model_path : str
            Path to a YAML configuration file that contains at least the
            path to the saved model checkpoint.
        """
        # ---- config ----
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ---- model + tokenizer ----
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_path"],
            max_seq_length=self.config["max_seq_length"],
            load_in_4bit=self.config["load_in_4bit"],
        )
        FastLanguageModel.for_inference(self.model)  # 2× faster

    # ------------------------------------------------------------------
    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single customer message.

        Parameters
        ----------
        message : str
            The customer's banking query.

        Returns
        -------
        predicted_label : str
            One of the 77 BANKING77 intent labels.
        """
        prompt = self._PROMPT.format(
            instruction=self._INSTRUCTION,
            input=message.strip(),
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.get("max_new_tokens", 32),
            do_sample=False,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_label = (
            decoded.split("### Response:")[-1].strip().split("\n")[0].strip()
        )
        return predicted_label


# =====================================================================
# CLI Demo
# =====================================================================
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/inference.yaml"

    print("🔄 Loading model ...")
    classifier = IntentClassification(config_path)
    print("✅ Model loaded!\n")

    # ---- Example messages ----
    examples = [
        "I am still waiting on my card?",
        "How do I change my PIN number?",
        "I want to cancel my transfer",
        "Why was I charged extra for my card payment?",
        "How do I top up my account using a bank transfer?",
        "I lost my card, what should I do?",
        "Is my card compatible with Apple Pay?",
        "My transfer hasn't arrived yet",
    ]

    print("=" * 60)
    print("  Banking Intent Classification — Inference Demo")
    print("=" * 60)
    for msg in examples:
        label = classifier(msg)
        print(f"\n  📩 \"{msg}\"")
        print(f"  🏷️  → {label}")
    print("\n" + "=" * 60)

    # ---- Interactive mode ----
    print("\n💬 Interactive mode — type 'quit' to exit\n")
    while True:
        try:
            user_input = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        label = classifier(user_input)
        print(f"  🏷️  → {label}\n")
