"""
Package training outputs into a repo-friendly artifacts folder.

This script is designed for Kaggle-first workflow:
1) Train in repo clone on Kaggle
2) Package `outputs/`, `sample_data/`, `configs/` into `artifacts/run_*`
3) Optionally commit/push artifacts back to GitHub
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required path: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def extract_accuracy(eval_file: Path) -> str | None:
    if not eval_file.exists():
        return None
    content = eval_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Test Accuracy:\s*([0-9]+(?:\.[0-9]+)?)%", content)
    return match.group(1) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".", help="Project root path")
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional run name. Default: timestamp (UTC)",
    )
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()
    outputs_dir = root / "outputs"
    sample_data_dir = root / "sample_data"
    configs_dir = root / "configs"

    run_name = args.run_name.strip() or datetime.now(timezone.utc).strftime(
        "run_%Y%m%d_%H%M%S_utc"
    )
    artifact_dir = root / "artifacts" / run_name
    artifact_dir.mkdir(parents=True, exist_ok=True)

    copy_tree(outputs_dir, artifact_dir / "outputs")
    copy_tree(sample_data_dir, artifact_dir / "sample_data")
    copy_tree(configs_dir, artifact_dir / "configs")

    eval_file = outputs_dir / "eval_results.txt"
    accuracy = extract_accuracy(eval_file)
    if eval_file.exists():
        shutil.copy2(eval_file, artifact_dir / "eval_results.txt")

    manifest = {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_paths": {
            "outputs": str(outputs_dir.relative_to(root)),
            "sample_data": str(sample_data_dir.relative_to(root)),
            "configs": str(configs_dir.relative_to(root)),
        },
        "test_accuracy_percent": accuracy,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    latest_pointer = root / "artifacts" / "LATEST.txt"
    latest_pointer.write_text(f"{run_name}\n", encoding="utf-8")

    print("Artifact packaged successfully")
    print(f"- Artifact dir: {artifact_dir}")
    if accuracy is not None:
        print(f"- Test Accuracy: {accuracy}%")
    else:
        print("- Test Accuracy: not found in outputs/eval_results.txt")


if __name__ == "__main__":
    main()
