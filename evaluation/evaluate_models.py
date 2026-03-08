import json
from pathlib import Path

from config import MODEL_DIR
from evaluation.test_model import evaluate_xgb, evaluate_seq, evaluate_distilled


def main():
    results = {
        "xgb": evaluate_xgb("Validation"),
        "lstm": evaluate_seq("lstm", split="Validation"),
        "bilstm": evaluate_seq("bilstm", split="Validation"),
        "tcn": evaluate_seq("tcn", split="Validation"),
        "tcn_distilled": evaluate_distilled(split="Validation"),
    }

    best_model, best_metrics = max(results.items(), key=lambda kv: kv[1]["accuracy"])
    summary = {"best_model": best_model, "best_accuracy": best_metrics["accuracy"], "results": results}

    out_path = MODEL_DIR / "model_comparison.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Best model: {best_model} (acc={best_metrics['accuracy']:.4f})")
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()
