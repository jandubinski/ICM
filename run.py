# run.py
import os
import json
import math
import argparse
from typing import List, Dict, Any

from src.hyperbolic_client import HyperbolicClient
from src.baselines import run_zero_shot, run_golden_icl
from src.icm import run_icm, load_json_or_jsonl


# ======================
# Default model settings
# ======================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"
CHAT_MODEL = "meta-llama/Meta-Llama-3.1-405B-Instruct"


# ======================
# Data normalization
# ======================
def coerce_truthfulqa_format(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalizes raw JSON examples into:
        {question, choice, label}
    label ∈ {0,1}
    """
    out = []
    for ex in records:
        q = ex.get("question")
        c = ex.get("choice") or ex.get("statement") or ex.get("claim") or ex.get("answer") or ex.get("text")
        lab = ex.get("label")
        if q is None or c is None or lab is None:
            continue
        out.append({"question": str(q), "choice": str(c), "label": 1 if int(lab) != 0 else 0})
    return out


# ======================
# Result formatting
# ======================
def stderr(acc: float, n: int) -> float:
    """Standard error (percentage points)."""
    if n <= 1:
        return 0.0
    return 100.0 * math.sqrt(max(0.0, acc * (1.0 - acc) / n))


def fmt(name: str, acc: float, n: int) -> str:
    correct = int(round(acc * n))
    return f"{name:>18}: {acc*100:5.1f}%  ({correct}/{n}, ±{stderr(acc, n):.1f}%)"


# ======================
# Main
# ======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Directory with train.json + test.json")
    ap.add_argument("--base_model", type=str, default=BASE_MODEL)
    ap.add_argument("--chat_model", type=str, default=CHAT_MODEL)
    ap.add_argument("--init_k", type=int, default=20)
    ap.add_argument("--max_iters", type=int, default=400)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--search_ctx_cap", type=int, default=256)
    ap.add_argument("--eval_ctx_cap", type=int, default=256)
    ap.add_argument("--results_json", type=str, default="results.json")
    args = ap.parse_args()

    # Load data
    train_path = os.path.join(args.data_dir, "truthfulqa_train.json")
    test_path  = os.path.join(args.data_dir, "truthfulqa_test.json")

    train_raw = load_json_or_jsonl(train_path)
    test_raw  = load_json_or_jsonl(test_path)

    train = coerce_truthfulqa_format(train_raw)
    test  = coerce_truthfulqa_format(test_raw)

    if not train or not test:
        raise RuntimeError("Train/test empty or invalid. Must contain: question, choice, label.")

    n_test = len(test)

    client = HyperbolicClient()
    results = {}

    print("\n===== Running TruthfulQA Baselines + ICM =====\n")

    # ----------------------
    # Baseline 1: Zero-Shot (BASE)
    # ----------------------
    zs_base_acc = run_zero_shot(args.base_model, test, client)
    print(fmt("Zero-Shot Base", zs_base_acc, n_test))
    results["zero_shot_base"] = zs_base_acc

    # ----------------------
    # Baseline 2: Zero-Shot (CHAT)
    # ----------------------
    zs_chat_acc = run_zero_shot(args.chat_model, test, client)
    print(fmt("Zero-Shot Chat", zs_chat_acc, n_test))
    results["zero_shot_chat"] = zs_chat_acc

    # ----------------------
    # Baseline 3: Golden ICL (BASE, full-train)
    # ----------------------
    golden_acc = run_golden_icl(args.base_model, train, test, client)
    print(fmt("Golden ICL", golden_acc, n_test))
    results["golden_icl"] = golden_acc

    # ----------------------
    # ICM (Algorithm 1 only)
    # ----------------------
    icm_acc = run_icm(
        train=train,
        test=test,
        client=client,
        init_k=args.init_k,
        max_iters=args.max_iters,
        alpha=args.alpha,
        context_cap=args.search_ctx_cap,
        eval_context_cap=args.eval_ctx_cap,
    )
    print(fmt("ICM", icm_acc, n_test))
    results["icm"] = icm_acc

    # ----------------------
    # Summary
    # ----------------------
    print("\n=========== Summary ===========")
    print(fmt("Zero-Shot Base", zs_base_acc, n_test))
    print(fmt("Zero-Shot Chat", zs_chat_acc, n_test))
    print(fmt("Golden ICL", golden_acc, n_test))
    print(fmt("ICM", icm_acc, n_test))

    # Save results
    try:
        with open(args.results_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results → {args.results_json}")
    except Exception as e:
        print(f"Warning: Could not write results.json: {e}")


if __name__ == "__main__":
    main()
