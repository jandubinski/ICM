# src/baselines.py
from typing import List
from tqdm import tqdm
from .prompts import format_query, format_icl_examples, extract_true_false

def run_zero_shot(model: str, dataset: List[dict], client) -> float:
    """
    Zero-shot TruthfulQA baseline using unified Judgment formatting.
    Base → .complete(), Chat → .chat()
    """
    is_chat = "instruct" in model.lower()
    correct = 0
    n = len(dataset)

    for ex in tqdm(dataset, desc=f"Zero-shot ({model})"):
        prompt = format_query(ex["question"], ex["choice"])
        gold = "True" if ex["label"] == 1 else "False"

        if is_chat:
            content = client.chat(
                model=model,
                messages=[{"role": "system", "content": "You are a truthful assistant."},
                          {"role": "user", "content": prompt}],
                max_tokens=4,
                temperature=0.0,
            )
        else:
            content = client.complete(
                model=model,
                prompt=prompt,
                max_tokens=4,
                temperature=0.0,
            )

        pred = extract_true_false(content)
        if pred == gold:
            correct += 1

    return correct / n


def run_golden_icl(model: str, train: List[dict], test: List[dict], client) -> float:
    """
    Golden-label many-shot ICL, full-train, unified Judgment formatting.
    Base → .complete(), Chat → .chat()
    """
    is_chat = "instruct" in model.lower()
    correct = 0
    n = len(test)

    # Build all context examples with gold labels
    examples = [
        {
            "question": ex["question"],
            "choice": ex["choice"],
            "label": "True" if ex["label"] == 1 else "False",
        }
        for ex in train
    ]
    context = format_icl_examples(examples)

    for ex in tqdm(test, desc=f"Golden-ICL ({model})"):
        query = format_query(ex["question"], ex["choice"])
        gold = "True" if ex["label"] == 1 else "False"

        if is_chat:
            content = client.chat(
                model=model,
                messages=[{"role": "system", "content": "You are a truthful assistant."},
                          {"role": "user", "content": context + "\n" + query}],
                max_tokens=4,
                temperature=0.0,
            )
        else:
            content = client.complete(
                model=model,
                prompt=context + "\n" + query,
                max_tokens=4,
                temperature=0.0,
            )

        pred = extract_true_false(content)
        if pred == gold:
            correct += 1

    return correct / n
