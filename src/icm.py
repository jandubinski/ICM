# src/icm.py
import json
import random
import math
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from src.hyperbolic_client import HyperbolicClient

# ===============================
# Config / constants
# ===============================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-405B"   # Algorithm 1: base (pretrained) model only

# Tokens we accept as the first output token after "Judgment:"
TRUE_TOKS = {"true", "True", "TRUE"}
FALSE_TOKS = {"false", "False", "FALSE"}

# Instruction and prompt formatting used for both scoring and evaluation
INSTRUCTION = "Decide whether each claim is truthful. Respond with True or False.\n"


# ===============================
# Algorithm 2 placeholder (NOT used)
# ===============================
def consistencyfix(D: Dict[int, str]) -> Dict[int, str]:
    """
    Placeholder for Algorithm 2. DO NOT implement per user instruction.
    """
    return D


# ===============================
# Prompt helpers (ICL-style)
# ===============================
def _render_demo(question: str, choice: str, label_str: str) -> str:
    # label_str should be "True" or "False"
    return (
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        f"Judgment: {label_str}\n"
    )

def _render_query(question: str, choice: str) -> str:
    # Ends at "Judgment:" so we can take the next-token logprobs.
    return (
        f"Question: {question}\n"
        f"Claim: {choice}\n"
        f"Judgment:"
    )

def _build_prediction_prompt(
    rows: List[Dict[str, Any]],
    labeled: Dict[int, str],
    target_idx: int,
    context_cap: Optional[int]
) -> str:
    # Build up to context_cap demonstrations from currently labeled items (excluding target)
    pool = [i for i in labeled.keys() if i != target_idx]
    random.shuffle(pool)
    if context_cap is not None and len(pool) > context_cap:
        pool = pool[:context_cap]

    parts = [INSTRUCTION, ""]
    for i in pool:
        parts.append(_render_demo(rows[i]["question"], rows[i]["choice"], labeled[i]))
    parts.append(_render_query(rows[target_idx]["question"], rows[target_idx]["choice"]))
    return "\n".join(parts)


# ===============================
# Logprob extraction
# ===============================
def _extract_tf_logprobs(top: Dict[str, float]) -> Tuple[float, float]:
    """
    Hyperbolic format for first token top-logprobs:
        top = { " False": -1.80, " True": -3.91, ... }  (keys may have leading spaces)
    Returns (logp_true, logp_false).
    """
    lp_t, lp_f = -1e9, -1e9
    for tok, lp in top.items():
        st = tok.strip()
        if st in TRUE_TOKS:
            lp_t = max(lp_t, float(lp))
        if st in FALSE_TOKS:
            lp_f = max(lp_f, float(lp))
    return lp_t, lp_f


def _score_label_logprob(
    client: HyperbolicClient,
    prompt: str,
    top_logprobs: int = 8,
) -> Tuple[float, float]:
    """
    Returns (logp_true, logp_false) for the first token after 'Judgment:'.
    """
    resp = client.complete(
        model=BASE_MODEL,
        prompt=prompt,
        max_tokens=1,
        temperature=0.0,
        logprobs=True,
        top_logprobs=top_logprobs,
        return_json=True,
    )

    choice = resp["choices"][0]
    logprobs_block = choice.get("logprobs", {})
    # Hyperbolic structure example:
    # logprobs_block["top_logprobs"] = [ { " False": -1.80, " True": -3.9, ... } ]
    top_list = logprobs_block.get("top_logprobs", None)
    if top_list and len(top_list) > 0 and isinstance(top_list[0], dict):
        return _extract_tf_logprobs(top_list[0])

    # Fallback ↦ no signal:
    return -1e9, -1e9


# ===============================
# Utility: evaluation parsing
# ===============================
def _parse_true_false(out: str) -> str:
    """
    Parse leading token for True/False with a conservative fallback.
    """
    if not out:
        return "False"
    s = out.strip()
    low = s.lower()
    if low.startswith("true"):
        return "True"
    if low.startswith("false"):
        return "False"
    # general scan
    tpos = low.find("true")
    fpos = low.find("false")
    if tpos == -1 and fpos == -1:
        return "False"
    if tpos == -1:
        return "False"
    if fpos == -1:
        return "True"
    return "True" if tpos < fpos else "False"


# ===============================
# Algorithm 1 (pure): Pθ only, no I(D), no Alg. 2
# Uniform random sampling each iteration
# Local Δ update of the average logprob
# ===============================
def icm_search(
    train: List[Dict[str, Any]],
    client: HyperbolicClient,
    init_k: int = 20,
    max_iters: int = 400,
    alpha: float = 100.0,         # scale on Pθ(D); only term used
    T0: float = 3.0,
    Tmin: float = 1e-3,
    cooling_beta: float = 1.0,    # annealing β
    context_cap: Optional[int] = None,
    top_logprobs: int = 8,
) -> Dict[int, str]:

    n = len(train)
    idxs = list(range(n))
    random.shuffle(idxs)

    # Step 1: Initialize by labeling K random examples with random labels
    labeled: Dict[int, str] = {i: random.choice(["True", "False"]) for i in idxs[:init_k]}

    # Step 2: D ← consistencyfix(D) (no-op)
    labeled = consistencyfix(labeled)

    # Helper: per-item contribution (logprob of current label under context of other labels)
    def item_logp(i: int, lab_str: str) -> float:
        prompt = _build_prediction_prompt(train, labeled, i, context_cap)
        lp_true, lp_false = _score_label_logprob(client, prompt, top_logprobs=top_logprobs)
        return lp_true if lab_str == "True" else lp_false

    # Initialize Pθ as average logprob over currently labeled items
    if labeled:
        lp_sum = 0.0
        for i in labeled.keys():
            lp_sum += item_logp(i, labeled[i])
        P_curr = lp_sum / max(1, len(labeled))
        U_curr = alpha * P_curr
    else:
        P_curr = 0.0
        U_curr = 0.0

    # Main loop (uniform random sampling)
    for it in tqdm(range(1, max_iters + 1), desc="ICM Search"):
        # Step 4: Temperature update
        T = max(Tmin, T0 / (1.0 + cooling_beta * math.log(it + 1)))

        # Step 5: Sample example uniformly at random (can be labeled or unlabeled)
        j = random.choice(idxs)

        # Step 6: Assign label via argmax Pθ(y | x_j, D \ (x_j, y_j))
        prompt_j = _build_prediction_prompt(train, labeled, j, context_cap)
        lp_true, lp_false = _score_label_logprob(client, prompt_j, top_logprobs=top_logprobs)
        y_hat = "True" if lp_true >= lp_false else "False"

        # Tentative update D̂
        old_lab_present = j in labeled
        old_lab = labeled.get(j, None)

        # Old local contribution
        if old_lab_present:
            lp_old = item_logp(j, old_lab)
        else:
            lp_old = 0.0

        # Apply proposed label
        D_hat = dict(labeled)
        D_hat[j] = y_hat
        D_hat = consistencyfix(D_hat)  # still a no-op

        # New local contribution
        # For local scoring, temporarily switch 'labeled' to use D_hat context when computing lp_new
        # (We need the prompt conditioned on D_hat \ {(x_j, y_j)}; since we compute for j, this is fine.)
        labeled_backup = labeled
        labeled = D_hat
        lp_new = item_logp(j, D_hat[j])
        labeled = labeled_backup

        # Pθ update (average)
        if old_lab_present:
            denom = max(1, len(labeled))
            P_new = P_curr + (lp_new - lp_old) / denom
        else:
            denom_old = max(1, len(labeled))
            denom_new = denom_old + 1
            P_new = (P_curr * denom_old + lp_new) / denom_new

        U_new = alpha * P_new
        delta = U_new - U_curr

        # Accept / reject
        if delta > 0 or random.random() < math.exp(delta / max(T, 1e-8)):
            labeled = D_hat
            P_curr = P_new
            U_curr = U_new

    return labeled


# ===============================
# Many-shot evaluation on TEST using discovered labels
# ===============================
def _pack_manyshot_context(train: List[Dict[str, Any]], labels: Dict[int, str], max_examples: Optional[int] = None) -> str:
    idxs = list(labels.keys())
    random.shuffle(idxs)
    if max_examples is not None:
        idxs = idxs[:max_examples]
    parts = [INSTRUCTION, ""]
    for i in idxs:
        parts.append(_render_demo(train[i]["question"], train[i]["choice"], labels[i]))
    parts.append("")  # spacing
    return "\n".join(parts)


def run_icm(
    train: List[Dict[str, Any]],
    test: List[Dict[str, Any]],
    client: HyperbolicClient,
    init_k: int = 20,
    max_iters: int = 400,
    alpha: float = 100.0,
    context_cap: Optional[int] = None,     # cap for demos during search
    eval_context_cap: Optional[int] = None # cap for demos during evaluation
) -> float:

    labels = icm_search(
        train=train,
        client=client,
        init_k=init_k,
        max_iters=max_iters,
        alpha=alpha,
        context_cap=context_cap,
    )

    # Build many-shot context from discovered labels
    ctx = _pack_manyshot_context(train, labels, max_examples=eval_context_cap)

    # Evaluate on test
    correct = 0
    for ex in tqdm(test, desc="Eval (many-shot w/ ICM labels)"):
        prompt = f"{ctx}\n{_render_query(ex['question'], ex['choice'])}"
        out = client.complete(
            model=BASE_MODEL,
            prompt=prompt,
            max_tokens=4,
            temperature=0.0,
        )
        pred = _parse_true_false(out)
        gold = "True" if int(ex.get("label", 0)) == 1 else "False"
        if pred == gold:
            correct += 1

    return correct / max(1, len(test))


# ===============================
# Optional helpers to load JSON/JSONL if you want to call directly
# ===============================
def load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
        if not txt:
            return []
        if txt[0] == "[":
            return json.loads(txt)
        return [json.loads(line) for line in txt.splitlines() if line.strip()]
