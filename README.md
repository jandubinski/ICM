# ICM TruthfulQA (Prompt-only) — Work Test

This repo implements **Algorithm 1 (ICM)** *without* logical consistency fix and **prompt-only** (no fine-tuning), per the task spec.  
It compares four bars on **TruthfulQA** (1/10 subset provided):

1. **Zero-shot (Base)** — `meta-llama/Meta-Llama-3.1-405B`
2. **Zero-shot (Chat)** — `meta-llama/Meta-Llama-3.1-405B-Instruct`
3. **ICM (Prompt-only)** — Algorithm 1 with simulated-annealing search and mutual predictability (no consistency fix)
4. **Golden Labels (Many-shot ICL)** — Same prompt format as ICM but with **true labels** from the train set

**Models are called via Hyperbolic (OpenAI-compatible) API.**

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HYPERBOLIC_API_KEY=sk-...  # set your API key
```

Hyperbolic API endpoint is OpenAI-compatible:
- `https://api.hyperbolic.xyz/v1/chat/completions`
- Models:
  - Base: `meta-llama/Meta-Llama-3.1-405B`
  - Chat: `meta-llama/Meta-Llama-3.1-405B-Instruct`

## Data

Provided in `data/`:
- `truthfulqa_train.json` (256 rows)
- `truthfulqa_test.json` (100 rows)

Each row: `{ "question": str, "choice": str, "label": 0|1 }` where `label=1` means the claim is **True** (truthful).

## How to Run

### 1) Run all experiments + plot
```bash
python run.py 
python eval_plot.py
```
Outputs:
- `results.json`
- `truthfulqa_bars.png`

