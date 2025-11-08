# eval_plot.py
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_bars(series, title: str, save_path: str):
    """
    series: dict with accuracy floats in [0,1], expected keys:
        "zero_shot_base"
        "zero_shot_chat"
        "golden_icl"
        "icm"
    """

    # Canonical label remap for plotting
    mapped = {}

    if "zero_shot_base" in series:
        mapped["Zero-shot"] = series["zero_shot_base"]
    if "zero_shot_chat" in series:
        mapped["Zero-shot (Chat)"] = series["zero_shot_chat"]
    if "icm" in series:
        mapped["Unsupervised (Ours)"] = series["icm"]
    if "golden_icl" in series:
        mapped["Golden Supervision"] = series["golden_icl"]

    # Plot order
    labels = ["Zero-shot", "Zero-shot (Chat)", "Unsupervised (Ours)", "Golden Supervision"]
    labels = [lbl for lbl in labels if lbl in mapped]

    vals = np.array([mapped[k] for k in labels]) * 100

    # Colors matching your example
    colors = {
        "Zero-shot": "#A67899",
        "Zero-shot (Chat)": "#A67899",
        "Golden Supervision": "#E1B866",
        "Unsupervised (Ours)": "#90C8D8",
    }

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    x = np.arange(len(vals))

    barlist = ax.bar(x, vals, color=[colors[l] for l in labels], width=0.65)

    # Add hatch style for Chat bar
    for i, lbl in enumerate(labels):
        if lbl == "Zero-shot (Chat)":
            barlist[i].set_hatch('o')

    # Horizontal grid every 10%
    ax.set_ylim(30, 100)
    ax.set_yticks(np.arange(30, 101, 10))
    ax.yaxis.grid(True, linestyle='-', linewidth=0.6, color='#c6e8cd')
    ax.set_axisbelow(True)

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=17, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=11)

    # Legend
    legend_handles = []
    for lbl in labels:
        patch = plt.Rectangle((0, 0), 1, 1, color=colors[lbl])
        if lbl == "Zero-shot (Chat)":
            patch.set_hatch('o')
        legend_handles.append(patch)

    ax.legend(
        legend_handles, labels,
        bbox_to_anchor=(0.5, -0.25), loc='upper center',
        ncol=2, frameon=False
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", type=str, default="results.json",
                    help="Path to results.json saved by run.py")
    ap.add_argument("--save_path", type=str, default="truthfulqa_plot.png",
                    help="Where to save the output plot")
    args = ap.parse_args()

    with open(args.results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    plot_bars(results, title="TruthfulQA", save_path=args.save_path)
    print(f"Saved plot → {args.save_path}")


if __name__ == "__main__":
    main()
