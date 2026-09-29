"""Draw the README figures from the recorded walk-forward study.

Run: uv run --with matplotlib python docs/paper/figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
STUDY = HERE.parent / "results" / "studies" / "20260929T192107Z.json"

INK, MUTED, BLUE, RED = "#1f2933", "#6b7785", "#2563eb", "#c2410c"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

summaries = json.loads(STUDY.read_text())["summaries"]
by_horizon = {s["horizon_minutes"]: s for s in summaries}


def direction_signal() -> None:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    for i, (h, s) in enumerate(sorted(by_horizon.items())):
        p = s["pooled"]
        ax.plot([i, i], [p["ci_low"], p["ci_high"]], color=BLUE, lw=2.5, zorder=2)
        ax.scatter(i, p["gross_correlation"], color=BLUE, s=70, zorder=3)
        ax.scatter(
            [i + 0.18] * len(s["fold_correlations"]),
            s["fold_correlations"],
            color=MUTED,
            s=16,
            alpha=0.7,
            zorder=1,
        )
        ax.annotate(
            f"p = {p['p_value']:.3f}",
            (i, p["gross_correlation"]),
            textcoords="offset points",
            xytext=(-12, -3),
            ha="right",
            color=INK,
            fontsize=10,
        )
    ax.axhline(0, color=MUTED, lw=1, ls="--")
    ax.set_xticks(range(len(by_horizon)))
    ax.set_xticklabels([f"{h} min" for h in sorted(by_horizon)])
    ax.set_xlim(-0.5, len(by_horizon) - 0.5)
    ax.set_xlabel("Label horizon\nblue: pooled with 95% block-bootstrap interval, grey: single folds", color=INK)
    ax.set_ylabel("Out-of-sample correlation\nwith pre-cost price move")
    ax.set_title(
        "A faint direction signal at 30 minutes, gone by 240",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(HERE / "direction_signal.png", facecolor="white")


def signal_vs_cost() -> None:
    slices = by_horizon[30]["top_slices"]
    labels = [f"top {int(t['top_fraction'] * 100)}%" for t in slices]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=200)
    x = range(len(slices))
    w = 0.38
    ax.bar([i - w / 2 for i in x], [t["mean_gross"] * 1e4 for t in slices], w, color=BLUE, label="mean gross move")
    ax.bar([i + w / 2 for i in x], [t["mean_cost"] * 1e4 for t in slices], w, color=RED, label="mean round-trip cost")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Bars the model is most confident about (30-minute horizon)")
    ax.set_ylabel("Basis points")
    ax.set_title(
        "Even the most confident bars earn a fraction of what they cost",
        loc="left",
        color=INK,
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "signal_vs_cost.png", facecolor="white")


direction_signal()
signal_vs_cost()
