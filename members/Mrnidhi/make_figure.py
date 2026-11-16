"""Draw figure.png: my models against the baseline (local validation, 250 molecules).

The numbers are the ones logged in experiments/experiment_log.md.

Run from the repo root:
    python -m members.Mrnidhi.make_figure
"""
from pathlib import Path

import matplotlib.pyplot as plt

plt.switch_backend("Agg")      # draw to a file, no window

OUT = Path(__file__).parent / "figure.png"
COSINE, ENTROPY = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (label, MRR@25, measure) grouped by the pipeline the run used
GROUPS = [
    ("Pipeline on main (top 64 peaks)", [
        ("Team baseline, first 50 spectra (E002)", 0.8769, COSINE),
        ("Cosine, all same-mode spectra (E500)", 0.9245, COSINE),
        ("Entropy, unweighted", 0.9267, ENTROPY),
        ("Entropy, weighted: Model 1 (E501)", 0.9294, ENTROPY),
    ]),
    ("Tuned pipeline of PR #8 (top 32 peaks)", [
        ("Cosine, tuned baseline (E006)", 0.9286, COSINE),
        ("Entropy, weighted: Model 1", 0.9341, ENTROPY),
    ]),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    y, ticks, labels = 0.0, [], []
    for title, rows in GROUPS:
        ax.text(0.862, y, title, fontsize=9, color=MUTED, va="center", fontweight="bold")
        y -= 1
        for label, value, colour in rows:
            ax.plot([0.862, value], [y, y], color=GRID, linewidth=1, zorder=1)
            ax.scatter(value, y, s=70, color=colour, zorder=3, edgecolor="white", linewidth=1.2)
            ax.text(value + 0.0018, y, f"{value:.4f}", fontsize=9, color=INK, va="center")
            ticks.append(y)
            labels.append(label)
            y -= 1
        y -= 0.4

    ax.set_yticks(ticks, labels, fontsize=9, color=INK)
    ax.set_xlim(0.862, 0.945)
    ax.set_ylim(y + 0.6, 0.8)
    ax.set_xlabel("Local validation MRR@25 (250 molecules). Axis starts at 0.86, not 0.", fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=name)
               for name, c in (("Entropy similarity", ENTROPY), ("Cosine similarity", COSINE))]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=9, labelcolor=INK,
              borderaxespad=0.2, columnspacing=1.2, handletextpad=0.2)
    fig.suptitle("Entropy is about 0.005 ahead of cosine: within noise on both pipelines",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "One molecule moving from 2nd to 1st place changes MRR by 0.002, "
             "so gaps under about 0.01 are within noise.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
