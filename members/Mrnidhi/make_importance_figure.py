"""Draw figure_importance.png: which features the trained ranker (Model 2) relies on.

Importance is "gain": how much each feature improved the ranking objective over all the
splits that use it, shown as a share of the total. Two features that carry the same
information (cosine and entropy) share their gain, so read the groups, not single bars.

Run from the repo root, after train_ranker.py:
    python -m members.Mrnidhi.make_importance_figure
"""
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt

from members.Mrnidhi.model_ranker import FORMULA_FEATURES, MODEL_PATH

plt.switch_backend("Agg")      # draw to a file, no window

OUT = Path(__file__).parent / "figure_importance.png"
BAR, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"
GROUPS = {
    "Spectrum match": ["cosine", "entropy", "matched_peaks", "cosine_rank"],
    "Library evidence": ["log_n_spectra"],
    "Candidate list": ["n_candidates", "n_isomers"],
    "Formula": FORMULA_FEATURES,
}
LABELS = {
    "cosine": "Cosine score", "entropy": "Entropy score (Model 1)", "matched_peaks": "Matched peaks",
    "cosine_rank": "Rank by cosine", "log_n_spectra": "Reference spectra (log)",
    "n_candidates": "Number of candidates", "n_isomers": "Same-formula candidates",
    "n_C": "Carbon atoms", "n_H": "Hydrogen atoms", "n_N": "Nitrogen atoms", "n_O": "Oxygen atoms",
    "n_S": "Sulfur atoms", "n_P": "Phosphorus atoms", "has_N": "Has nitrogen",
    "rdbe": "Rings + double bonds",
}


def gain_shares(model_path: Path = MODEL_PATH) -> dict:
    """{feature: share of the total gain}, largest first."""
    booster = lgb.Booster(model_file=str(model_path))
    gain = booster.feature_importance(importance_type="gain")
    shares = dict(zip(booster.feature_name(), gain / gain.sum()))
    return dict(sorted(shares.items(), key=lambda item: -item[1]))


def group_shares(shares: dict) -> dict:
    return {group: sum(shares[f] for f in features) for group, features in GROUPS.items()}


def main() -> None:
    shares = gain_shares()
    groups = group_shares(shares)
    names = list(shares)[::-1]          # largest at the top

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    ax.barh([LABELS[n] for n in names], [100 * shares[n] for n in names], color=BAR, height=0.62)
    for y, name in enumerate(names):
        value = 100 * shares[name]
        ax.text(value + 0.6, y, f"{value:.1f}%" if value >= 0.05 else "0%", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, 100 * max(shares.values()) * 1.12)
    ax.set_xlabel("Share of the ranker's total gain (%)", fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", labelsize=9, colors=INK, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    top = next(iter(shares))
    fig.suptitle(f"One feature, {LABELS[top].lower()}, carries {100 * shares[top]:.0f}% of the ranker's gain",
                 fontsize=11, color=INK, x=0.01, ha="left")
    summary = " · ".join(f"{group} {100 * share:.1f}%" for group, share in groups.items())
    fig.text(0.01, 0.01, f"By group: {summary}.\nLightGBM LambdaRank, 300 trees, gain importance.",
             fontsize=8, color=MUTED, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT}")
    for group, share in groups.items():
        print(f"  {group}: {100 * share:.1f}%")


if __name__ == "__main__":
    main()
