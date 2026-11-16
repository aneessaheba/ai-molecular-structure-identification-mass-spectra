"""Where does the cosine baseline go wrong?

Runs the baseline on the 250 validation molecules and, for every molecule where the true
structure is not ranked first, records why that might be. Nothing is trained here.

Run from the repo root:
    python -m members.Mrnidhi.analyze_baseline_errors
"""
import numpy as np
import pandas as pd

from src.evaluation.validate import load_validation, rank
from src.models.base import CosineScorer


def describe(mol, spectra, cands):
    """One row of facts about a validation molecule and how the baseline ranked it."""
    ranked = rank(CosineScorer(), spectra, cands).reset_index(drop=True)
    hit = ranked.index[ranked.inchikey14 == mol]
    pos = int(hit[0]) + 1 if len(hit) else None
    row = {
        "molecule": mol,
        "rank": pos,
        "n_candidates": len(ranked),
        "n_query_spectra": len(spectra),
        "mean_query_peaks": np.mean([s["mz"].size for s in spectra]),
        "modes": "/".join(sorted({s["ionization_mode"] for s in spectra})),
        "adducts": "/".join(sorted({s["adduct"] for s in spectra})),
    }
    if pos is None:
        return row
    truth, top = ranked.iloc[pos - 1], ranked.iloc[0]
    row.update({
        "true_cosine": truth.cosine,
        "top_cosine": top.cosine,
        "gap": top.cosine - truth.cosine,
        "true_n_ref": truth.n_spectra,
        "top_n_ref": top.n_spectra,
        "top_is_isomer": top.formula == truth.formula,
        "n_same_formula": int((ranked.formula == truth.formula).sum()),
    })
    return row


def main():
    queries, cands = load_validation()
    df = pd.DataFrame([describe(m, sp, cands[m]) for m, sp in queries.items()])
    # same rule as the competition: no credit if the truth is below position 25
    df["rr"] = np.where(df["rank"] <= 25, 1.0 / df["rank"], 0.0)
    miss = df[df["rank"] != 1]
    lost = (1.0 - miss.rr).sum() / len(df)

    print(f"molecules: {len(df)}   MRR: {df.rr.mean():.4f}   not ranked first: {len(miss)}")
    print(f"MRR lost on those: {lost:.4f}  (perfect ranking would give {df.rr.mean() + lost:.4f})")

    print("\nRank of the true structure:")
    print(df["rank"].fillna(-1).astype(int).value_counts().sort_index().to_string())

    found = miss.dropna(subset=["true_cosine"])
    print(f"\nAmong the {len(found)} misses where the truth is in the candidate list:")
    print(f"  top guess has the same formula (an isomer): {found.top_is_isomer.mean():.0%}")
    print(f"  true structure scored zero cosine:          {(found.true_cosine < 1e-6).mean():.0%}")
    print(f"  gap to the top guess under 0.05:            {(found.gap < 0.05).mean():.0%}")
    print(f"  gap to the top guess over 0.20:             {(found.gap > 0.20).mean():.0%}")
    print(f"  true structure has 3 or fewer ref spectra:  {(found.true_n_ref <= 3).mean():.0%}")

    print("\nCorrect vs missed (medians):")
    df["first"] = np.where(df["rank"] == 1, "ranked first", "missed")
    cols = ["n_candidates", "n_query_spectra", "mean_query_peaks", "true_cosine", "true_n_ref",
            "n_same_formula"]
    print(df.groupby("first")[cols].median().round(2).to_string())

    print("\nMiss rate by ionization mode:")
    print(df.groupby("modes")["first"].apply(lambda s: f"{(s == 'missed').mean():.0%} of {len(s)}").to_string())

    print("\nThe misses, worst first:")
    show = ["rank", "n_candidates", "n_query_spectra", "true_cosine", "top_cosine", "gap",
            "true_n_ref", "top_n_ref", "top_is_isomer", "n_same_formula", "adducts"]
    print(miss.sort_values("rr")[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
