"""Small one-at-a-time changes to Model 1 (experiments E510 and E511).

Each block changes a single setting of the entropy scorer and prints the validation score,
so it is clear which change did what. None of them gave a clear gain over the defaults.

Run from the repo root:
    python -m members.Mrnidhi.entropy_variants
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from members.Mrnidhi.model_entropy import EntropyScorer, to_entropy_peaks
from src.evaluation.validate import evaluate
from src.models.base import Scorer


class Blend(EntropyScorer):
    """weight * entropy + (1 - weight) * cosine, both on all same-mode references."""

    def __init__(self, weight: float) -> None:
        super().__init__(metric="entropy")
        self.cosine = EntropyScorer(metric="cosine")
        self.weight = weight
        self.name = f"blend entropy {weight:.2f} / cosine {1 - weight:.2f}"

    def score(self, spectra: list[dict], candidates: pd.DataFrame) -> np.ndarray:
        return (self.weight * super().score(spectra, candidates)
                + (1 - self.weight) * self.cosine.score(spectra, candidates))


class Aggregate(EntropyScorer):
    """Combine the per-spectrum best matches with mean, max, median or min."""

    def __init__(self, how: str) -> None:
        super().__init__()
        self.how = getattr(np, how)
        self.name = f"entropy agg={how}"

    def score(self, spectra: list[dict], candidates: pd.DataFrame) -> np.ndarray:
        peaks = [to_entropy_peaks(q["mz"], q["intensity"]) for q in spectra]
        return np.array([self.how([self.best_match(q, p, int(s)) for q, p in zip(spectra, peaks)])
                         for s in candidates.struct_idx.values])


class AdductBonus(EntropyScorer):
    """Raise the similarity when the reference spectrum has the same adduct as the query."""

    def __init__(self, bonus: float) -> None:
        super().__init__()
        self.bonus = bonus
        self.adduct = self.lib.meta.adduct.values
        self.name = f"entropy same-adduct bonus={bonus}"

    def best_match(self, query: dict, query_peaks: np.ndarray, struct_idx: int) -> float:
        best = 0.0
        for j in self.lib.spectra_of(struct_idx):
            if self.lib.mode[j] != query["ionization_mode"]:
                continue
            sim = self._similarity(query, query_peaks, j)
            if self.adduct[j] == query["adduct"]:
                sim *= 1 + self.bonus
            best = max(best, sim)
        return best


def run(scorer: Scorer) -> None:
    r = evaluate(scorer, verbose=False)
    print(f"{scorer.name:40s} MRR {r['mrr']:.4f}  top1 {r['top1']:.3f}  top10 {r['top10']:.3f}")


def with_tolerance(tol: float) -> EntropyScorer:
    scorer = EntropyScorer(tol=tol)
    scorer.name = f"entropy tol={tol}"
    return scorer


if __name__ == "__main__":
    print("E510 peak tolerance in Da")
    for tol in [0.005, 0.01, 0.02, 0.05]:
        run(with_tolerance(tol))
    print("E510 how a molecule's spectra are combined")
    for how in ["mean", "max", "median", "min"]:
        run(Aggregate(how))
    print("E510 blend with cosine")
    for weight in [0.25, 0.5, 0.75]:
        run(Blend(weight))
    print("E511 same-adduct bonus")
    for bonus in [0.0, 0.1, 0.3]:
        run(AdductBonus(bonus))
