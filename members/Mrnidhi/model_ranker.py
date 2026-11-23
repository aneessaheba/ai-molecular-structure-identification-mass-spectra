"""Model 2: LightGBM LambdaRank re-ranker (E502).

Model 1 gives each candidate one number (entropy similarity). Model 2 describes each
candidate with several numbers, the features, and a ranking model learns how to combine
them. One row per candidate:

  spectrum match    cosine, entropy, matched_peaks
  library evidence  log_n_spectra
  the competition   n_candidates, cosine_rank, n_isomers
  the formula       n_C, n_H, n_N, n_O, n_S, n_P, has_N, rdbe

The ranker is trained by train_ranker.py, which never uses the validation molecules.

Run from the repo root:
    python -m members.Mrnidhi.train_ranker      # once: builds the training table, saves the ranker
    python -m members.Mrnidhi.model_ranker      # validate Model 2
"""
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from members.Mrnidhi.model_entropy import (
    TOL_DA,
    VAL_LIB,
    EntropyScorer,
    Peaks,
    entropy_similarity,
)

MODEL_PATH = Path(__file__).parent / "weights" / "ranker.txt"

ELEMENTS = ("C", "H", "N", "O", "S", "P")
HALOGENS = ("F", "Cl", "Br", "I")
FORMULA_FEATURES = [f"n_{el}" for el in ELEMENTS] + ["has_N", "rdbe"]
FEATURES = ["cosine", "entropy", "matched_peaks", "log_n_spectra", "n_candidates", "cosine_rank",
            *FORMULA_FEATURES, "n_isomers"]


def parse_formula(formula: object) -> dict:
    """'C6H12O6' -> {'C': 6, 'H': 12, 'O': 6}. A missing formula gives {}."""
    if not isinstance(formula, str):
        return {}
    counts: dict = {}
    for element, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[element] = counts.get(element, 0) + int(number or 1)
    return counts


def rdbe(counts: dict) -> float:
    """Rings plus double bonds, from the formula alone: C - (H + halogens)/2 + (N + P)/2 + 1.

    Benzene C6H6 gives 4 (one ring and three double bonds).
    """
    hydrogen_like = counts.get("H", 0) + sum(counts.get(x, 0) for x in HALOGENS)
    return counts.get("C", 0) - hydrogen_like / 2 + (counts.get("N", 0) + counts.get("P", 0)) / 2 + 1


def formula_features(formulas: pd.Series) -> pd.DataFrame:
    """Atom counts, a has-nitrogen flag and RDBE per formula. A missing formula gives NaN."""
    rows = []
    for formula in formulas:
        counts = parse_formula(formula)
        if not counts:
            rows.append(dict.fromkeys(FORMULA_FEATURES, np.nan))
            continue
        row = {f"n_{el}": counts.get(el, 0) for el in ELEMENTS}
        rows.append({**row, "has_N": int(counts.get("N", 0) > 0), "rdbe": rdbe(counts)})
    return pd.DataFrame(rows, columns=FORMULA_FEATURES, index=formulas.index, dtype=float)


def group_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Features that place a candidate among the other candidates of the same molecule."""
    same_formula = candidates.formula.map(candidates.formula.value_counts())
    return pd.DataFrame({
        "log_n_spectra": np.log1p(candidates.n_spectra.astype(float)),
        "n_candidates": float(len(candidates)),
        "cosine_rank": candidates.cosine.rank(ascending=False, method="min"),
        "n_isomers": same_formula.astype(float).fillna(1.0),
    }, index=candidates.index)


def matched_peaks(a: Peaks, b: Peaks, tol: float = TOL_DA) -> int:
    """How many peaks the two spectra share (within `tol` Da). Same walk as entropy_similarity."""
    i = j = count = 0
    while i < len(a.mz) and j < len(b.mz):
        diff = a.mz[i] - b.mz[j]
        if diff < -tol:
            i += 1
        elif diff > tol:
            j += 1
        else:
            count += 1
            i += 1
            j += 1
    return count


class RankerFeatures(EntropyScorer):
    """Builds the feature table. It reuses Model 1's library, cleaning and reference cache."""

    def best_reference(self, query: dict, query_peaks: Peaks, struct_idx: int) -> tuple:
        """(best entropy similarity, index of that reference spectrum, or -1 when nothing matches)."""
        best, best_j = 0.0, -1
        if struct_idx < 0:      # a structure that is not in the library has no spectra
            return best, best_j
        for j in self.lib.spectra_of(struct_idx):
            if self.lib.mode[j] != query["ionization_mode"]:
                continue
            similarity = entropy_similarity(query_peaks, self._ref_peaks(j), self.tol)
            if similarity > best:
                best, best_j = similarity, j
        return best, best_j

    def spectral_features(self, spectra: list, candidates: pd.DataFrame) -> pd.DataFrame:
        """entropy: Model 1's score. matched_peaks: peaks shared with the best-matching reference.
        Both are means over the molecule's query spectra."""
        entropy, matched = np.zeros(len(candidates)), np.zeros(len(candidates))
        query_peaks = [self.prepare(q["mz"], q["intensity"]) for q in spectra]
        for ci, struct_idx in enumerate(candidates.struct_idx.values if spectra else []):
            similarities, counts = [], []
            for query, peaks in zip(spectra, query_peaks):
                best, best_j = self.best_reference(query, peaks, int(struct_idx))
                similarities.append(best)
                counts.append(matched_peaks(peaks, self._ref_peaks(best_j), self.tol) if best_j >= 0 else 0)
            entropy[ci], matched[ci] = np.mean(similarities), np.mean(counts)
        return pd.DataFrame({"entropy": entropy, "matched_peaks": matched}, index=candidates.index)

    def features(self, spectra: list, candidates: pd.DataFrame) -> pd.DataFrame:
        """One row of FEATURES per candidate, in the same order as `candidates`."""
        table = pd.concat([
            candidates[["cosine"]].astype(float),
            self.spectral_features(spectra, candidates),
            group_features(candidates),
            formula_features(candidates.formula),
        ], axis=1)
        return table[FEATURES]


class RankerScorer(RankerFeatures):
    """Model 2: the candidate's score is the trained ranker's output for its feature row.

    For validation the default leaves the validation library out. On Kaggle pass
    exclude_libs=() and the path of the uploaded ranker.txt.
    """

    def __init__(self, weights_path: Path | str = MODEL_PATH, ref_dir: str = "data/processed",
                 exclude_libs: tuple = (VAL_LIB,)) -> None:
        super().__init__(ref_dir=ref_dir, exclude_libs=exclude_libs)
        self.booster = lgb.Booster(model_file=str(weights_path))
        if self.booster.feature_name() != FEATURES:
            raise ValueError(f"{weights_path} was trained on other features: {self.booster.feature_name()}")
        self.name = "lgbm-ranker"

    def score(self, spectra: list, candidates: pd.DataFrame) -> np.ndarray:
        if len(candidates) == 0:
            return np.zeros(0)
        return self.booster.predict(self.features(spectra, candidates))


if __name__ == "__main__":
    from src.evaluation.validate import evaluate

    evaluate(EntropyScorer())       # Model 1, for reference
    evaluate(RankerScorer())        # Model 2 (E502)
