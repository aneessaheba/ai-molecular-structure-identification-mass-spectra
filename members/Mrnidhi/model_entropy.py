"""Model 1: spectral entropy similarity scorer (Li et al., Nature Methods 2021).

Same idea as the cosine baseline: compare the unknown molecule's spectra with each
candidate's reference spectra and score the candidate by how well they match. Two changes:

  1. The match is measured with entropy similarity instead of cosine.
  2. Every reference spectrum in the same ionization mode is used. The shared baseline
     only looks at a candidate's first 50 spectra, before the mode filter.

Score of a candidate = mean over the query spectra of the best similarity against any of
the candidate's reference spectra in the same mode.

Run from the repo root:
    python -m members.Mrnidhi.model_entropy
"""
from __future__ import annotations

import ms_entropy
import numpy as np
import pandas as pd

from src.data.spectra import cosine
from src.evaluation.validate import evaluate
from src.models.base import Scorer
from src.models.library_search import ReferenceLibrary

VAL_LIB = "enveda-np-examples"
_LIBS: dict[tuple, ReferenceLibrary] = {}


def get_library(ref_dir: str, exclude_libs: tuple[str, ...]) -> ReferenceLibrary:
    """Load the reference library once and reuse it."""
    key = (ref_dir, exclude_libs)
    if key not in _LIBS:
        _LIBS[key] = ReferenceLibrary(ref_dir, exclude_libs=list(exclude_libs))
    return _LIBS[key]


def to_entropy_peaks(mz: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Cleaned team spectrum -> (n, 2) array ready for ms_entropy.

    The team's cleaned intensities are square-rooted, and entropy similarity expects
    raw-like intensities, so they are squared back first.
    """
    peaks = np.column_stack([mz, intensity.astype(np.float64) ** 2]).astype(np.float32)
    return ms_entropy.clean_spectrum(peaks)


class EntropyScorer(Scorer):
    """metric: 'entropy' (weighted, the paper's default), 'unweighted', or 'cosine'.

    'cosine' is here only for a fair comparison: the same reference spectra, the old measure.
    """

    def __init__(self, ref_dir: str = "data/processed", exclude_libs: tuple[str, ...] = (VAL_LIB,),
                 metric: str = "entropy", tol: float = 0.02) -> None:
        self.lib = get_library(ref_dir, tuple(exclude_libs))
        self.metric, self.tol = metric, tol
        self.name = f"{metric}-all-refs"
        self._ref_cache: dict[int, np.ndarray] = {}

    def _ref_peaks(self, j: int) -> np.ndarray:
        if j not in self._ref_cache:
            self._ref_cache[j] = to_entropy_peaks(*self.lib.peaks(j))
        return self._ref_cache[j]

    def _similarity(self, query: dict, query_peaks: np.ndarray, j: int) -> float:
        if self.metric == "cosine":
            return cosine(query["mz"], query["intensity"], *self.lib.peaks(j))
        fn = (ms_entropy.calculate_entropy_similarity if self.metric == "entropy"
              else ms_entropy.calculate_unweighted_entropy_similarity)
        return float(fn(query_peaks, self._ref_peaks(j), ms2_tolerance_in_da=self.tol,
                        clean_spectra=False))

    def best_match(self, query: dict, query_peaks: np.ndarray, struct_idx: int) -> float:
        """Best similarity between one query spectrum and a candidate's same-mode references."""
        best = 0.0
        for j in self.lib.spectra_of(struct_idx):
            if self.lib.mode[j] == query["ionization_mode"]:
                best = max(best, self._similarity(query, query_peaks, j))
        return best

    def score(self, spectra: list[dict], candidates: pd.DataFrame) -> np.ndarray:
        query_peaks = [to_entropy_peaks(q["mz"], q["intensity"]) for q in spectra]
        scores = np.zeros(len(candidates))
        for ci, struct_idx in enumerate(candidates.struct_idx.values):
            if struct_idx < 0:      # a structure that is not in the library has no spectra
                continue
            scores[ci] = np.mean([self.best_match(q, p, int(struct_idx))
                                  for q, p in zip(spectra, query_peaks)])
        return scores


if __name__ == "__main__":
    evaluate(EntropyScorer(metric="cosine"))       # fair reference: old measure, all references
    evaluate(EntropyScorer(metric="unweighted"))
    evaluate(EntropyScorer(metric="entropy"))      # Model 1 (E501)
