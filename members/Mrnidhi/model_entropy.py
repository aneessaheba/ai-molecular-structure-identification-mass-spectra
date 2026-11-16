"""Model 1: spectral entropy similarity scorer (Li et al., Nature Methods 2021).

Same idea as the cosine baseline: compare the unknown molecule's spectra with each
candidate's reference spectra and score the candidate by how well they match. Two changes:

  1. The match is measured with entropy similarity instead of cosine.
  2. Every reference spectrum in the same ionization mode is used.

Score of a candidate = mean over the query spectra of the best similarity against any of
the candidate's reference spectra in the same mode.

The similarity is written here in plain NumPy, so the Kaggle notebook (internet off) needs
no extra package. tests/test_entropy.py checks it against the ms_entropy package.

Run from the repo root:
    python -m members.Mrnidhi.model_entropy
"""
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from src.data.spectra import cosine
from src.models.base import Scorer
from src.models.library_search import ReferenceLibrary

VAL_LIB = "enveda-np-examples"
MERGE_DA = 0.05       # peaks closer than this are merged into one before comparing
NOISE_FLOOR = 0.01    # peaks below 1% of the tallest peak are dropped
TOL_DA = 0.02         # two peaks this close in m/z count as the same peak
_LIBS: dict = {}


class Peaks(NamedTuple):
    """A spectrum ready for entropy similarity: sorted m/z, intensities that sum to 1,
    and x * log2(x) of every intensity (computed once, reused in every comparison)."""
    mz: list
    intensity: list
    xlogx: list


def merge_close_peaks(mz: np.ndarray, intensity: np.ndarray, window: float = MERGE_DA):
    """Merge peaks that sit within `window` Da of each other, strongest peak first.

    The merged peak gets the summed intensity and the intensity-weighted mean m/z.
    Returns (mz, intensity) sorted by m/z, with every gap at least `window`.
    """
    mz, intensity = np.asarray(mz, np.float32), np.asarray(intensity, np.float32)
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    while mz.size > 1 and np.any(np.diff(mz) < window):
        left = intensity.copy()
        new_mz, new_int = [], []
        for i in np.argsort(left)[::-1]:
            if left[i] <= 0:      # already merged into a stronger neighbour
                continue
            lo, hi = i, i + 1
            while lo > 0 and mz[i] - mz[lo - 1] <= window:
                lo -= 1
            while hi < mz.size and mz[hi] - mz[i] <= window:
                hi += 1
            total = left[lo:hi].sum()
            new_mz.append((left[lo:hi] * mz[lo:hi]).sum() / total)
            new_int.append(total)
            left[lo:hi] = 0
        order = np.argsort(new_mz)
        mz = np.asarray(new_mz, np.float32)[order]
        intensity = np.asarray(new_int, np.float32)[order]
    return mz, intensity


def entropy_weight(intensity: np.ndarray) -> np.ndarray:
    """The paper's weighting: a spectrum with low entropy (few dominant peaks) gets its small
    peaks raised, so one giant peak does not decide the whole comparison."""
    entropy = float(-np.sum(intensity * np.log(intensity)))
    if entropy >= 3:
        return intensity
    weighted = np.power(intensity, 0.25 + 0.25 * entropy)
    return weighted / weighted.sum()


def prepare_spectrum(mz: np.ndarray, intensity: np.ndarray, weighted: bool = True) -> Peaks:
    """Team-cleaned spectrum -> Peaks.

    The team's cleaned intensities are square-rooted. Entropy similarity expects raw-like
    intensities, so they are squared back first.
    """
    mz = np.asarray(mz, np.float32)
    intensity = np.asarray(intensity, np.float32)
    keep = (mz > 0) & (intensity > 0)
    mz, intensity = merge_close_peaks(mz[keep], intensity[keep] ** 2)
    if mz.size == 0:
        return Peaks([], [], [])
    keep = intensity >= NOISE_FLOOR * intensity.max()
    mz, intensity = mz[keep], intensity[keep]
    intensity = intensity / intensity.sum()
    if weighted:
        intensity = entropy_weight(intensity)
    return Peaks(mz.tolist(), intensity.tolist(), (intensity * np.log2(intensity)).tolist())


def entropy_similarity(a: Peaks, b: Peaks, tol: float = TOL_DA) -> float:
    """Entropy similarity of two prepared spectra: 1 = identical, 0 = no shared peaks.

    Walk through both peak lists in m/z order. For every pair of peaks within `tol`, add
    f(a + b) - f(a) - f(b), where f(x) = x * log2(x). Halve the total at the end.
    """
    i = j = 0
    total = 0.0
    while i < len(a.mz) and j < len(b.mz):
        diff = a.mz[i] - b.mz[j]
        if diff < -tol:
            i += 1
        elif diff > tol:
            j += 1
        else:
            both = a.intensity[i] + b.intensity[j]
            total += both * math.log2(both) - a.xlogx[i] - b.xlogx[j]
            i += 1
            j += 1
    return total / 2


def get_library(ref_dir: str, exclude_libs: tuple) -> ReferenceLibrary:
    """Load the reference library once and reuse it."""
    if isinstance(exclude_libs, str):     # a bare string would be split into letters and exclude nothing
        exclude_libs = (exclude_libs,)
    key = (ref_dir, tuple(sorted(exclude_libs)))
    if key not in _LIBS:
        _LIBS[key] = ReferenceLibrary(ref_dir, exclude_libs=list(key[1]))
    return _LIBS[key]


class EntropyScorer(Scorer):
    """metric: 'entropy' (weighted, the paper's default), 'unweighted', or 'cosine'.

    'cosine' is here only for a fair comparison: the same reference spectra, the old measure.
    For validation the default leaves the validation library out. On Kaggle pass
    exclude_libs=() so the full library is used.
    """

    def __init__(self, ref_dir: str = "data/processed", exclude_libs: tuple = (VAL_LIB,),
                 metric: str = "entropy", tol: float = TOL_DA) -> None:
        self.lib = get_library(ref_dir, exclude_libs)
        self.metric, self.tol = metric, tol
        self.name = f"{metric}-all-refs"
        self._ref_cache: dict = {}

    def prepare(self, mz: np.ndarray, intensity: np.ndarray) -> Peaks:
        return prepare_spectrum(mz, intensity, weighted=self.metric == "entropy")

    def _ref_peaks(self, j: int) -> Peaks:
        if j not in self._ref_cache:
            self._ref_cache[j] = self.prepare(*self.lib.peaks(j))
        return self._ref_cache[j]

    def _similarity(self, query: dict, query_peaks: Peaks, j: int) -> float:
        if self.metric == "cosine":
            return cosine(query["mz"], query["intensity"], *self.lib.peaks(j))
        return entropy_similarity(query_peaks, self._ref_peaks(j), self.tol)

    def best_match(self, query: dict, query_peaks: Peaks, struct_idx: int) -> float:
        """Best similarity between one query spectrum and a candidate's same-mode references."""
        best = 0.0
        if struct_idx < 0:      # a structure that is not in the library has no spectra
            return best
        for j in self.lib.spectra_of(struct_idx):
            if self.lib.mode[j] == query["ionization_mode"]:
                best = max(best, self._similarity(query, query_peaks, j))
        return best

    def score(self, spectra: list, candidates: pd.DataFrame) -> np.ndarray:
        scores = np.zeros(len(candidates))
        if not spectra:
            return scores
        query_peaks = [self.prepare(q["mz"], q["intensity"]) for q in spectra]
        for ci, struct_idx in enumerate(candidates.struct_idx.values):
            scores[ci] = np.mean([self.best_match(q, p, int(struct_idx))
                                  for q, p in zip(spectra, query_peaks)])
        return scores


if __name__ == "__main__":
    from src.evaluation.validate import evaluate

    evaluate(EntropyScorer(metric="cosine"))       # fair reference: old measure, all references
    evaluate(EntropyScorer(metric="unweighted"))
    evaluate(EntropyScorer(metric="entropy"))      # Model 1 (E501)
