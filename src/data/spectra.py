"""Shared spectrum utilities: adduct masses, neutral mass, peak cleaning."""
import re

import numpy as np

PROTON = 1.007276

# Mass added to the neutral molecule M to obtain the singly charged ion m/z.
ADDUCT_DELTA = {
    "[M+H]+": PROTON,
    "[M+NH4]+": 18.033823,
    "[M-H2O+H]+": PROTON - 18.010565,
    "[M-2H2O+H]+": PROTON - 2 * 18.010565,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M-H]-": -PROTON,
    "[M-H2O-H]-": -PROTON - 18.010565,
    "[M+CH2O2-H]-": 44.998201,
    "[M+Cl]-": 34.969402,
}

MONO = {
    "H": 1.0078250319, "C": 12.0, "N": 14.0030740052, "O": 15.9949146221,
    "P": 30.97376151, "S": 31.97207069, "F": 18.99840320, "Cl": 34.96885271,
    "Br": 78.9183376, "I": 126.904468, "Si": 27.9769265327, "B": 11.0093055,
    "Se": 79.9165218, "Na": 22.98976966, "K": 38.9637069, "As": 74.9215964,
}
_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def formula_mass(formula):
    """Monoisotopic neutral mass of a molecular formula, NaN if unparseable/charged."""
    if not isinstance(formula, str) or any(c in formula for c in "+-[]()."):
        return np.nan
    total = 0.0
    for el, n in _FORMULA_RE.findall(formula):
        if el not in MONO:
            return np.nan
        total += MONO[el] * (int(n) if n else 1)
    return total


def neutral_mass(precursor_mz, adduct):
    """Neutral monoisotopic mass implied by a precursor m/z and adduct (NaN if adduct unknown)."""
    delta = ADDUCT_DELTA.get(adduct)
    return np.nan if delta is None else precursor_mz - delta


def clean_peaks(mzs, ints, precursor_mz, top_n=64, min_rel=0.01, max_above_precursor=2.0):
    """Relative-intensity floor, drop peaks above precursor, keep top-N, sqrt-scale, L2-normalise.

    Returns (mz float32, intensity float32) sorted by m/z; empty arrays if nothing survives.
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    ints = np.asarray(ints, dtype=np.float64)
    if mzs.size == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    keep = ints >= min_rel * ints.max()
    if precursor_mz and np.isfinite(precursor_mz):
        keep &= mzs <= precursor_mz + max_above_precursor
    mzs, ints = mzs[keep], ints[keep]
    if mzs.size > top_n:
        idx = np.argpartition(-ints, top_n)[:top_n]
        mzs, ints = mzs[idx], ints[idx]
    order = np.argsort(mzs)
    mzs, ints = mzs[order], np.sqrt(ints[order])
    norm = np.linalg.norm(ints)
    if norm > 0:
        ints = ints / norm
    return mzs.astype(np.float32), ints.astype(np.float32)


def cosine(mz_a, int_a, mz_b, int_b, tol=0.01):
    """Greedy cosine similarity between two cleaned (L2-normalised) spectra."""
    if mz_a.size == 0 or mz_b.size == 0:
        return 0.0
    diff = np.abs(mz_a[:, None] - mz_b[None, :])
    ia, ib = np.nonzero(diff <= tol)
    if ia.size == 0:
        return 0.0
    prod = int_a[ia] * int_b[ib]
    order = np.argsort(-prod)
    used_a, used_b, score = set(), set(), 0.0
    for k in order:
        a, b = ia[k], ib[k]
        if a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            score += prod[k]
    return float(score)
