"""Baseline: precursor-mass candidate retrieval + spectral library search.

For each query molecule:
  1. neutral mass from each spectrum's precursor m/z and adduct (median across spectra)
  2. candidates = reference structures whose neutral mass is within `ppm`
  3. score each candidate = mean over query spectra of the best cosine against the
     candidate's reference spectra (same ionisation mode)
  4. rank by score, ties broken by how often the structure appears in the libraries
"""
import os

import numpy as np
import pandas as pd

from src.data.spectra import clean_peaks, cosine, neutral_mass

K = 25


class ReferenceLibrary:
    def __init__(self, ref_dir, exclude_libs=()):
        meta = pd.read_parquet(os.path.join(ref_dir, "ref_meta.parquet"))
        self.structs = pd.read_parquet(os.path.join(ref_dir, "structures.parquet"))
        peaks = np.load(os.path.join(ref_dir, "ref_peaks.npz"))
        self.mz, self.inten = peaks["mz"], peaks["intensity"]
        if exclude_libs:
            meta = meta[~meta.ingest_lib.isin(exclude_libs)]
        meta = meta.sort_values("struct_idx").reset_index(drop=True)
        self.meta = meta
        n = len(self.structs)
        counts = np.bincount(meta.struct_idx.values, minlength=n)
        self.spec_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self.spec_count = counts
        # Structures with no remaining spectra are not retrievable (mimics unseen molecules).
        self.available = counts > 0
        self.masses = self.structs.mass.values
        self.prior = np.log1p(counts)
        self.mode = meta.ionization_mode.values
        self.pstart = meta.peak_start.values
        self.plen = meta.n_peaks.values

    def candidates(self, mass, ppm):
        tol = mass * ppm * 1e-6
        lo, hi = np.searchsorted(self.masses, [mass - tol, mass + tol])
        idx = np.arange(lo, hi)
        return idx[self.available[idx]]

    def spectra_of(self, s):
        a = self.spec_start[s]
        return range(a, a + self.spec_count[s])

    def peaks(self, j):
        a = self.pstart[j]
        return self.mz[a:a + self.plen[j]], self.inten[a:a + self.plen[j]]


def query_molecule(lib, spectra, ppm=15, use_cosine=True, max_ref_per_struct=None, tol=0.01, agg="mean",
                   cross_mode=False):
    """spectra: list of dicts with precursor_mz, adduct, ionization_mode, mz, intensity (cleaned).
    max_ref_per_struct: compare against at most this many reference spectra per candidate
    (None = all; the original baseline E002 used 50 with top_n=64).
    tol: m/z tolerance (Da) for matching peaks. agg: how to combine the per-spectrum best scores
    ("mean" or "max"). cross_mode: also compare against reference spectra of the other ionisation mode.
    Returns list of (struct_idx, score) best first."""
    masses = [neutral_mass(s["precursor_mz"], s["adduct"]) for s in spectra]
    masses = [m for m in masses if np.isfinite(m)]
    if not masses:
        return []
    cands = lib.candidates(float(np.median(masses)), ppm)
    if cands.size == 0:
        return []
    scores = np.zeros(cands.size)
    if use_cosine:
        for ci, s in enumerate(cands):
            refs = list(lib.spectra_of(s))[:max_ref_per_struct] if max_ref_per_struct else lib.spectra_of(s)
            per_query = []
            for q in spectra:
                best = 0.0
                for j in refs:
                    if not cross_mode and lib.mode[j] != q["ionization_mode"]:
                        continue
                    mz_r, in_r = lib.peaks(j)
                    best = max(best, cosine(q["mz"], q["intensity"], mz_r, in_r, tol=tol))
                per_query.append(best)
            scores[ci] = np.max(per_query) if agg == "max" else np.mean(per_query)
    # tiny prior so ties (and the mass-only baseline) prefer frequently measured structures
    total = scores + 1e-3 * lib.prior[cands]
    order = np.argsort(-total)
    return [(int(cands[i]), float(total[i])) for i in order]


def candidate_table(lib, spectra, ppm=15, with_cosine=True):
    """All mass-matched candidates for one molecule as a DataFrame:
    struct_idx, smiles, inchikey14, formula, mass, n_spectra, cosine (baseline spectral score)."""
    ranked = query_molecule(lib, spectra, ppm=ppm, use_cosine=with_cosine)
    if not ranked:
        return pd.DataFrame(columns=["struct_idx", "smiles", "inchikey14", "formula", "mass",
                                     "n_spectra", "cosine"])
    idx = np.array([s for s, _ in ranked])
    score = np.array([v for _, v in ranked]) - 1e-3 * lib.prior[idx]
    st = lib.structs.iloc[idx]
    return pd.DataFrame({
        "struct_idx": idx, "smiles": st.smiles.values, "inchikey14": st.inchikey14.values,
        "formula": st.formula.values, "mass": st.mass.values,
        "n_spectra": lib.spec_count[idx], "cosine": score,
    })


def prepare_queries(df, **clean_kw):
    """Group a test-format dataframe into {molecule_id: [spectrum dicts]} with cleaned peaks."""
    out = {}
    for row in df.itertuples(index=False):
        mz, it = clean_peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, **clean_kw)
        ce = getattr(row, "collision_energy_ev", None)
        try:
            ce = float(np.mean(ce)) if len(ce) else np.nan
        except TypeError:
            ce = np.nan
        out.setdefault(row.molecule_id, []).append(dict(
            precursor_mz=row.precursor_mz, adduct=row.adduct,
            ionization_mode=row.ionization_mode, collision_energy=ce, mz=mz, intensity=it))
    return out


def predict(lib, queries, k=K, **kw):
    """Returns {molecule_id: [smiles, ...]} (up to k, best first, unique InChIKey14)."""
    smiles = lib.structs.smiles.values
    preds = {}
    for mol_id, spectra in queries.items():
        ranked = query_molecule(lib, spectra, **kw)[:k]
        preds[mol_id] = [smiles[s] for s, _ in ranked]
    return preds


def to_submission(preds, molecule_ids, fallback="CCO"):
    rows = []
    for m in molecule_ids:
        guesses = preds.get(m) or [fallback]
        rows.append({"molecule_id": m, "smiles": ";".join(guesses[:K])})
    return pd.DataFrame(rows)
