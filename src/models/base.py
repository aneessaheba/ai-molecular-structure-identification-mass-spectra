"""The one interface every team model implements.

A model is a *scorer*: given one molecule's spectra and a table of candidate structures,
it returns one score per candidate (higher = more likely). The shared pipeline handles
everything else: candidate retrieval by precursor mass, ranking, validation and the
Kaggle submission file.

    class MyModel(Scorer):
        name = "fingerprint-mlp"
        def score(self, spectra, candidates):
            ...  # return np.ndarray of len(candidates)

`spectra`: list of dicts, one per spectrum of the molecule, with keys
    mz, intensity (cleaned float32 arrays), precursor_mz, adduct, ionization_mode, collision_energy
`candidates`: pandas DataFrame with columns
    smiles, inchikey14, formula, mass, n_spectra (times seen in libraries),
    cosine (baseline library-search score; useful as a feature)

Optional: override `extra_candidates(spectra)` to return additional SMILES that are not in the
training library (e.g. from PubChem/COCONUT or a generative model). They are added to
`candidates` (with n_spectra=0 and cosine=0) before `score` is called.
"""
import numpy as np


class Scorer:
    name = "scorer"

    def score(self, spectra, candidates):
        raise NotImplementedError

    def extra_candidates(self, spectra):
        return []


class CosineScorer(Scorer):
    """The baseline (E002): library-search cosine."""
    name = "library-cosine"

    def score(self, spectra, candidates):
        return candidates.cosine.values


class MassOnlyScorer(Scorer):
    """E001: mass match only, prefer frequently measured structures."""
    name = "mass-only"

    def score(self, spectra, candidates):
        return np.log1p(candidates.n_spectra.values.astype(float))
