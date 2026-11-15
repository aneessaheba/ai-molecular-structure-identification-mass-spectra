"""TEMPLATE: copy this file to members/<your-username>/model_<name>.py and fill in the TODOs.

Your model is a Scorer: for one molecule it gets that molecule's spectra and a table of
candidate structures (already filtered by precursor mass), and returns one score per candidate.
Higher score = more likely to be the right structure. The shared code does the rest.

Run from the repo root:
    python -m members.<your-username>.model_<name>
"""
import numpy as np

from src.evaluation.validate import evaluate
from src.models.base import Scorer
# Useful helpers (see src/features/featurize.py and src/data/dataset.py):
# from src.data.dataset import load_training, split_by_structure
# from src.features.featurize import bin_spectrum, peak_tensor, metadata_vector, morgan_fps, tanimoto


class MyModel(Scorer):
    name = "my-model"          # TODO: short name, used in logs

    def __init__(self, weights_path=None):
        # TODO: load your trained model here (e.g. torch.load(weights_path))
        self.model = None

    def score(self, spectra, candidates):
        """spectra: list of dicts (mz, intensity, precursor_mz, adduct, ionization_mode, collision_energy)
        candidates: DataFrame (smiles, inchikey14, formula, mass, n_spectra, cosine)
        return: np.ndarray with len(candidates) scores"""
        # TODO: replace this placeholder with your model's prediction.
        # Example: predict a fingerprint from the spectra, compare with each candidate's fingerprint.
        return candidates.cosine.values

    # Optional, only if your model proposes NEW structures (candidate expansion / de novo):
    # def extra_candidates(self, spectra):
    #     return ["CCO", ...]   # SMILES


def train():
    """TODO: training code (or keep it in a notebook). Save weights to members/<you>/weights/ (git-ignored)."""
    pass


if __name__ == "__main__":
    evaluate(MyModel())                    # normal validation (Class 1 molecules)
    # evaluate(MyModel(), hide_true=True)  # only if you implement extra_candidates
