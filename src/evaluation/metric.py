"""Local re-implementation of the competition metric (MRR@25 on InChIKey14).

Official notebook: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
The official metric tautomer-canonicalises with RDKit 2026.03.3 before taking the InChIKey14.
"""
from functools import lru_cache

import numpy as np

K = 25


@lru_cache(maxsize=None)
def smiles_to_ik14(smiles, tautomer=True):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if tautomer:
        try:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        except Exception:
            pass
    key = Chem.MolToInchiKey(mol)
    return key.split("-")[0] if key else None


def mrr_from_keys(pred_keys, true_keys, k=K):
    """pred_keys: list of ranked key lists; true_keys: list of keys. Returns MRR@k."""
    scores = []
    for preds, truth in zip(pred_keys, true_keys):
        rr = 0.0
        for rank, key in enumerate(preds[:k], start=1):
            if key == truth:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return float(np.mean(scores)) if scores else 0.0


def mrr_from_smiles(pred_smiles, true_smiles, k=K):
    """Same as the official metric: SMILES lists → tautomer-canonical InChIKey14 → MRR@k."""
    pred_keys = [[smiles_to_ik14(s) for s in preds[:k]] for preds in pred_smiles]
    true_keys = [smiles_to_ik14(s) for s in true_smiles]
    return mrr_from_keys(pred_keys, true_keys, k)
