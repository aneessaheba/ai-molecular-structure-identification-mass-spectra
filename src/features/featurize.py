"""Featurisers shared by all models: binned spectra, peak tensors, molecular fingerprints."""
from functools import lru_cache

import numpy as np

ADDUCTS = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
           "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


# ---------- spectrum features ----------

def bin_spectrum(mz, intensity, max_mz=1000.0, bin_size=1.0):
    """Fixed-length vector: sum of intensities per m/z bin. Good input for MLP/CNN models."""
    n = int(np.ceil(max_mz / bin_size))
    v = np.zeros(n, np.float32)
    idx = (np.asarray(mz) / bin_size).astype(int)
    ok = (idx >= 0) & (idx < n)
    np.add.at(v, idx[ok], np.asarray(intensity)[ok])
    return v


def peak_tensor(mz, intensity, precursor_mz, max_peaks=64):
    """(max_peaks, 3) array of [m/z, intensity, neutral loss] plus a padding mask.
    Good input for transformer / set models. Row 0 is reserved for the precursor."""
    x = np.zeros((max_peaks, 3), np.float32)
    mask = np.zeros(max_peaks, bool)
    x[0] = [precursor_mz, 1.0, 0.0]
    mask[0] = True
    mz = np.asarray(mz)[: max_peaks - 1]
    it = np.asarray(intensity)[: max_peaks - 1]
    k = len(mz)
    x[1:k + 1, 0] = mz
    x[1:k + 1, 1] = it
    x[1:k + 1, 2] = precursor_mz - mz
    mask[1:k + 1] = True
    return x, mask


def metadata_vector(adduct, collision_energy, precursor_mz):
    """One-hot adduct + scaled collision energy + scaled precursor m/z (12 values)."""
    v = np.zeros(len(ADDUCTS) + 2, np.float32)
    if adduct in ADDUCTS:
        v[ADDUCTS.index(adduct)] = 1.0
    v[-2] = 0.0 if collision_energy is None or not np.isfinite(collision_energy) else collision_energy / 100.0
    v[-1] = precursor_mz / 1000.0
    return v


# ---------- molecule features ----------

@lru_cache(maxsize=500_000)
def _fp_bytes(smiles, n_bits, radius):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return gen.GetFingerprintAsNumPy(mol).astype(np.uint8).tobytes()


def morgan_fp(smiles, n_bits=2048, radius=2):
    """Morgan (ECFP4) bit vector as uint8 array of 0/1, or all zeros if SMILES is invalid."""
    b = _fp_bytes(smiles, n_bits, radius)
    return np.zeros(n_bits, np.uint8) if b is None else np.frombuffer(b, np.uint8).copy()


def morgan_fps(smiles_list, n_bits=2048, radius=2):
    return np.stack([morgan_fp(s, n_bits, radius) for s in smiles_list]) if len(smiles_list) else \
        np.zeros((0, n_bits), np.uint8)


def tanimoto(query, fps):
    """Tanimoto between one fingerprint (bits or probabilities) and a matrix of bit fingerprints."""
    q = np.asarray(query, np.float32)
    F = np.asarray(fps, np.float32)
    inter = F @ q
    union = F.sum(1) + q.sum() - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def cosine_sim(query, fps):
    q = np.asarray(query, np.float32)
    F = np.asarray(fps, np.float32)
    return (F @ q) / (np.linalg.norm(F, axis=1) * np.linalg.norm(q) + 1e-9)
