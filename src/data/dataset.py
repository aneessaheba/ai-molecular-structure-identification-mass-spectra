"""Load cleaned training spectra for model training, without leaking validation molecules.

Example:
    from src.data.dataset import load_training
    df = load_training(max_per_structure=3)          # ~740k spectra, ~30 s, ~5 GB RAM
    df = load_training(libs=["gnps", "riken"], sample=50_000)   # quick experiments

Returned columns: inchikey14, smiles, formula, ingest_lib, adduct, ionization_mode,
precursor_mz, collision_energy (mean eV or NaN), mz (float32 array), intensity (float32 array).
Peaks are cleaned with src.data.spectra.clean_peaks (1% floor, top-N, sqrt, L2-normalised).
"""
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.spectra import ADDUCT_DELTA, clean_peaks

TRAIN_PATH = "data/raw/train.parquet"
VAL_LIB = "enveda-np-examples"
TEST_ADDUCTS = list(ADDUCT_DELTA)
COLS = ["ingest_lib", "normalized_smiles", "inchikey14", "molecular_formula", "ionization_mode",
        "adduct", "precursor_mz", "precursor_error_ppm", "collision_energy_ev",
        "ms2_mzs", "ms2_normalized_intensities"]


def validation_keys(train_path=TRAIN_PATH):
    """InChIKey14s of the validation molecules. Never train on these."""
    t = pq.read_table(train_path, columns=["ingest_lib", "inchikey14"],
                      filters=[("ingest_lib", "=", VAL_LIB)])
    return set(t.column("inchikey14").to_pylist())


def _mean_ce(x):
    return float(np.mean(x)) if x is not None and len(x) else np.nan


def load_training(train_path=TRAIN_PATH, libs=None, exclude_val=True, max_ppm=20.0,
                  adducts=TEST_ADDUCTS, max_per_structure=None, sample=None, top_n=64,
                  min_peaks=3, seed=0):
    """Load cleaned training spectra.

    libs:              list of ingest_lib values to keep (None = all except the validation library)
    exclude_val:       drop every spectrum whose structure is a validation molecule (no leakage)
    max_ppm:           drop spectra with |precursor_error_ppm| above this (likely mislabelled)
    adducts:           keep only these adducts (default: the 10 that occur in the test set)
    max_per_structure: keep at most this many spectra per structure (saves memory)
    sample:            randomly keep this many spectra in total (for quick experiments)
    """
    rng = np.random.default_rng(seed)
    meta_cols = [c for c in COLS if c not in ("ms2_mzs", "ms2_normalized_intensities")]
    meta = pq.read_table(train_path, columns=meta_cols).to_pandas()
    meta["row"] = np.arange(len(meta))

    keep = meta.ingest_lib != VAL_LIB
    if libs is not None:
        keep &= meta.ingest_lib.isin(libs)
    if exclude_val:
        keep &= ~meta.inchikey14.isin(validation_keys(train_path))
    if max_ppm is not None:
        keep &= meta.precursor_error_ppm.abs().fillna(0) <= max_ppm
    if adducts is not None:
        keep &= meta.adduct.isin(adducts)
    meta = meta[keep]
    if max_per_structure:
        meta = meta.sample(frac=1, random_state=seed).groupby("inchikey14").head(max_per_structure)
    if sample and sample < len(meta):
        meta = meta.iloc[rng.choice(len(meta), sample, replace=False)]
    meta = meta.sort_values("row")

    # Read peaks only for the kept rows, one row group at a time.
    pf = pq.ParquetFile(train_path)
    wanted = meta.row.values
    mz_out, in_out, start = [], [], 0
    for rg in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        lo, hi = np.searchsorted(wanted, [start, start + n])
        if hi > lo:
            tbl = pf.read_row_group(rg, columns=["ms2_mzs", "ms2_normalized_intensities"])
            local = wanted[lo:hi] - start
            tbl = tbl.take(local)
            for mz, it, pmz in zip(tbl.column(0).to_pylist(), tbl.column(1).to_pylist(),
                                   meta.precursor_mz.values[lo:hi]):
                m, s = clean_peaks(mz or [], it or [], pmz, top_n=top_n)
                mz_out.append(m)
                in_out.append(s)
        start += n

    df = pd.DataFrame({
        "inchikey14": meta.inchikey14.values,
        "smiles": meta.normalized_smiles.values,
        "formula": meta.molecular_formula.values,
        "ingest_lib": meta.ingest_lib.values,
        "adduct": meta.adduct.values,
        "ionization_mode": meta.ionization_mode.values,
        "precursor_mz": meta.precursor_mz.values,
        "collision_energy": meta.collision_energy_ev.map(_mean_ce).values,
        "mz": mz_out,
        "intensity": in_out,
    })
    df = df[df.mz.map(len) >= min_peaks].reset_index(drop=True)
    return df


def split_by_structure(df, val_frac=0.1, seed=0):
    """Train/val split where no structure (inchikey14) appears on both sides."""
    keys = df.inchikey14.unique()
    rng = np.random.default_rng(seed)
    val_keys = set(rng.choice(keys, int(len(keys) * val_frac), replace=False))
    is_val = df.inchikey14.isin(val_keys)
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)
