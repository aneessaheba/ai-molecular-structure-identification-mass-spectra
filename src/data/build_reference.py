"""Build a compact spectral reference library from train.parquet.

Outputs (in --out, default data/processed/):
  ref_meta.parquet   one row per kept spectrum: struct_idx, ion_mode, adduct, lib, peak offsets
  ref_peaks.npz      concatenated cleaned peaks (mz, intensity) for all kept spectra
  structures.parquet one row per inchikey14: smiles, formula, neutral mass, n_spectra

Usage:  python -m src.data.build_reference --train data/raw/train.parquet --out data/processed
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.spectra import clean_peaks, formula_mass

COLS = ["ingest_lib", "normalized_smiles", "inchikey14", "molecular_formula", "ionization_mode",
        "adduct", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]


def build(train_path, out_dir, top_n=64, min_rel=0.01, min_peaks=3):
    os.makedirs(out_dir, exist_ok=True)
    pf = pq.ParquetFile(train_path)
    metas, mz_chunks, int_chunks = [], [], []
    offset, t0 = 0, time.time()
    for rg in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(rg, columns=COLS)
        mz_list = tbl.column("ms2_mzs").to_pylist()
        in_list = tbl.column("ms2_normalized_intensities").to_pylist()
        df = tbl.drop(["ms2_mzs", "ms2_normalized_intensities"]).to_pandas()
        starts, lens = np.zeros(len(df), np.int64), np.zeros(len(df), np.int32)
        for i, (mz, it, pmz) in enumerate(zip(mz_list, in_list, df.precursor_mz.values)):
            m, s = clean_peaks(mz or [], it or [], pmz, top_n=top_n, min_rel=min_rel)
            starts[i], lens[i] = offset, m.size
            if m.size:
                mz_chunks.append(m)
                int_chunks.append(s)
                offset += m.size
        df["peak_start"], df["n_peaks"] = starts, lens
        metas.append(df[df.n_peaks >= min_peaks])
        print(f"row group {rg + 1}/{pf.metadata.num_row_groups}  {time.time() - t0:.0f}s", flush=True)
        del tbl, mz_list, in_list

    meta = pd.concat(metas, ignore_index=True)
    mz_all = np.concatenate(mz_chunks)
    int_all = np.concatenate(int_chunks)

    structs = (meta.groupby("inchikey14")
               .agg(smiles=("normalized_smiles", "first"),
                    formula=("molecular_formula", "first"),
                    n_spectra=("normalized_smiles", "size"),
                    n_libs=("ingest_lib", "nunique"))
               .reset_index())
    structs["mass"] = structs.formula.map(formula_mass)
    structs = structs.dropna(subset=["mass"]).sort_values("mass").reset_index(drop=True)
    idx = pd.Series(np.arange(len(structs)), index=structs.inchikey14)
    meta["struct_idx"] = meta.inchikey14.map(idx)
    meta = meta.dropna(subset=["struct_idx"])
    meta["struct_idx"] = meta.struct_idx.astype(np.int32)
    meta = meta[["struct_idx", "inchikey14", "ingest_lib", "ionization_mode", "adduct",
                 "precursor_mz", "peak_start", "n_peaks"]]

    meta.to_parquet(os.path.join(out_dir, "ref_meta.parquet"), index=False)
    structs.to_parquet(os.path.join(out_dir, "structures.parquet"), index=False)
    np.savez(os.path.join(out_dir, "ref_peaks.npz"), mz=mz_all, intensity=int_all)
    print(f"done: {len(meta):,} spectra, {len(structs):,} structures, {mz_all.size:,} peaks, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/raw/train.parquet")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--top_n", type=int, default=64)
    args = ap.parse_args()
    build(args.train, args.out, top_n=args.top_n)
