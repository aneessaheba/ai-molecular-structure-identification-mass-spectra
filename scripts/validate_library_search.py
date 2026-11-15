"""Local validation of the library-search baseline.

Validation set: `enveda-np-examples` (same timsTOF instrument and pipeline as the hidden test set).
Those spectra are removed from the reference library, then each molecule is queried.
Structures that appear only in the held-out library become unreachable, as with novel test molecules.

Usage: python -m scripts.validate_library_search [--no-cosine] [--ppm 15]
"""
import argparse
import time

import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metric import mrr_from_keys
from src.models.library_search import ReferenceLibrary, prepare_queries, query_molecule

VAL_LIB = "enveda-np-examples"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/processed")
    ap.add_argument("--train", default="data/raw/train.parquet")
    ap.add_argument("--ppm", type=float, default=15)
    ap.add_argument("--no-cosine", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    val = pq.read_table(args.train, filters=[("ingest_lib", "=", VAL_LIB)]).to_pandas()
    val = val[val.adduct.isin(["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
                               "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"])]
    val["molecule_id"] = val.inchikey14
    queries = prepare_queries(val)
    lib = ReferenceLibrary(args.ref, exclude_libs=[VAL_LIB])
    print(f"{len(val)} spectra, {len(queries)} molecules; loaded in {time.time() - t0:.0f}s")

    keys = lib.structs.inchikey14.values
    pred_keys, true_keys, reachable = [], [], 0
    for mol, spectra in queries.items():
        ranked = query_molecule(lib, spectra, ppm=args.ppm, use_cosine=not args.no_cosine)
        cand_keys = [keys[s] for s, _ in ranked]
        reachable += mol in cand_keys
        pred_keys.append(cand_keys[:25])
        true_keys.append(mol)

    n = len(true_keys)
    hit1 = sum(p[:1] == [t] for p, t in zip(pred_keys, true_keys)) / n
    hit25 = sum(t in p for p, t in zip(pred_keys, true_keys)) / n
    print(f"mode={'mass-only' if args.no_cosine else 'cosine'} ppm={args.ppm}")
    print(f"MRR@25 = {mrr_from_keys(pred_keys, true_keys):.4f}   top-1 = {hit1:.3f}   "
          f"top-25 = {hit25:.3f}   true structure in candidate set = {reachable / n:.3f}   "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
