"""Validate any Scorer on the team's shared validation set, and build Kaggle submissions.

Everyone uses the same protocol so our numbers are comparable in the report:
  * queries   = the 250 molecules of `enveda-np-examples` (same instrument as the hidden test)
  * reference = every other library (enveda-np-examples removed)
  * candidates = reference structures within ±15 ppm of the molecule's neutral mass
  * metric    = MRR@25 on InChIKey14 (same as Kaggle)

    from src.evaluation.validate import evaluate
    result = evaluate(MyModel())
    print(result["mrr"])

Command line (baseline check, should print MRR@25 ≈ 0.879):
    python -m src.evaluation.validate
"""
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.evaluation.metric import mrr_from_keys, smiles_to_ik14
from src.models.library_search import ReferenceLibrary, candidate_table, prepare_queries

VAL_LIB = "enveda-np-examples"
K = 25
_CACHE = {}


def _add_extra(cands, extra_smiles):
    if not extra_smiles:
        return cands
    seen = set(cands.inchikey14)
    rows = []
    for s in extra_smiles:
        key = smiles_to_ik14(s, tautomer=False)
        if key and key not in seen:
            seen.add(key)
            rows.append({"struct_idx": -1, "smiles": s, "inchikey14": key, "formula": None,
                         "mass": np.nan, "n_spectra": 0, "cosine": 0.0})
    return pd.concat([cands, pd.DataFrame(rows)], ignore_index=True) if rows else cands


def rank(scorer, spectra, cands):
    """Score candidates and return them sorted best first (ties → more library spectra)."""
    cands = _add_extra(cands, scorer.extra_candidates(spectra))
    if len(cands) == 0:
        return cands
    scores = np.asarray(scorer.score(spectra, cands), dtype=float)
    assert len(scores) == len(cands), f"{scorer.name}: got {len(scores)} scores for {len(cands)} candidates"
    cands = cands.assign(score=scores)
    return cands.sort_values(["score", "n_spectra"], ascending=False, kind="stable")


def load_validation(ref_dir="data/processed", train_path="data/raw/train.parquet", ppm=15):
    """Validation queries + their candidate tables (cached, ~5 s the first time)."""
    key = (ref_dir, ppm)
    if key not in _CACHE:
        val = pq.read_table(train_path, filters=[("ingest_lib", "=", VAL_LIB)]).to_pandas()
        val = val[val.adduct.isin(["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+",
                                   "[M+K]+", "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"])]
        val["molecule_id"] = val.inchikey14
        queries = prepare_queries(val)
        lib = ReferenceLibrary(ref_dir, exclude_libs=[VAL_LIB])
        cands = {m: candidate_table(lib, sp, ppm=ppm) for m, sp in queries.items()}
        _CACHE[key] = (queries, cands)
    return _CACHE[key]


def evaluate(scorer, ref_dir="data/processed", ppm=15, hide_true=False, verbose=True, limit=None):
    """Returns dict(mrr, top1, top10, top25, per_molecule DataFrame).

    hide_true=True removes the correct structure from the library candidates, which simulates
    Class 2/3 test molecules (no reference spectra). Then the only way to score is for the model's
    extra_candidates() to propose the right structure. Use this for candidate-expansion and de novo models.
    limit: evaluate only the first N molecules (quick debugging).
    """
    t0 = time.time()
    queries, cands = load_validation(ref_dir, ppm=ppm)
    rows = []
    for i, (mol, spectra) in enumerate(queries.items()):
        if limit and i >= limit:
            break
        c = cands[mol]
        if hide_true:
            c = c[c.inchikey14 != mol]
        ranked = rank(scorer, spectra, c)
        keys = list(ranked.inchikey14[:K])
        pos = keys.index(mol) + 1 if mol in keys else None
        rows.append({"molecule": mol, "rank": pos, "n_candidates": len(ranked),
                     "rr": 1.0 / pos if pos else 0.0})
    per = pd.DataFrame(rows)
    res = {
        "model": scorer.name,
        "mrr": per.rr.mean(),
        "top1": (per["rank"] == 1).mean(),
        "top10": (per["rank"] <= 10).mean(),
        "top25": per["rank"].notna().mean(),
        "per_molecule": per,
    }
    if verbose:
        tag = " (true structure hidden)" if hide_true else ""
        print(f"[{scorer.name}{tag}] MRR@25 = {res['mrr']:.4f} | top-1 {res['top1']:.3f} | "
              f"top-10 {res['top10']:.3f} | top-25 {res['top25']:.3f} | "
              f"{len(per)} molecules | {time.time() - t0:.0f}s")
    return res


def make_submission(scorer, test_path, ref_dir, out_path="submission.csv", ppm=15):
    """Run a scorer on a test-format parquet and write a valid submission.csv."""
    test = pd.read_parquet(test_path)
    lib = ReferenceLibrary(ref_dir)
    queries = prepare_queries(test)
    rows = []
    for mol in test.molecule_id.drop_duplicates():
        ranked = rank(scorer, queries[mol], candidate_table(lib, queries[mol], ppm=ppm))
        smiles = list(ranked.drop_duplicates("inchikey14").smiles[:K]) or ["CCO"]
        rows.append({"molecule_id": mol, "smiles": ";".join(smiles)})
    sub = pd.DataFrame(rows)
    sub.to_csv(out_path, index=False)
    return sub


if __name__ == "__main__":
    from src.models.base import CosineScorer, MassOnlyScorer
    evaluate(MassOnlyScorer())
    evaluate(CosineScorer())
