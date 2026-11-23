"""Model 2, steps 2 and 3: build the training table and train the LightGBM ranker.

Training data comes from "leave one library out", so the 250 validation molecules are never
used. For each training library (msdial, massbank, mona):

  1. take molecules of that library as the unknowns (validation molecules removed, and
     spectra with a precursor error above 20 ppm dropped, as the shared loader does),
  2. hide that library and the validation library from the reference,
  3. build each molecule's candidates and their features,
  4. label the true structure 1 and every other candidate 0.

A molecule is kept only if its true structure is among the candidates, otherwise there is
nothing to learn from it.

Run from the repo root (about 10 minutes the first time, then the table is cached):
    python -m members.Mrnidhi.train_ranker
"""
import argparse
import gc
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from members.Mrnidhi.model_entropy import _LIBS, VAL_LIB
from members.Mrnidhi.model_ranker import FEATURES, RankerFeatures
from src.data.dataset import TEST_ADDUCTS, TRAIN_PATH, validation_keys
from src.models.library_search import candidate_table, prepare_queries

SEED = 0
TRAIN_LIBS = ("msdial", "massbank", "mona")
MOLECULES_PER_LIB = 3000
MAX_QUERY_SPECTRA = 8        # validation molecules have 4 to 5 spectra on average, the test set 3
MAX_PPM = 20.0               # same limit as src.data.dataset.load_training: beyond it, likely mislabelled
K = 25
WEIGHTS = Path(__file__).parent / "weights"
TABLE_PATH = WEIGHTS / "ranker_train.parquet"
MODEL_PATH = WEIGHTS / "ranker.txt"
QUERY_COLUMNS = ["inchikey14", "adduct", "ionization_mode", "precursor_mz", "precursor_error_ppm",
                 "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
PARAMS = {"objective": "lambdarank", "n_estimators": 300, "learning_rate": 0.05,
          "random_state": SEED, "deterministic": True, "force_row_wise": True, "n_jobs": 4,
          "verbose": -1}


def library_queries(lib_name: str, n_molecules: int = MOLECULES_PER_LIB) -> dict:
    """{molecule: [spectrum dicts]} for a seeded sample of one library's molecules."""
    rows = pq.read_table(TRAIN_PATH, columns=QUERY_COLUMNS,
                         filters=[("ingest_lib", "=", lib_name)]).to_pandas()
    rows = rows[~rows.inchikey14.isin(validation_keys()) & rows.adduct.isin(TEST_ADDUCTS)]
    molecules = rows.inchikey14.drop_duplicates()
    chosen = molecules.sample(min(n_molecules, len(molecules)), random_state=SEED)
    rows = rows[rows.inchikey14.isin(chosen)].sample(frac=1, random_state=SEED)
    rows = rows[rows.precursor_error_ppm.abs().fillna(0) <= MAX_PPM]     # drop likely mislabelled spectra
    rows = rows.groupby("inchikey14").head(MAX_QUERY_SPECTRA)
    rows = rows.sort_index().assign(molecule_id=lambda d: d.inchikey14)
    return prepare_queries(rows)


def library_table(lib_name: str, n_molecules: int = MOLECULES_PER_LIB) -> pd.DataFrame:
    """Feature rows for one library's molecules, with that library hidden from the reference."""
    queries = library_queries(lib_name, n_molecules)
    builder = RankerFeatures(exclude_libs=(VAL_LIB, lib_name))
    parts = []
    for molecule, spectra in queries.items():
        candidates = candidate_table(builder.lib, spectra)
        is_truth = (candidates.inchikey14 == molecule).to_numpy()
        if len(candidates) < 2 or not is_truth.any():       # nothing to rank, or no right answer
            continue
        rows = builder.features(spectra, candidates)
        parts.append(rows.assign(label=is_truth.astype(int), query=f"{lib_name}:{molecule}", lib=lib_name))
    _LIBS.clear()            # each library copy is large; keep one in memory at a time
    gc.collect()
    return pd.concat(parts, ignore_index=True)


def build_table(n_molecules: int = MOLECULES_PER_LIB) -> pd.DataFrame:
    """The whole training table: one row per candidate, grouped by `query`."""
    parts = []
    for lib_name in TRAIN_LIBS:
        started = time.time()
        part = library_table(lib_name, n_molecules)
        print(f"{lib_name}: {part['query'].nunique()} molecules, {len(part)} candidates, "
              f"{time.time() - started:.0f}s", flush=True)
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    leaked = set(table["query"].str.split(":").str[1]) & validation_keys()
    assert not leaked, f"validation molecules in the training table: {sorted(leaked)[:3]}"
    return table


def fit(table: pd.DataFrame) -> lgb.LGBMRanker:
    """Train the ranker. Rows of one molecule must sit together, so the table is sorted by query."""
    table = table.sort_values("query", kind="stable")
    ranker = lgb.LGBMRanker(**PARAMS)
    ranker.fit(table[FEATURES], table.label, group=table.groupby("query", sort=False).size().to_numpy())
    return ranker


def mrr(table: pd.DataFrame, scores: np.ndarray) -> float:
    """MRR@25 of a score column over the molecules of a feature table (ties: more library spectra)."""
    ranked = table.assign(score=scores).sort_values(["score", "log_n_spectra"], ascending=False, kind="stable")
    position = ranked.groupby("query", sort=False).cumcount() + 1
    hits = position[(ranked.label == 1) & (position <= K)]
    return float((1 / hits).sum() / table["query"].nunique())


def cross_check(table: pd.DataFrame) -> pd.DataFrame:
    """Train on two libraries and test on the third, for each library. No validation data involved."""
    rows = []
    for held_out in TRAIN_LIBS:
        train, test = table[table.lib != held_out], table[table.lib == held_out]
        ranker = fit(train)
        rows.append({"held_out": held_out, "molecules": test["query"].nunique(),
                     "cosine": mrr(test, test.cosine.to_numpy()),
                     "entropy": mrr(test, test.entropy.to_numpy()),
                     "ranker": mrr(test, ranker.predict(test[FEATURES]))})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rebuild", action="store_true", help="rebuild the cached training table")
    parser.add_argument("--molecules", type=int, default=MOLECULES_PER_LIB, help="molecules per library")
    args = parser.parse_args()

    WEIGHTS.mkdir(exist_ok=True)
    if args.rebuild or not TABLE_PATH.exists():
        build_table(args.molecules).to_parquet(TABLE_PATH, index=False)
    table = pd.read_parquet(TABLE_PATH)
    print(f"training table: {table['query'].nunique()} molecules, {len(table)} candidates, "
          f"{table.label.mean():.1%} of rows are the right answer")
    print("leave-one-library-out check (MRR@25):")
    print(cross_check(table).round(4).to_string(index=False))

    ranker = fit(table)
    ranker.booster_.save_model(str(MODEL_PATH))
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
