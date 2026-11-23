"""Tests for Model 2 (LightGBM re-ranker): the features, the training and the scorer.

Run from the repo root:
    python -m pytest members/Mrnidhi/tests/test_ranker.py -q
"""
import numpy as np
import pandas as pd
import pytest

from members.Mrnidhi import train_ranker
from members.Mrnidhi.model_entropy import EntropyScorer, Peaks, prepare_spectrum
from members.Mrnidhi.model_ranker import (
    FEATURES,
    MODEL_PATH,
    RankerFeatures,
    RankerScorer,
    formula_features,
    group_features,
    matched_peaks,
    parse_formula,
    rdbe,
)
from members.Mrnidhi.train_ranker import TABLE_PATH, fit, mrr


def peaks(pairs: list) -> Peaks:
    """(m/z, raw intensity) pairs -> Peaks, through the same cleaning the model uses."""
    mz, raw = zip(*pairs)
    return prepare_spectrum(np.array(mz, np.float32), np.sqrt(np.array(raw, dtype=float)), weighted=False)


# ---- formula features ------------------------------------------------------------------

@pytest.mark.parametrize("formula, expected", [
    ("C6H12O6", {"C": 6, "H": 12, "O": 6}),
    ("CH4", {"C": 1, "H": 4}),                           # no number means one atom
    ("C5H8ClN5", {"C": 5, "H": 8, "Cl": 1, "N": 5}),     # Cl is chlorine, not carbon
    (None, {}),
    (float("nan"), {}),
])
def test_parse_formula(formula: object, expected: dict) -> None:
    assert parse_formula(formula) == expected


@pytest.mark.parametrize("formula, expected", [
    ("C6H6", 4),            # benzene: one ring + three double bonds
    ("C6H12O6", 1),         # glucose: one ring
    ("C8H10N4O2", 6),       # caffeine
    ("C5H8ClN5", 4),        # a halogen counts like a hydrogen
])
def test_rdbe(formula: str, expected: float) -> None:
    assert rdbe(parse_formula(formula)) == expected


def test_formula_features_of_one_formula() -> None:
    row = formula_features(pd.Series(["C15H16N2O4S"])).iloc[0]
    assert row.to_dict() == {"n_C": 15, "n_H": 16, "n_N": 2, "n_O": 4, "n_S": 1, "n_P": 0,
                             "has_N": 1, "rdbe": 9}


def test_formula_features_of_a_missing_formula_are_nan() -> None:
    table = formula_features(pd.Series(["C2H6O", None]))
    assert table.iloc[0].n_C == 2 and table.iloc[0].has_N == 0
    assert table.iloc[1].isna().all()


# ---- features that compare a candidate with the other candidates -----------------------

def test_group_features() -> None:
    candidates = pd.DataFrame({
        "cosine": [0.2, 0.9, 0.9, 0.1],
        "n_spectra": [0, 1, 9, 99],
        "formula": ["C2H6O", "C2H6O", "CH4", "C2H6O"],
    })
    table = group_features(candidates)
    assert table.cosine_rank.tolist() == [3, 1, 1, 4]            # 1 = best, ties share a rank
    assert table.n_candidates.tolist() == [4, 4, 4, 4]
    assert table.n_isomers.tolist() == [3, 3, 1, 3]              # candidates with the same formula
    assert table.log_n_spectra.tolist() == pytest.approx(np.log1p([0, 1, 9, 99]))


# ---- matched peaks ----------------------------------------------------------------------

def test_matched_peaks_counts_shared_peaks() -> None:
    a = peaks([(100, 0.5), (150, 0.5)])
    assert matched_peaks(a, a) == 2
    assert matched_peaks(a, peaks([(100, 0.5), (200, 0.5)])) == 1
    assert matched_peaks(a, peaks([(300, 0.5), (400, 0.5)])) == 0


def test_matched_peaks_uses_the_tolerance() -> None:
    a = peaks([(100.000, 1)])
    assert matched_peaks(a, peaks([(100.015, 1)])) == 1          # inside 0.02 Da
    assert matched_peaks(a, peaks([(100.030, 1)])) == 0          # outside


# ---- on the real data -------------------------------------------------------------------

@pytest.fixture(scope="module")
def builder(needs_data: None) -> RankerFeatures:
    return RankerFeatures()


@pytest.fixture(scope="module")
def validation(needs_data: None) -> tuple:
    from src.evaluation.validate import load_validation
    return load_validation()


@pytest.mark.integration
def test_features_have_one_clean_row_per_candidate(builder: RankerFeatures, validation: tuple) -> None:
    queries, cands = validation
    for molecule in list(queries)[:5]:
        table = builder.features(queries[molecule], cands[molecule])
        assert list(table.columns) == FEATURES
        assert len(table) == len(cands[molecule])
        assert not table.isna().any().any()
        assert (table.matched_peaks[table.entropy == 0] == 0).all()
        assert (table.matched_peaks[table.entropy > 0] > 0).all()


@pytest.mark.integration
def test_entropy_feature_is_the_model_1_score(builder: RankerFeatures, validation: tuple) -> None:
    queries, cands = validation
    model_1 = EntropyScorer()
    for molecule in list(queries)[:5]:
        table = builder.features(queries[molecule], cands[molecule])
        assert table.entropy.values == pytest.approx(model_1.score(queries[molecule], cands[molecule]))


@pytest.mark.integration
def test_features_handle_odd_inputs(builder: RankerFeatures, validation: tuple) -> None:
    queries, cands = validation
    molecule = next(iter(queries))
    spectra, table = queries[molecule], cands[molecule]

    assert builder.features(spectra, table.iloc[:0]).shape == (0, len(FEATURES))

    no_spectra = builder.features([], table)
    assert (no_spectra.entropy == 0).all() and (no_spectra.matched_peaks == 0).all()

    outsider = pd.DataFrame([{"struct_idx": -1, "smiles": "CCO", "inchikey14": "LFQSCWFLJHTTHZ",
                              "formula": None, "mass": np.nan, "n_spectra": 0, "cosine": 0.0}])
    row = builder.features(spectra, pd.concat([table, outsider], ignore_index=True)).iloc[-1]
    assert row.entropy == 0 and row.matched_peaks == 0 and np.isnan(row.n_C)


# ---- training ---------------------------------------------------------------------------

def toy_table(n_molecules: int = 60, n_candidates: int = 5) -> pd.DataFrame:
    """Made-up molecules where the right answer always has the highest entropy score."""
    rng = np.random.default_rng(0)
    rows = []
    for m in range(n_molecules):
        for c in range(n_candidates):
            row = dict.fromkeys(FEATURES, 0.0)
            row.update(entropy=0.9 if c == 0 else rng.uniform(0, 0.5), cosine=rng.uniform(0, 1),
                       label=int(c == 0), query=f"toy:{m:03d}", lib="toy")
            rows.append(row)
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def test_mrr_of_a_score_column() -> None:
    table = pd.DataFrame({
        "query": ["a", "a", "b", "b", "b"],
        "label": [1, 0, 0, 1, 0],
        "log_n_spectra": 0.0,
    })
    scores = np.array([0.9, 0.1, 0.8, 0.5, 0.2])       # a: truth first, b: truth second
    assert mrr(table, scores) == pytest.approx((1 + 1 / 2) / 2)


def test_mrr_gives_nothing_beyond_rank_25() -> None:
    table = pd.DataFrame({"query": "a", "label": [0] * 25 + [1], "log_n_spectra": 0.0})
    assert mrr(table, -np.arange(26.0)) == 0.0


def test_ranker_learns_an_obvious_rule() -> None:
    table = toy_table()
    ranker = fit(table)
    assert mrr(table, table.cosine.to_numpy()) < 0.7            # the noise column cannot rank
    assert mrr(table, ranker.predict(table[FEATURES])) == pytest.approx(1.0)


def test_training_is_repeatable() -> None:
    table = toy_table()
    first, second = fit(table).predict(table[FEATURES]), fit(table).predict(table[FEATURES])
    assert np.array_equal(first, second)


@pytest.mark.integration
def test_training_queries_drop_spectra_over_the_ppm_limit(needs_data: None, monkeypatch: pytest.MonkeyPatch) -> None:
    kept = train_ranker.library_queries("massbank", n_molecules=20)
    assert 0 < len(kept) <= 20
    monkeypatch.setattr(train_ranker, "MAX_PPM", -1.0)          # no spectrum can pass this limit
    assert train_ranker.library_queries("massbank", n_molecules=20) == {}


# ---- the trained model, on the real data ------------------------------------------------

@pytest.fixture(scope="module")
def trained(needs_data: None) -> RankerScorer:
    if not MODEL_PATH.exists() or not TABLE_PATH.exists():
        pytest.skip("no trained ranker; run python -m members.Mrnidhi.train_ranker first")
    return RankerScorer()


@pytest.mark.integration
def test_training_table_has_no_validation_molecules(trained: RankerScorer) -> None:
    from src.data.dataset import validation_keys
    table = pd.read_parquet(TABLE_PATH, columns=["query", "label"])
    assert not set(table["query"].str.split(":").str[1]) & validation_keys()
    assert (table.groupby("query").label.sum() == 1).all()       # one right answer per molecule


@pytest.mark.integration
def test_ranker_gives_one_score_per_candidate(trained: RankerScorer, validation: tuple) -> None:
    queries, cands = validation
    molecule = next(iter(queries))
    spectra, table = queries[molecule], cands[molecule]

    scores = trained.score(spectra, table)
    assert scores.shape == (len(table),) and np.isfinite(scores).all()
    assert trained.score(spectra, table.iloc[:0]).shape == (0,)
    assert np.isfinite(trained.score([], table)).all()

    outsider = pd.DataFrame([{"struct_idx": -1, "smiles": "CCO", "inchikey14": "LFQSCWFLJHTTHZ",
                              "formula": None, "mass": np.nan, "n_spectra": 0, "cosine": 0.0}])
    with_outsider = trained.score(spectra, pd.concat([table, outsider], ignore_index=True))
    assert len(with_outsider) == len(table) + 1 and np.isfinite(with_outsider).all()


@pytest.mark.integration
def test_validation_score_is_the_logged_one(trained: RankerScorer) -> None:
    """E502 in experiments/experiment_log.md, on the top-64 reference library of main."""
    from src.evaluation.validate import evaluate
    if trained.lib.plen.max() <= 32:
        pytest.skip("the logged E502 score is for the top-64 library; retrain after the library changes")
    assert evaluate(trained, verbose=False)["mrr"] == pytest.approx(0.9107, abs=2e-3)


@pytest.mark.integration
def test_submission_file_is_valid(trained: RankerScorer, tmp_path) -> None:
    from src.evaluation.validate import make_submission
    model = RankerScorer(exclude_libs=())
    sub = make_submission(model, "data/raw/test.parquet", "data/processed", tmp_path / "submission.csv")
    sample = pd.read_csv("data/raw/sample_submission.csv")
    guesses = sub.smiles.str.split(";").map(len)
    assert set(sub.molecule_id) == set(sample.molecule_id) and sub.molecule_id.is_unique
    assert not sub.isna().any().any() and guesses.min() >= 1 and guesses.max() <= 25
