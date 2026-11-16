"""Tests for Model 1 (entropy similarity).

Run from the repo root:
    python -m pytest members/Mrnidhi/tests -q
"""
import numpy as np
import pandas as pd
import pytest

from members.Mrnidhi.model_entropy import (
    VAL_LIB,
    EntropyScorer,
    Peaks,
    entropy_similarity,
    entropy_weight,
    merge_close_peaks,
    prepare_spectrum,
)

# Three small spectra as (m/z, raw intensity). A and B contain peaks closer than 0.05 Da.
SPEC_A = [(85.03, 12), (100.00, 40), (100.04, 100), (100.08, 60), (150.02, 30), (150.06, 5),
          (221.10, 80), (300.15, 55)]
SPEC_B = [(85.04, 20), (100.05, 90), (150.03, 45), (190.20, 10), (221.11, 70), (221.115, 30),
          (300.20, 60)]
SPEC_C = [(60.01, 5), (100.06, 100), (221.09, 4), (410.30, 1)]


def team_format(pairs: list) -> tuple:
    """(m/z, raw intensity) pairs -> float32 m/z and sqrt-scaled intensity, as the team stores them."""
    mz, raw = zip(*pairs)
    return np.array(mz, np.float32), np.sqrt(np.array(raw, dtype=float)).astype(np.float32)


def peaks(pairs: list, weighted: bool = False) -> Peaks:
    return prepare_spectrum(*team_format(pairs), weighted=weighted)


# ---- the similarity itself -------------------------------------------------------------

def test_identical_spectra_score_one() -> None:
    a = peaks([(100, 0.5), (150, 0.5)])
    assert entropy_similarity(a, a) == pytest.approx(1.0)


def test_one_of_two_shared_peaks_scores_half() -> None:
    a = peaks([(100, 0.5), (150, 0.5)])
    b = peaks([(100, 0.5), (200, 0.5)])
    assert entropy_similarity(a, b) == pytest.approx(0.5)


def test_no_shared_peaks_scores_zero() -> None:
    a = peaks([(100, 0.5), (150, 0.5)])
    b = peaks([(300, 0.5), (400, 0.5)])
    assert entropy_similarity(a, b) == 0.0


def test_similarity_is_symmetric() -> None:
    a, b = peaks(SPEC_A, weighted=True), peaks(SPEC_B, weighted=True)
    assert entropy_similarity(a, b) == pytest.approx(entropy_similarity(b, a))


def test_peaks_match_only_within_tolerance() -> None:
    a = peaks([(100.000, 1.0)])
    assert entropy_similarity(a, peaks([(100.015, 1.0)]), tol=0.02) == pytest.approx(1.0)
    assert entropy_similarity(a, peaks([(100.030, 1.0)]), tol=0.02) == 0.0


def test_empty_spectrum_scores_zero() -> None:
    empty = prepare_spectrum(np.empty(0), np.empty(0))
    assert empty == Peaks([], [], [])
    assert entropy_similarity(empty, peaks([(100, 1.0)])) == 0.0


@pytest.mark.parametrize("first, second, weighted, expected", [
    (SPEC_A, SPEC_B, True, 0.773635), (SPEC_A, SPEC_B, False, 0.778634),
    (SPEC_A, SPEC_C, True, 0.641812), (SPEC_A, SPEC_C, False, 0.752753),
    (SPEC_B, SPEC_C, True, 0.348301), (SPEC_B, SPEC_C, False, 0.464945),
])
def test_known_values(first: list, second: list, weighted: bool, expected: float) -> None:
    """Values computed once with the ms_entropy reference code, so this runs without the package."""
    assert entropy_similarity(peaks(first, weighted), peaks(second, weighted)) == pytest.approx(expected, abs=1e-5)


# ---- cleaning before the comparison ----------------------------------------------------

def test_intensities_are_squared_back_and_sum_to_one() -> None:
    p = prepare_spectrum(np.array([100.0, 200.0]), np.array([1.0, 2.0]), weighted=False)
    assert p.intensity == pytest.approx([0.2, 0.8])      # 1 and 4, divided by 5


def test_non_positive_intensities_are_dropped() -> None:
    p = prepare_spectrum(np.array([100.0, 150.0, 200.0]), np.array([1.0, -1.0, 0.0]), weighted=False)
    assert p.mz == pytest.approx([100.0])


def test_close_peaks_are_merged() -> None:
    mz, intensity = merge_close_peaks(np.array([100.00, 100.03, 200.0]), np.array([3.0, 1.0, 2.0]))
    assert mz == pytest.approx([100.0075, 200.0], abs=1e-3)   # intensity-weighted mean m/z
    assert intensity == pytest.approx([4.0, 2.0])


def test_merging_starts_from_the_strongest_peak() -> None:
    # The middle peak is strongest and both neighbours are within 0.05 of it: one peak.
    # Merging left to right would pair the first two (100.029) and leave 100.08 on its own.
    mz, intensity = merge_close_peaks([100.00, 100.04, 100.08], [40.0, 100.0, 60.0])
    assert mz == pytest.approx([100.044], abs=1e-3)
    assert intensity == pytest.approx([200.0])


def test_merging_repeats_until_no_gap_is_under_the_window() -> None:
    # Pass 1: 100.000 takes 100.049 and becomes 100.016; 100.055 is left alone.
    # Those two are now 0.039 apart, so a second pass has to merge them.
    mz, intensity = merge_close_peaks([100.000, 100.049, 100.055], [10.0, 5.0, 9.0])
    assert mz == pytest.approx([100.0308], abs=1e-3)
    assert intensity == pytest.approx([24.0])


def test_peaks_further_apart_than_the_window_are_kept() -> None:
    mz, _ = merge_close_peaks([100.00, 100.06], [1.0, 1.0])
    assert len(mz) == 2


def test_peaks_below_the_noise_floor_are_dropped() -> None:
    p = peaks([(100, 1.0), (200, 0.005)])
    assert p.mz == pytest.approx([100.0])


def test_weighting_lifts_small_peaks_of_a_simple_spectrum() -> None:
    simple = np.array([0.9, 0.1])
    assert entropy_weight(simple)[1] > 0.1
    assert entropy_weight(simple).sum() == pytest.approx(1.0)


def test_weighting_leaves_a_complex_spectrum_alone() -> None:
    flat = np.full(30, 1 / 30)          # entropy = ln(30) = 3.4, above the threshold of 3
    assert entropy_weight(flat) == pytest.approx(flat)


# ---- against the ms_entropy reference code ---------------------------------------------
# ms_entropy ships two builds. These tests use its plain-Python code (tools.py, entropy.py),
# which this port follows line by line. The compiled build merges dense peak clusters a
# little differently on about 0.3% of our library spectra.

def random_spectrum(rng: np.random.Generator, kind: str) -> tuple:
    """Float32 (m/z, sqrt intensity). 'dense' adds runs of peaks a few mDa apart, like
    un-centroided library spectra; 'equal' gives every peak the same intensity (all ties)."""
    mz = rng.uniform(50, 500, int(rng.integers(1, 40)))
    if kind != "sparse":
        centres = rng.uniform(50, 500, int(rng.integers(1, 5)))
        mz = np.concatenate([mz, *(c + np.cumsum(rng.uniform(0.002, 0.03, int(rng.integers(5, 40))))
                                   for c in centres)])
    raw = np.ones(mz.size) if kind == "equal" else rng.uniform(0.001, 1, mz.size)
    return np.sort(mz).astype(np.float32), np.sqrt(raw).astype(np.float32)


def reference_clean(mz: np.ndarray, root: np.ndarray) -> np.ndarray:
    """The package's cleaning on exactly the numbers prepare_spectrum works with."""
    tools = pytest.importorskip("ms_entropy.spectra.tools")
    return tools.clean_spectrum(np.column_stack([mz, root ** 2]))


@pytest.mark.parametrize("kind", ["sparse", "dense", "equal"])
def test_cleaning_matches_the_reference_code(kind: str) -> None:
    rng = np.random.default_rng(1)
    merged = 0
    for _ in range(300):
        mz, root = random_spectrum(rng, kind)
        mine = prepare_spectrum(mz, root, weighted=False)
        theirs = reference_clean(mz, root)
        assert len(mine.mz) == len(theirs)
        assert mine.mz == pytest.approx(theirs[:, 0].tolist(), abs=1e-4)
        assert mine.intensity == pytest.approx(theirs[:, 1].tolist(), abs=1e-6)
        merged += len(theirs) < mz.size
    if kind != "sparse":
        assert merged > 250         # the dense cases must really go through the merging


@pytest.mark.parametrize("weighted", [True, False])
@pytest.mark.parametrize("kind", ["sparse", "dense"])
def test_similarity_matches_the_reference_code(kind: str, weighted: bool) -> None:
    reference = pytest.importorskip("ms_entropy.spectra.entropy")
    theirs_fn = (reference.calculate_entropy_similarity if weighted
                 else reference.calculate_unweighted_entropy_similarity)
    rng = np.random.default_rng(0)
    clearly_similar = 0
    for _ in range(300):
        mz_a, root_a = random_spectrum(rng, kind)
        shared = rng.random(mz_a.size) < 0.6            # b shares some of a's peaks, slightly shifted
        mz_b = np.sort(np.concatenate([mz_a[shared] + rng.normal(0, 0.008, int(shared.sum())),
                                       rng.uniform(50, 500, int(rng.integers(1, 20)))])).astype(np.float32)
        root_b = np.sqrt(rng.uniform(0.001, 1, mz_b.size)).astype(np.float32)

        mine = entropy_similarity(prepare_spectrum(mz_a, root_a, weighted),
                                  prepare_spectrum(mz_b, root_b, weighted))
        theirs = float(theirs_fn(reference_clean(mz_a, root_a), reference_clean(mz_b, root_b),
                                 ms2_tolerance_in_da=0.02, clean_spectra=False))
        assert mine == pytest.approx(theirs, abs=1e-5)
        clearly_similar += theirs > 0.05
    assert clearly_similar > 150    # must not pass just because every similarity is zero


# ---- the scorer on real data -----------------------------------------------------------

@pytest.fixture(scope="module")
def validation(needs_data: None) -> tuple:
    from src.evaluation.validate import load_validation
    return load_validation()


@pytest.fixture(scope="module")
def scorer(needs_data: None) -> EntropyScorer:
    return EntropyScorer()


@pytest.mark.integration
def test_validation_library_has_no_validation_spectra(scorer: EntropyScorer) -> None:
    assert VAL_LIB not in set(scorer.lib.meta.ingest_lib.unique())


@pytest.mark.integration
def test_a_string_for_exclude_libs_still_excludes_the_library(needs_data: None) -> None:
    lib = EntropyScorer(exclude_libs=VAL_LIB).lib
    assert VAL_LIB not in set(lib.meta.ingest_lib.unique())


@pytest.mark.integration
def test_full_library_is_used_when_nothing_is_excluded(scorer: EntropyScorer, needs_data: None) -> None:
    full = EntropyScorer(exclude_libs=()).lib
    assert VAL_LIB in set(full.meta.ingest_lib.unique())
    assert len(full.meta) > len(scorer.lib.meta)


@pytest.mark.integration
def test_scorer_handles_odd_inputs(scorer: EntropyScorer, validation: tuple) -> None:
    queries, cands = validation
    mol, spectra = next(iter(queries.items()))
    table = cands[mol]

    no_peaks = [dict(spectra[0], mz=np.empty(0, np.float32), intensity=np.empty(0, np.float32))]
    assert (scorer.score(no_peaks, table) == 0).all()

    assert (scorer.score([], table) == 0).all()
    assert len(scorer.score(spectra, table.iloc[:0])) == 0

    other_mode = [dict(q, ionization_mode="no-such-mode") for q in spectra]
    assert (scorer.score(other_mode, table) == 0).all()

    unknown = pd.DataFrame([{"struct_idx": -1, "smiles": "CCO", "inchikey14": "X", "formula": None,
                             "mass": np.nan, "n_spectra": 0, "cosine": 0.0}])
    assert scorer.score(spectra, pd.concat([table, unknown], ignore_index=True))[-1] == 0.0


@pytest.mark.integration
def test_validation_score_is_the_logged_one(scorer: EntropyScorer, needs_data: None) -> None:
    from src.evaluation.validate import evaluate
    result = evaluate(scorer, verbose=False)
    # 0.9294 on the top-64 library (E501); 0.9341 once the library is rebuilt with top-32 peaks.
    # Cosine on the same spectra scores 0.9245 / 0.9286, so this also catches a fall back to cosine,
    # and a leak of the validation spectra would push the score far above either number.
    tuned = int(scorer.lib.plen.max()) <= 32
    assert result["mrr"] == pytest.approx(0.9341 if tuned else 0.9294, abs=2e-3)
    assert result["top1"] == pytest.approx(0.900 if tuned else 0.892, abs=5e-3)


@pytest.mark.integration
def test_submission_file_is_valid(needs_data: None, tmp_path) -> None:
    from src.evaluation.validate import make_submission
    out = tmp_path / "submission.csv"
    make_submission(EntropyScorer(exclude_libs=()), "data/raw/test.parquet", "data/processed", out_path=str(out))
    sub = pd.read_csv(out)
    sample = pd.read_csv("data/raw/sample_submission.csv")
    assert list(sub.columns) == list(sample.columns)
    assert set(sub.molecule_id) == set(sample.molecule_id)
    assert not sub.molecule_id.duplicated().any()
    assert not sub.isna().any().any()
    assert sub.smiles.str.split(";").map(len).max() <= 25
