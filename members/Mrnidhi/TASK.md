# Srinidhi: Entropy Similarity + Learning-to-Rank (LightGBM)

**Kaggle team:** GROUP_PROGRAMMING_01_a · **Experiment IDs:** E500–E599
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Two classic-ML models that improve how candidates are ranked, with no deep learning needed:
1. A better **spectral similarity** than plain cosine: **spectral entropy similarity** (Li et al., *Nature Methods* 2021), which beat 42 other similarity measures in that paper.
2. A **learning-to-rank** model (LightGBM LambdaRank) that combines many signals (several similarities, how common the structure is, formula features) into one score, the same way search engines rank results.

---

## Model 1: Entropy similarity scorer (due Thu Nov 19)
Same as the baseline, but replace cosine with entropy similarity.
- Implementation: `pip install ms_entropy`, then `ms_entropy.calculate_entropy_similarity(peaks_a, peaks_b, ms2_tolerance_in_da=0.02)`, where peaks are `np.array([[mz, intensity], ...], dtype=np.float32)`. You can also implement it yourself from the paper (about 20 lines).
- Use raw-like intensities: our cleaned intensities are sqrt-scaled, so pass `intensity ** 2`.

You need each candidate's **reference spectra**:
```python
from src.models.library_search import ReferenceLibrary
lib = ReferenceLibrary("data/processed", exclude_libs=["enveda-np-examples"])  # validation
# lib = ReferenceLibrary("data/processed")                                    # Kaggle
for idx in candidates.struct_idx:
    for j in lib.spectra_of(idx):                 # that structure's reference spectra
        if lib.mode[j] != spectrum["ionization_mode"]: continue
        mz, intensity = lib.peaks(j)
```
Score = mean over the molecule's spectra of the max similarity over the candidate's reference spectra (same as the baseline). Validate → E501. Did entropy beat cosine (0.929)?

## Model 2: LightGBM LambdaRank re-ranker (due Sun Nov 22)
**Features per candidate** (build a function `features(spectra, candidates) → DataFrame`):
- `cosine` (already in `candidates`), your entropy similarity, the number of matched peaks
- `log1p(n_spectra)`, number of candidates for this molecule, rank of the candidate by cosine
- formula features: counts of C, H, N, O, S, P (parse `candidates.formula`), whether it has N, a rings-plus-double-bonds estimate
- number of same-formula candidates (isomer count)

**Training data, without touching the validation set.** Use "leave-one-library-out": query molecules from one library, with that library removed from the reference:
```python
import pyarrow.parquet as pq
from src.models.library_search import ReferenceLibrary, prepare_queries, candidate_table
from src.data.dataset import validation_keys
q = pq.read_table("data/raw/train.parquet", filters=[("ingest_lib", "=", "msdial")]).to_pandas()
q = q[~q.inchikey14.isin(validation_keys())]
q = q[q.inchikey14.isin(q.inchikey14.drop_duplicates().sample(3000, random_state=0))]  # ~3k molecules
q["molecule_id"] = q.inchikey14
queries = prepare_queries(q)
lib = ReferenceLibrary("data/processed", exclude_libs=["enveda-np-examples", "msdial"])
# for each molecule: cands = candidate_table(lib, spectra); label = (cands.inchikey14 == molecule)
# keep only molecules where the truth is among the candidates
```
Repeat for 2–3 libraries (e.g. `msdial`, `massbank`, `mona`) to get several thousand training queries.
Train `lightgbm.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.05)` with `group=` = candidates per molecule.
The scorer returns `ranker.predict(features(spectra, candidates))`. Validate → E502. Also plot **feature importance** (a great report figure).

## Model 3 (optional, Tue Nov 24)
Add more features (e.g. modified cosine from `matchms`, peak-count ratio) and/or tune LightGBM. → E503.
If teammates' models are ready, their scores can become features too (coordinate with Anees).

---

## What to expect
Entropy similarity should be close to or above the baseline cosine. The LightGBM ranker is the most likely single model to beat 0.929. Watch for **leakage**: never use `enveda-np-examples` molecules in training.

## Kaggle notes
Save the ranker with `ranker.booster_.save_model("ranker.txt")` and upload it as `casmi-srinidhi-weights`. If `ms_entropy` or `lightgbm` isn't installed on Kaggle, upload its wheel (`pip download ms_entropy -d wheels/`) in the same dataset and install with `--no-index --find-links`.

## Checklist
- [ ] Setup done, baseline 0.929 (Nov 15)
- [ ] Model 1 (entropy) validated, logged (E501), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 (LightGBM ranker) validated, logged (E502), submitted (Nov 22)
- [ ] (Optional) Model 3 (E503)
- [ ] Hand-off README + feature-importance figure + screenshots, PR opened (Nov 25, 9 PM)
