# Experiment Log (G.3)

Log **every** experiment, including failed ones. One row per run.

| ID | Date | Member | Model / Change | Why (hypothesis) | Local Val Score | Kaggle Score | Result / Takeaway | Code / Notebook |
|----|------|--------|----------------|------------------|-----------------|--------------|-------------------|-----------------|
| E001 | 2026-11-14 | aneessaheba | Baseline A: precursor-mass match (15 ppm) over training structures, ranked by library frequency | Simplest valid ranking; sets the floor | 0.8235 (val: enveda-np-examples held out, 250 mol) | | Top-1 73%, top-25 99%. Validation molecules are common compounds, so this val set is easy (Class 1 only) | `src/models/library_search.py --no-cosine` |
| E002 | 2026-11-14 | aneessaheba | Baseline B: E001 + greedy cosine (sqrt intensity, top-64 peaks, 1% floor, 0.01 Da) vs candidate reference spectra | Spectral similarity should separate isomers that share a mass | 0.8789 | | +0.055 MRR over E001; top-1 73% → 82%. First Kaggle submission | `src/models/library_search.py`, `kaggle/library_search_baseline/` |
| E500 | 2026-11-15 | Mrnidhi | E002 cosine, but compared against every reference spectrum in the same ionization mode (same change as Anees's E004, reached separately from an error study of E002's misses) | Error study of E002: 91% of the 46 misses were isomers, and the true structure often scored low or zero despite having 100+ reference spectra, because E002 takes a candidate's first 50 spectra and filters by mode afterwards | 0.9245 | | Confirms E004 (same 0.9245). Filtering by mode before the cut of 50 already gives 0.9065. Used as the fair comparison for E501 | `members/Mrnidhi/analyze_baseline_errors.py`, `members/Mrnidhi/model_entropy.py` (metric="cosine") |
| E501 | 2026-11-15 | Mrnidhi | Model 1: weighted spectral entropy similarity (ms_entropy, 0.02 Da) against all same-mode reference spectra; mean over query spectra of the best match | Entropy similarity gives small peaks more weight than cosine, which should help separate isomers | 0.9294 | | +0.053 over E002, but only +0.005 over E500, which is within noise for 250 molecules. Unweighted entropy: 0.9267. On the tuned pipeline of E006 (top-32 peaks) the same scorer gives 0.9341 against 0.9286 for cosine. Entropy is at least as good as cosine in both settings, not clearly better | `members/Mrnidhi/model_entropy.py` |
| E510 | 2026-11-15 | Mrnidhi | E501 with one setting changed at a time: tolerance 0.005 / 0.01 / 0.05 Da; combine query spectra by max / median / min; blend with E500 cosine at 25 / 50 / 75% | Check whether the paper's defaults are the best choice for this data | 0.9243 / 0.9221 / 0.9275; 0.9213 / 0.9250 / 0.8890; 0.9269 / 0.9315 / 0.9303 | | No clear gain (best is +0.002 from a 50/50 blend). Kept the defaults to avoid fitting to the validation set. Min is clearly worse: one poor spectrum should not sink a candidate | `members/Mrnidhi/entropy_variants.py` |
| E511 | 2026-11-15 | Mrnidhi | E501 plus a bonus (10% / 30%) when the reference spectrum has the same adduct as the query | The same adduct should fragment in a more similar way, so those matches should be more reliable | 0.9248 / 0.9276 | | Failed: slightly worse than E501 (0.9294). See below | `members/Mrnidhi/entropy_variants.py` |

## Failed / Ineffective Approaches
_For each: what was tried, what happened, why we think it failed, and what we learned._

### E511: same-adduct bonus on entropy similarity (Mrnidhi)
- **What was tried:** raise a reference spectrum's entropy similarity by 10% or 30% when it has the same adduct as the query spectrum.
- **What happened:** MRR@25 fell from 0.9294 to 0.9248 (10%) and 0.9276 (30%).
- **Why we think it failed:** the best match is already taken over all same-mode reference spectra, so a same-adduct spectrum wins on its own merit when it really is the closest. The bonus mostly lifts weaker matches of wrong candidates that happen to have the same adduct. The differences are also small enough to be noise on 250 molecules.
- **What we learned:** adduct information is better given to the LightGBM ranker (E502) as a separate feature than forced into the similarity score by hand.
