# Experiment Log (G.3)

Log **every** experiment, including failed ones. One row per run.

| ID | Date | Member | Model / Change | Why (hypothesis) | Local Val Score | Kaggle Score | Result / Takeaway | Code / Notebook |
|----|------|--------|----------------|------------------|-----------------|--------------|-------------------|-----------------|
| E001 | 2026-11-14 | aneessaheba | Baseline A: precursor-mass match (15 ppm) over training structures, ranked by library frequency | Simplest valid ranking; sets the floor | 0.8235 (val: enveda-np-examples held out, 250 mol) | | Top-1 73%, top-25 99%. Validation molecules are common compounds, so this val set is easy (Class 1 only) | `src/models/library_search.py --no-cosine` |
| E002 | 2026-11-14 | aneessaheba | Baseline B: E001 + greedy cosine (sqrt intensity, top-64 peaks, 1% floor, 0.01 Da) vs candidate reference spectra | Spectral similarity should separate isomers that share a mass | 0.8789 | | +0.055 MRR over E001; top-1 73% → 82%. First Kaggle submission | `src/models/library_search.py`, `kaggle/library_search_baseline/` |
| E003 | 2026-11-15 | aneessaheba | Sweep m/z tolerance (0.002–0.05 Da), max vs mean over spectra, cross-mode matching | Check the baseline's first-guess search settings | 0.850–0.885 | | 0.01 Da was already best; max aggregation hurts (0.850); cross-mode +0.006. See sweep CSV/figure | `scripts/sweep_library_search.py` |
| E004 | 2026-11-15 | aneessaheba | Sweep reference spectra per candidate (10 / 50 / 200 / 500 / all) | The cap of 50 throws away evidence for well-measured structures | 0.783 → **0.9245 (all)** | | Biggest single gain (+0.046). 10 refs drops to 0.783 | same |
| E005 | 2026-11-15 | aneessaheba | Sweep peak cleaning: intensity floor (0–2%), top-N (32/64/128), intensity power (0.25/0.5/1) | Noise peaks may dilute the cosine | 0.873–0.884 | | top-32 +0.005; other changes within ±0.006 | same |
| E006 | 2026-11-15 | aneessaheba | **Tuned library search**: all reference spectra + top-32 peaks (new default) | Combine the two gains | **0.9289** (top-1 89.2%) | | +0.050 over E002. Floor 0% adds only +0.001, so not adopted. Gain likely smaller on the hidden test (validation molecules are common, well-measured compounds) | `kaggle/library_search_tuned/` |

## Failed / Ineffective Approaches
_For each: what was tried, what happened, why we think it failed, and what we learned._

- **E003: max aggregation across a molecule's spectra (0.850 vs 0.879).** Taking only the single best-matching spectrum lets one lucky match on a wrong isomer win. Averaging over all of a molecule's spectra (different collision energies/adducts) is more robust: evidence must agree across spectra.
- **E003: tighter m/z tolerance (0.002 Da: 0.867).** Library spectra come from many instruments with different mass accuracy, so a very tight window drops true fragment matches. 0.01 Da is a good compromise.
- **E004: fewer reference spectra (10 refs: 0.783).** Each structure's library spectra cover different energies and instruments; cutting them removes the spectrum most similar to the query.
