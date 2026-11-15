# Experiment Log (G.3)

Log **every** experiment, including failed ones. One row per run.

| ID | Date | Member | Model / Change | Why (hypothesis) | Local Val Score | Kaggle Score | Result / Takeaway | Code / Notebook |
|----|------|--------|----------------|------------------|-----------------|--------------|-------------------|-----------------|
| E001 | 2026-11-14 | aneessaheba | Baseline A: precursor-mass match (15 ppm) over training structures, ranked by library frequency | Simplest valid ranking; sets the floor | 0.8235 (val: enveda-np-examples held out, 250 mol) | | Top-1 73%, top-25 99%. Validation molecules are common compounds, so this val set is easy (Class 1 only) | `src/models/library_search.py --no-cosine` |
| E002 | 2026-11-14 | aneessaheba | Baseline B: E001 + greedy cosine (sqrt intensity, top-64 peaks, 1% floor, 0.01 Da) vs candidate reference spectra | Spectral similarity should separate isomers that share a mass | 0.8789 | | +0.055 MRR over E001; top-1 73% → 82%. First Kaggle submission | `src/models/library_search.py`, `kaggle/library_search_baseline/` |

## Failed / Ineffective Approaches
_For each: what was tried, what happened, why we think it failed, and what we learned._
