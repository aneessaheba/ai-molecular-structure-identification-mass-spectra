# Experiment Log (G.3)

Log **every** experiment, including failed ones. One row per run.

| ID | Date | Member | Model / Change | Why (hypothesis) | Local Val Score | Kaggle Score | Result / Takeaway | Code / Notebook |
|----|------|--------|----------------|------------------|-----------------|--------------|-------------------|-----------------|
| E001 | 2026-11-14 | aneessaheba | Baseline A: precursor-mass match (15 ppm) over training structures, ranked by library frequency | Simplest valid ranking; sets the floor | 0.8235 (val: enveda-np-examples held out, 250 mol) | | Top-1 73%, top-25 99%. Validation molecules are common compounds, so this val set is easy (Class 1 only) | `src/models/library_search.py --no-cosine` |
| E002 | 2026-11-14 | aneessaheba | Baseline B: E001 + greedy cosine (sqrt intensity, top-64 peaks, 1% floor, 0.01 Da) vs candidate reference spectra | Spectral similarity should separate isomers that share a mass | 0.8789 | | +0.055 MRR over E001; top-1 73% -> 82%. First Kaggle submission | `src/models/library_search.py`, `kaggle/library_search_baseline/` |
| E601 | 2026-11-26 | Viki2183 | Sparse same-ionisation-mode analog search: 1.0 m/z, L2-normalised CSR rows, top-20 at any precursor mass, weighted Morgan Tanimoto | Related spectra may identify structurally similar candidates even at a different mass | MRR@25 0.933871; top-1 0.904; top-10 0.992; top-25 0.996; 250 molecules; 406s | | Final executed Kaggle CPU notebook result using official competition data | `members/Viki2183/model_analog_coconut.py`, `members/Viki2183/Viki2183_CASMI26_Analog_COCONUT_E601_E602_final_executed.ipynb` |
| E602 | 2026-11-26 | Viki2183 | Analog search plus COCONUT candidate expansion, normal evaluation | Mass-matched natural products can recover Class 2 molecules without reference spectra | MRR@25 0.911390; top-1 0.884; top-10 0.964; top-25 0.972; 250 molecules; 451s | | COCONUT November 2026; 479,694 unique molecules after preprocessing; final executed Kaggle CPU notebook result | `members/Viki2183/model_analog_coconut.py`, `members/Viki2183/Viki2183_CASMI26_Analog_COCONUT_E601_E602_final_executed.ipynb` |
| E602 | 2026-11-26 | Viki2183 | Analog search plus COCONUT candidate expansion, hide_true=True evaluation | Candidate expansion should recover structures absent from the reference library | MRR@25 0.890668; top-1 0.856; top-10 0.952; top-25 0.960; 250 molecules; 444s | | COCONUT November 2026; 479,694 unique molecules after preprocessing; final executed Kaggle CPU notebook result | `members/Viki2183/model_analog_coconut.py`, `members/Viki2183/Viki2183_CASMI26_Analog_COCONUT_E601_E602_final_executed.ipynb` |

COCONUT validation coverage: **NOT COMPUTED**. This percentage must be
computed from the validation set; no verified coverage value is available.

## Failed / Ineffective Approaches

No additional model failure is reported here. The final E601 and E602 values
above are copied from the executed notebook output and are not invented.
