# Vrishin: Analog Search + New Candidates from COCONUT (Class 2 molecules)

**Kaggle team:** GROUP_PROGRAMMING_01_b · **Experiment IDs:** E600–E699
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Part of the hidden test set are **"Class 2" molecules**: known structures (in databases like COCONUT/PubChem) but with **no reference spectra**.
Our baseline can't find them at all, because it only proposes structures that have training spectra.
Your job is to fix that:
1. **Propose new candidates** from **COCONUT** (~700k natural products, free, CC0 license) with the right mass.
2. **Rank them without reference spectra**, using **analog search**: find library spectra that look similar (even at a different mass). Their structures are probably related to ours, so prefer candidates that are structurally similar (Tanimoto) to those analogs.

---

## Model 1: Analog-search scorer (due Thu Nov 19)
For each query spectrum:
1. Find the **top-k (k=20) most similar library spectra in the same ionisation mode, at any precursor mass**.
   Use `bin_spectrum(mz, intensity, bin_size=1.0)` (1000-dim) for every reference spectrum, stored as a `scipy.sparse.csr_matrix` with rows L2-normalised. Then similarity is one sparse matrix × vector product. Build the matrix once and save it with `scipy.sparse.save_npz`.
   ```python
   from src.models.library_search import ReferenceLibrary
   lib = ReferenceLibrary("data/processed", exclude_libs=["enveda-np-examples"])  # validation
   # lib.meta has struct_idx, ionization_mode for each spectrum j;  lib.peaks(j) -> (mz, intensity)
   # lib.structs.smiles[struct_idx] -> the analog's structure
   ```
2. Score each candidate as `Σ_k sim_k · tanimoto(fp(analog_k), fp(candidate)) / Σ_k sim_k`.

Validate normally with `evaluate(...)` → E601. Also try blending it with the baseline cosine.

## Model 2: COCONUT candidate expansion (due Sun Nov 22)
1. Download COCONUT: https://coconut.naturalproducts.net → **Download** → the CSV with SMILES (~700k rows).
2. Preprocess once with RDKit: SMILES → neutral monoisotopic mass (`Descriptors.ExactMolWt`) and InChIKey14. Save `coconut.parquet` (smiles, inchikey14, mass), sorted by mass.
3. In your scorer, implement:
   ```python
   def extra_candidates(self, spectra):
       m = median neutral mass of the spectra   # see src.data.spectra.neutral_mass(precursor_mz, adduct)
       return COCONUT SMILES with |mass - m| <= m * 15e-6
   ```
   The pipeline adds these to the candidate table (with `cosine = 0`, `n_spectra = 0`). Your Model 1 analog score then ranks everything.
4. **Validate the Class 2 setting:** `evaluate(MyModel(), hide_true=True)` removes the true structure from the library candidates, so the only way to score is if COCONUT supplies it and you rank it high. The baseline scores **0.000** here, so any number > 0 is a real gain. Also report what fraction of validation molecules are in COCONUT (that's the ceiling).
5. Also run normal `evaluate(MyModel())`: adding COCONUT candidates must not hurt the Class 1 score much. → E602.

## Model 3 (optional, Tue Nov 24)
Calibrate library vs COCONUT candidates: e.g. keep library candidates first when their cosine > t, otherwise mix in COCONUT candidates by analog score. Sweep t under both `hide_true=False` and `True` → E603. This trade-off curve is a great report figure.

---

## What to expect
Normal validation may stay near 0.929. **The hide_true score is your headline number.** It shows we can now get molecules the baseline can never find.

## Kaggle notes
Upload `coconut.parquet` and the sparse analog matrix (or build the matrix inside the notebook, ~minutes) as dataset `casmi-vrishin-data`. COCONUT is free and public, so it's allowed under the rules.

## Checklist
- [ ] Setup done, baseline 0.929 (Nov 15)
- [ ] Model 1 (analog scorer) validated, logged (E601), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 (COCONUT expansion) validated with and without hide_true, logged (E602), submitted (Nov 22)
- [ ] (Optional) Model 3 calibration (E603)
- [ ] Hand-off README + figure + screenshots, PR opened (Nov 25, 9 PM)
