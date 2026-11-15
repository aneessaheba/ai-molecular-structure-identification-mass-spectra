# Praveen: Pretrained Spectrum Embeddings (DreaMS)

**Kaggle team:** GROUP_PROGRAMMING_01_b · **Experiment IDs:** E400–E499
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
**DreaMS** is a transformer pretrained on ~700 million unlabelled MS/MS spectra (Pluskal lab, *Nature Biotechnology* 2025).
It turns any spectrum into a 1024-dim embedding where similar molecules are close.
Instead of training from scratch, we **reuse** it, which is transfer learning: a key "related work" story for the report.

- Code: https://github.com/pluskal-lab/DreaMS
- Weights and data: https://zenodo.org/records/10997887 (linked from the competition data page)

---

## Day 1 (Sun Nov 15–Mon Nov 16): get DreaMS running
1. Install it following the GitHub README (a separate conda/venv env is fine).
2. Embed **10 spectra** from `data/raw/test.parquet` and print the embedding shape.
3. Post "DreaMS works" in the group chat.

⚠️ **If it doesn't work by Tue Nov 17 evening, switch to the fallback: MS2DeepScore** (`pip install ms2deepscore`; pretrained model on Zenodo, linked in its README). Same plan below; just replace "DreaMS" with "MS2DeepScore". Don't lose more than 2 days on installation.

## Model 1: Embedding similarity search (due Thu Nov 19)
Replace the baseline's peak cosine with **DreaMS embedding cosine**:
score(candidate) = mean over the query spectra of the max cosine(query embedding, embedding of each of the candidate's reference spectra).

You need the candidates' **reference spectra**:
```python
from src.models.library_search import ReferenceLibrary
lib = ReferenceLibrary("data/processed", exclude_libs=["enveda-np-examples"])  # for validation
# lib = ReferenceLibrary("data/processed")                                    # for Kaggle
for idx in candidates.struct_idx:                 # one row per candidate
    for j in lib.spectra_of(idx):                 # that structure's reference spectra
        mz, intensity = lib.peaks(j)              # cleaned peaks; lib.meta.iloc[j] has precursor_mz, adduct
```
- **Cache embeddings** in a dict `j → vector`. Only candidate spectra are needed (~100k for validation), not all 2.3M.
- `lib.peaks` intensities are sqrt-scaled and L2-normalised. Check what DreaMS expects; `intensity ** 2` undoes the sqrt.
- Validate → E401.

## Model 2: DreaMS embedding → fingerprint head (due Sun Nov 22)
Freeze DreaMS. Train a small MLP: `1024 → 1024 → 2048` (BCE loss) on `load_training(max_per_structure=3)` embeddings, with targets `morgan_fps(smiles)`.
Score candidates with `tanimoto(predicted_prob, morgan_fps(candidates.smiles))`.
This can rank candidates **that have no reference spectra**. Log E402.
Embedding ~500k spectra takes a while: use a Kaggle/Colab GPU, save embeddings to disk (`np.save`), and start this early.

## Model 3 (optional, Tue Nov 24)
**Blend** Model 1 or 2 with the baseline: `a * dreams_score + (1 - a) * candidates.cosine`; sweep a. → E403.

---

## What to expect
Model 1 should be close to or above the 0.877 baseline (it's a better similarity measure on the same library). Model 2 alone will likely be lower, but it's the general one. Both results are valuable.

## Kaggle notes (internet is OFF)
- Upload the DreaMS **weights** and the **package wheels** as a dataset `casmi-praveen-dreams`: `pip download <dreams package and deps> -d wheels/`, then in the notebook `pip install --no-index --find-links /kaggle/input/casmi-praveen-dreams/wheels ...`.
- Turn the **GPU on** for the submission notebook. Embedding candidates' reference spectra at submission time must fit in 9 h. It should take minutes, but time it.

## Checklist
- [ ] Setup done, baseline 0.877 (Nov 15); DreaMS embeds 10 spectra (Nov 16)
- [ ] Model 1 validated, logged (E401), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 validated, logged (E402), submitted (Nov 22)
- [ ] (Optional) blend (E403)
- [ ] Hand-off README + figure + screenshots, PR opened (Nov 25, 9 PM)
