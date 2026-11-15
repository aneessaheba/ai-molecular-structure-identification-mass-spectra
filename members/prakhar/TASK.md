# Prakhar: Contrastive Spectrum–Molecule Retrieval (CLIP-style)

> Rename this folder to your GitHub username once you have it: `git mv members/prakhar members/<username>`.

**Kaggle team:** GROUP_PROGRAMMING_01_b · **Experiment IDs:** E300–E399
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Train **two encoders**, one for spectra and one for molecules, so that a spectrum and its true molecule land **close together** in the same 256-dim space, and wrong molecules land far away (like CLIP does for images and captions).
Score each candidate = cosine similarity between the spectrum embedding and the candidate's embedding.
This learns the ranking task directly, unlike fingerprint prediction, which learns a proxy.

```
spectrum ─► bin_spectrum ─► spectrum encoder (MLP) ─┐
                                                    ├─► cosine ─► score
candidate ─► morgan_fp ─► molecule encoder (MLP) ───┘
```

---

## Model 1: Two MLP encoders + InfoNCE (due Thu Nov 19)

**Data**
```python
from src.data.dataset import load_training, split_by_structure
from src.features.featurize import bin_spectrum, metadata_vector, morgan_fps
df = load_training(max_per_structure=3)          # sample=200_000 while debugging
tr, va = split_by_structure(df, val_frac=0.05)
```

**Network** (PyTorch)
- Spectrum encoder: input = `bin_spectrum` (1000) ⊕ `metadata_vector` (12) → `Linear 1012→1024 → ReLU → Dropout 0.2 → Linear 1024→256`, then L2-normalise.
- Molecule encoder: input = `morgan_fp` (2048) → `Linear 2048→1024 → ReLU → Linear 1024→256`, then L2-normalise.
- **Loss: InfoNCE** with in-batch negatives: `logits = S @ M.T / τ`, cross-entropy with targets = diagonal, applied both ways (spectra→molecules and molecules→spectra). τ learnable, init 0.07.
- **Important:** a batch must not contain the same structure twice (it would count as a "negative"). Sample one spectrum per structure per batch.
- AdamW lr 1e-3, batch 512 (bigger = more negatives = better), ~20 epochs, early stopping on `va` loss.

**Scorer:** embed each spectrum, **average** the embeddings (re-normalise), embed each candidate's fingerprint, return the cosine similarities. Validate → E301.

## Model 2: Hard negatives (due Sun Nov 22)
Random negatives are too easy. The real task is telling **isomers** apart (same formula).
For each training structure, add 1–3 **same-formula** structures as extra negatives:
```python
structs = pd.read_parquet("data/processed/structures.parquet")   # inchikey14, smiles, formula, mass
isomers = structs.groupby("formula").smiles.apply(list)           # formula → list of SMILES
```
Add their embeddings as extra columns in the logits matrix. Log E302: does it beat Model 1?

## Model 3 (optional, Tue Nov 24)
- **Blend** with the baseline: `a * clip_score + (1 - a) * candidates.cosine`, sweep a; and/or
- ablation: embedding size 128 vs 256 vs 512, or fixed vs learnable τ.

---

## What to expect
- Alone it may be below the 0.877 baseline on validation (all validation molecules are in the library). The hard-negative model and the blend are where you can gain.
- Watch for **collapse** (all embeddings identical, loss stuck at log(batch)). If so, lower lr and check the L2-normalisation.

## Kaggle notes
Upload both encoders' weights as `casmi-prakhar-weights`. Needs RDKit for fingerprints; tell Anees if `import rdkit` fails on Kaggle.

## Checklist
- [ ] Send Anees your GitHub username; accept invites; setup done, baseline 0.877 (Nov 15)
- [ ] Model 1 trained, validated, logged (E301), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 trained, validated, logged (E302), submitted (Nov 22)
- [ ] (Optional) Model 3 (E303)
- [ ] Hand-off README + figure + screenshots, PR opened (Nov 25, 9 PM)
