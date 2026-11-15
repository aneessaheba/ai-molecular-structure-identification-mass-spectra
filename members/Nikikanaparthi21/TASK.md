# Nikhil: Transformer Spectrum Encoder → Fingerprint

**Kaggle team:** GROUP_PROGRAMMING_01_a · **Experiment IDs:** E200–E299
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Treat a spectrum as a **set of peaks** and let a Transformer learn how peaks relate (e.g. two peaks 18.011 apart = water loss).
Predict the molecule's 2048-bit Morgan fingerprint, then score each candidate by Tanimoto similarity.
Unlike binning, this keeps the **exact m/z** (e.g. 163.0601 vs 163.0390 are different fragments).
Cyril does the MLP version, so the two of you are a direct architecture comparison for the report.

```
peaks (m/z, intensity) ──Fourier m/z encoding──► Transformer (4 layers) ──[precursor token]──► 2048 logits ──Tanimoto──► score
```

---

## Model 1: Peak-set Transformer (due Thu Nov 19)

**Data**
```python
from src.data.dataset import load_training, split_by_structure
from src.features.featurize import peak_tensor, morgan_fps
df = load_training(max_per_structure=3)          # use sample=200_000 while debugging
tr, va = split_by_structure(df, val_frac=0.05)
# per spectrum: x (64, 3) = [m/z, intensity, neutral loss], mask (64,) ; row 0 = precursor
x, mask = peak_tensor(mz, intensity, precursor_mz, max_peaks=64)
```

**Network** (PyTorch)
- **m/z encoding:** Fourier features: `[sin(2π·mz/λ_k), cos(2π·mz/λ_k)]` for 64 wavelengths λ log-spaced from 0.01 to 1000. Small λ captures the decimals that matter. Linear → d=256.
- Add `Linear(1, 256)(intensity)`.
- `nn.TransformerEncoder(d_model=256, nhead=8, dim_feedforward=1024, num_layers=4, dropout=0.1, batch_first=True)` with `src_key_padding_mask = ~mask`.
- Take the output at position 0 (precursor token) → `Linear(256, 2048)` logits → `BCEWithLogitsLoss`.
- AdamW lr 3e-4, weight decay 1e-2, batch 256, linear warm-up 1,000 steps, ~15 epochs, early stopping on `va` loss. **Use a GPU** (Kaggle/Colab).

**Scorer:** predict probabilities for each of the molecule's spectra, **average** them, then `tanimoto(prob, morgan_fps(candidates.smiles.tolist()))`. Validate with `evaluate(...)` → E201.

## Model 2: + neutral losses + metadata tokens (due Sun Nov 22)
- Encode the **neutral loss** column (`x[:, 2]`) with its own Fourier encoder and add it to each peak's embedding.
- Prepend **metadata tokens**: an `nn.Embedding` for the adduct (10 types) and a `Linear(1, 256)` for collision energy (use 0 when NaN).
Log E202 and compare with Model 1.

## Model 3 (optional, Tue Nov 24)
Pick one:
- **Blend** with the baseline: `a * fp_score + (1 - a) * candidates.cosine`, sweep a.
- **Multi-spectrum pooling**: encode all of a molecule's spectra and attention-pool before the fingerprint head (train on molecules, not spectra).

---

## What to expect
- The transformer alone will likely score below the 0.877 baseline on our validation set. The blend is where you can beat it. **Compare with Cyril's MLP** at a similar model size; "transformer vs MLP" is a key slide.
- If training is unstable: lower lr to 1e-4, check that padding is masked, clip gradients at 1.0.

## Kaggle notes
Upload weights as `casmi-nikhil-weights`. Turn the **GPU on** in the submission notebook if inference is slow (~1,500 spectra is small, so CPU is probably fine).

## Checklist
- [ ] Setup done, baseline check prints 0.877 (Nov 15)
- [ ] Model 1 trained, validated, logged (E201), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 trained, validated, logged (E202), submitted (Nov 22)
- [ ] (Optional) Model 3 (E203)
- [ ] Hand-off README + training-curve figure + screenshots, PR opened (Nov 25, 9 PM)
