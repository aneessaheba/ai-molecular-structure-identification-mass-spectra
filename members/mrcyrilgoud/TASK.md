# Cyril: Spectrum → Fingerprint with MLPs

**Kaggle team:** GROUP_PROGRAMMING_01_b (**you are the team leader**: create the team, rename it `GROUP_PROGRAMMING_01_b`, invite Praveen, Vrishin and Prakhar, and submit the baseline for _b) · **Experiment IDs:** E100–E199
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Instead of matching spectra against a library, **predict the molecule's fingerprint** (2048 bits, each meaning "this substructure is present") directly from the spectrum.
Then score each candidate by how similar its real fingerprint is to the predicted one.
This works even for molecules the library has never seen, which the baseline can't handle.

```
spectrum ──bin_spectrum──► 1000-dim vector ──MLP──► 2048 probabilities ──Tanimoto──► score per candidate
```

---

## Model 1: Binned-spectrum MLP (due Thu Nov 19)

**Data**
```python
from src.data.dataset import load_training, split_by_structure
from src.features.featurize import bin_spectrum, morgan_fps
df = load_training(max_per_structure=3)        # start with sample=200_000 while debugging
tr, va = split_by_structure(df, val_frac=0.05)
X = np.stack([bin_spectrum(m, i) for m, i in zip(tr.mz, tr.intensity)])   # (N, 1000)
Y = morgan_fps(tr.smiles.tolist())                                         # (N, 2048)
```
Tip: compute fingerprints once per unique SMILES and cache them (`np.save`), since the same structure appears many times.

**Network** (PyTorch)
```
Linear(1000, 1024) → ReLU → Dropout(0.3)
Linear(1024, 1024) → ReLU → Dropout(0.3)
Linear(1024, 2048)            # logits; loss = BCEWithLogitsLoss
```
Adam, lr 1e-3, batch 256, up to 30 epochs, **early stopping on `va` loss** (patience 3). Seed everything.

**Scorer**
```python
class FingerprintMLP(Scorer):
    name = "fp-mlp"
    def score(self, spectra, candidates):
        X = np.stack([bin_spectrum(s["mz"], s["intensity"]) for s in spectra])
        prob = sigmoid(self.net(torch.tensor(X))).mean(0).numpy()     # average over the molecule's spectra
        return tanimoto(prob, morgan_fps(candidates.smiles.tolist()))
```
Validate with `evaluate(FingerprintMLP(...))` and log E101.

## Model 2: MLP + neutral losses + metadata (due Sun Nov 22)
Same network, richer input (1512 dims):
- `bin_spectrum(mz, intensity)` (1000)
- **neutral losses**: `bin_spectrum(precursor_mz - mz, intensity, max_mz=500)` (500). Losses like −18 (water) or −162 (sugar) are very informative.
- `metadata_vector(adduct, collision_energy, precursor_mz)` (12)

Log as E102. Compare with Model 1: did losses/metadata help?

## Model 3 (optional, Tue Nov 24): blend with the baseline
`score = a * fp_score + (1 - a) * candidates.cosine.values`, try a ∈ {0.1, 0.2, 0.3, 0.5}. Report the best a **and** the curve (a vs MRR). That's a nice figure for the report.

---

## What to expect
- Alone, the fingerprint model will probably score **below** the baseline's 0.877 (the validation molecules are all in the library, which favours library search). That's fine and expected. **Explain it in your hand-off.** The blend is where you can beat the baseline.
- Things to try if time allows: 4096 bits, bigger hidden layers, `log1p` vs sqrt intensities, separate models per ionisation mode.

## Kaggle notes
Upload `model.pt` (+ any normalisation constants) as dataset `casmi-cyril-weights`. Your scorer needs RDKit for `morgan_fps`; if `import rdkit` fails on Kaggle, tell Anees (wheels dataset).

## Checklist
- [x] Setup done, baseline check prints 0.877 (Nov 15)
- [x] Model 1 trained, validated, logged (E101), **submitted to Kaggle** (Nov 19)
- [x] Model 2 trained, validated, logged (E102), submitted (Nov 22)
- [x] (Optional) blend, logged (E103)
- [x] Hand-off README + figure + screenshots, PR opened (Nov 25, 9 PM)
