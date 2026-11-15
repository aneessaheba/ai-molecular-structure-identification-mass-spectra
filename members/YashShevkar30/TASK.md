# Yash: Generating New Structures (Class 3 / novel molecules)

**Kaggle team:** GROUP_PROGRAMMING_01_a · **Experiment IDs:** E700–E799
Read [members/README.md](../README.md) first (setup, validation, Kaggle, logging).

## The idea
Part of the hidden test set is **"Class 3": brand-new molecules that exist in no database**. The only way to get them is to **generate** structures.
This is the hardest part of the competition (open research), so treat it as an experiment. **A well-documented negative result is a valid outcome** and gives the report its required "failed approach".

Your models implement `extra_candidates(spectra)`, which returns **new SMILES** that the pipeline adds to the candidate list.
Evaluate with `evaluate(MyModel(), hide_true=True)`: the true structure is removed from the library candidates, so you only score if **you generate it**. The baseline scores **0.000** here.

---

## Model 1: Analog + modification enumeration (due Thu Nov 19, do this first)
Natural products often come in **families**: the unknown is a known molecule **plus or minus a small modification**.
1. Compute the query's neutral mass `M` (`src.data.spectra.neutral_mass(precursor_mz, adduct)`).
2. For each modification with mass Δ:

   | Modification | Δ mass (Da) | RDKit edit (reaction SMARTS) |
   |---|---|---|
   | hydroxylation (+O) | +15.9949 | aromatic `[cH]` → `c(O)` |
   | methylation (+CH2) | +14.0157 | `[OH]` → `OC`, `[NH]` → `N(C)` |
   | demethylation (−CH2) | −14.0157 | `O[CH3]` → `[OH]` |
   | hydrogenation (+H2) | +2.0157 | `C=C` → `CC` |
   | dehydrogenation (−H2) | −2.0157 | `CC` → `C=C` (sparingly) |
   | glycosylation (+hexose) | +162.0528 | `[OH]` → `O-glucose` |
   | deglycosylation | −162.0528 | remove a sugar |

   find **analogs** = library structures with mass `M − Δ` (`lib.candidates(M - Δ, ppm=15)` from `ReferenceLibrary`), and keep the top few by spectral cosine (`src.data.spectra.cosine` against their reference spectra).
3. Apply the edit **at every possible site** with `AllChem.ReactionFromSmarts(...).RunReactants`, sanitise, and keep products whose exact mass is within 15 ppm of `M`.
4. Return up to ~200 unique products from `extra_candidates`. Rank them with `score` = the parent analog's cosine (or Tanimoto to the top analogs).

Validate with `evaluate(MyModel(), hide_true=True)` → E701. Report: how often the true structure was among the generated ones (recall), and the MRR.

## Model 2: Spectrum → SMILES Transformer (due Sun Nov 22; high risk)
A sequence-to-sequence model: **encoder** over the peaks (you can reuse `peak_tensor`; a 4-layer transformer is enough), **decoder** generating SMILES character by character (or **SELFIES** via `pip install selfies`, which always decodes to a valid molecule).
- Train on `load_training(max_per_structure=2)` (~300k pairs) on a GPU, about 10 epochs, with teacher forcing.
- Generate with **beam search (beam 25)**, keep valid molecules within 15 ppm of `M`, and return them from `extra_candidates`.
- Evaluate with `hide_true=True` → E702. Expect a very low score in this time frame. If so, **analyse why** (invalid SMILES rate? right formula but wrong structure? too little training time?). That analysis is your deliverable.

## Model 3 (optional, Tue Nov 24)
Improve whichever worked better (more modification types, better ranking of generated products), or combine both generators → E703.

---

## Report value
Even small hide_true numbers (e.g. MRR 0.02) are meaningful, because the baseline can never find these molecules.
Report: number of candidates generated per molecule, % valid, % within mass, recall@25 of the true structure, MRR.

## Kaggle notes
Model 1 needs no weights, only RDKit (tell Anees if `import rdkit` fails on Kaggle). Model 2: upload the weights as `casmi-yash-weights`. Generation must fit in the 9 h limit (~400 molecules), so time it on 20 molecules first.

## Checklist
- [ ] Setup done, baseline 0.877 (Nov 15)
- [ ] Model 1 validated with hide_true, logged (E701), **submitted to Kaggle** (Nov 19)
- [ ] Model 2 trained and evaluated, logged (E702), submitted if valid (Nov 22)
- [ ] (Optional) Model 3 (E703)
- [ ] Hand-off README with **failure analysis** + figure + screenshots, PR opened (Nov 25, 9 PM)
