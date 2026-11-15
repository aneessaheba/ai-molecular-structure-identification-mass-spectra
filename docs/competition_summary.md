# Competition Summary (G.1)

Link: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
Sponsor: Enveda Therapeutics · Prize pool: $50,000 · Data license: CC BY-NC 4.0

## Key Dates (11:59 PM UTC)
| Date | Event |
|---|---|
| Sep 14, 2026 | Competition start |
| **Dec 7, 2026** | Entry deadline and team merger deadline |
| **Dec 14, 2026** | Final submission deadline |

## Task
Identify the 2D chemical structure of unknown small molecules from their tandem mass spectra (LC-MS/MS).
For each **molecule** (not each spectrum), output up to **25 candidate structures as SMILES**, best guess first.
A molecule can have several spectra (different collision energies or adducts), so evidence must be aggregated per `molecule_id`.

The test molecules resemble natural products (plants, mammals, microbes). Each belongs to a hidden novelty class:

| Class | Meaning | How it can be solved |
|---|---|---|
| 1 | Structure has public reference spectra | Spectral library search against training data |
| 2 | Known structure (PubChem / COCONUT), no public spectra | Retrieve candidates from a structure database and rank them |
| 3 | Novel structure, not in PubChem | De novo generation |

The mix of classes is hidden.

## Input
**test.parquet** (~1,500 spectra, ~400 molecules, 1–16 spectra per molecule, median 3; all Bruker timsTOF; monoisotopic mass 157–1,159 Da, median 348).
The downloadable file only contains examples taken from train. Kaggle swaps in the hidden test set when the notebook is re-run.

| Column | Meaning |
|---|---|
| `molecule_id` | Unit of prediction |
| `spectrum_id` | One spectrum |
| `ms2_mzs`, `ms2_normalized_intensities` | Aligned peak arrays, base peak = 1.0 |
| `base_peak_intensity` | Raw intensity of base peak (higher → less noisy) |
| `adduct` | One of 10: `[M+H]+`, `[M+NH4]+`, `[M-H2O+H]+`, `[M-2H2O+H]+`, `[M+Na]+`, `[M+K]+`, `[M-H]-`, `[M-H2O-H]-`, `[M+CH2O2-H]-`, `[M+Cl]-` |
| `ionization_mode` | positive / negative |
| `precursor_mz` | Measured precursor m/z |
| `collision_energy_ev` | List of eV values (several = merged acquisitions) |

Test spectra are minimally cleaned: precursor-inconsistent spectra and very weak spectra (base peak < 1,000 counts) were dropped, and peaks above precursor + 2 Da were removed.

**train.parquet** (~2.5M spectra, ~275k unique structures, 3 GB) has the same columns plus the labels:
`normalized_smiles` (label), `inchikey`, `inchikey14`, `molecular_formula`, `ingest_lib`, `adduct_orig`, `precursor_error_ppm`, `num_peaks`.

Training sources (`ingest_lib`):
- **enveda-180** (1.15M spectra): same timsTOF instrument as test, but drug-like synthetic chemistry, so a different chemical space.
- **enveda-np-examples** (1,151 spectra, 250 molecules): same instrument and pipeline as test. It's the closest data to the test set, which makes it the best local validation set.
- Public libraries: pluskal_ms2 (Orbitrap, NCE), riken, gnps, massbank, mona, spectraverse, msdial, drug_plus, masaryk. These have mixed instruments and preprocessing.
- Use `collision_energy_ev` (converted to eV) rather than the original CE column.
- The same compound recurs across libraries, so validation splits must be done **by structure (`inchikey14`)** to avoid leakage.

## Output / Submission Format
`submission.csv`:
```
molecule_id,smiles
m_0014ef,CC1=CC(=O)C=CC1=O;OC(=O)c1ccccc1O;CN1C=NC2=C1C(=O)N(C)C(=O)N2C
```
- Every `molecule_id` exactly once; up to 25 SMILES joined by `;`, best first.
- Rejected if: missing column, empty, nulls, duplicate `molecule_id`, or > 25 guesses.

## Evaluation Metric
**MRR@25** (Mean Reciprocal Rank): score = mean over molecules of 1 / rank of the first correct guess (0 if not in top 25).
Rank 1 = 1.0, rank 2 = 0.5, rank 25 = 0.04.

A guess is correct if its **InChIKey14** (first block of the InChIKey, after RDKit tautomer canonicalization, RDKit 2026.03.3) matches the answer.
Stereochemistry and tautomer form don't count, only atom connectivity. So duplicate guesses that share an InChIKey14 waste slots and should be removed.
Official metric notebook: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank

## Rules & Constraints
- **Code competition:** submissions must be made from a Kaggle Notebook that writes `submission.csv`.
- **Runtime:** ≤ 9 hours (CPU or GPU notebook).
- **Internet:** disabled during the submission run. All models and data must be attached as Kaggle datasets or models.
- **External data:** allowed if freely and publicly available, including pre-trained models.
- **Submissions:** 5 per day per team.
- **Team size:** max **5** on Kaggle, so our 8-person class team can't be a single Kaggle team. Possible splits: two Kaggle teams (4 + 4), or members submitting individually. The course brief requires every member to submit to the leaderboard.
- One Kaggle account per person.

## Available Resources
- Training data: 2.5M labelled spectra (see above).
- Suggested curation: relative-intensity floor (0.1–2%), drop peaks above the precursor, minimum ~5 peaks, top-N peaks (e.g. 128), deisotoping, sqrt/log intensity.
- Tools: [matchms](https://github.com/matchms/matchms) (spectrum cleaning and similarity), RDKit, SIRIUS / MIST-CF (molecular formula annotation).
- Pre-training: MassIVE, GeMS, and the DreaMS pretrained spectrum-embedding models (https://zenodo.org/records/10997887).
- Extra libraries: FragHub. Candidate structure databases (for Class 2): PubChem, COCONUT.
- Compute: Kaggle notebooks (GPU quota ~30 h/week per user), Colab, local machines.
