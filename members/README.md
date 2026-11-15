# Team Guide: Start Here

Hi team! This page tells you exactly what to do. **Your only job is your models.** Anees (team leader) handles the report, slides, shared code, EDA and everything else.

👉 After reading this page, open **your own task file**: `members/<your-github-username>/TASK.md`.

| Member | GitHub | Kaggle team | Your task file |
|---|---|---|---|
| Anees (leader) | aneessaheba | GROUP_PROGRAMMING_01_a | (report, slides, shared code, ensemble) |
| Cyril | mrcyrilgoud | GROUP_PROGRAMMING_01_a | [mrcyrilgoud/TASK.md](mrcyrilgoud/TASK.md) |
| Nikhil | Nikikanaparthi21 | GROUP_PROGRAMMING_01_a | [Nikikanaparthi21/TASK.md](Nikikanaparthi21/TASK.md) |
| Prakhar | _send your username_ | GROUP_PROGRAMMING_01_a | [prakhar/TASK.md](prakhar/TASK.md) |
| Praveen | Praveen-sud | GROUP_PROGRAMMING_01_b | [Praveen-sud/TASK.md](Praveen-sud/TASK.md) |
| Srinidhi | Mrnidhi | GROUP_PROGRAMMING_01_b | [Mrnidhi/TASK.md](Mrnidhi/TASK.md) |
| Vrishin | Viki2183 | GROUP_PROGRAMMING_01_b | [Viki2183/TASK.md](Viki2183/TASK.md) |
| Yash | YashShevkar30 | GROUP_PROGRAMMING_01_b | [YashShevkar30/TASK.md](YashShevkar30/TASK.md) |

---

## Deadlines (non-negotiable)

| When | What you must have done |
|---|---|
| **Sun Nov 15** | Setup done (Steps 1–4 below). Baseline check prints `MRR@25 ≈ 0.877`. |
| **Thu Nov 19** | **Model 1** validated locally **and submitted to Kaggle**. Score posted in the group chat. |
| **Sun Nov 22** | **Model 2** validated and submitted to Kaggle. |
| **Tue Nov 24** | Model 3 (optional), last Kaggle submissions. |
| **Wed Nov 25, 9 PM** | **Hand-off:** your `README.md` filled in, experiment log rows added, PR opened. **Anees writes the report from this, so anything missing won't be in the report.** |

The report is due Dec 1. There is no time after Nov 25 for model work.

---

## Part 1: The problem in 60 seconds

- Each **molecule** has 1–9 **MS/MS spectra**, lists of peaks (`m/z`, `intensity`). We must guess its **structure** (a SMILES string).
- We submit up to **25 guesses per molecule**, best first. The score is **MRR@25**: 1.0 if the right answer is first, 0.5 if second, … 0 if not in the top 25.
- **How our pipeline works:**
  1. From the precursor mass we already get a short list of **candidate structures** (all training structures with the same mass, about 15 per molecule).
  2. **Your model gives each candidate a score.** The pipeline sorts by your score and writes the submission.
- So **every model in this project is a "scorer"**: it receives one molecule's spectra and its candidates, and returns one number per candidate. You don't need to deal with candidate search, ranking, file formats or the metric. The shared code does all of that.

Also read: [docs/competition_summary.md](../docs/competition_summary.md) (the rules) and [notebooks/eda/01_data_exploration.ipynb](../notebooks/eda/01_data_exploration.ipynb) (what the data looks like).

---

## Part 2: Setup (do this today)

**Step 1: Accept invites and join.**
- GitHub invite: https://github.com/aneessaheba/ai-molecular-structure-identification-mass-spectra/invitations
- Kaggle: open the [competition](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra) → **Join Competition** → accept the rules. Then accept the **team invite** from your Kaggle team leader (see table above).

**Step 2: Clone and install** (Python 3.10–3.12):
```bash
git clone https://github.com/aneessaheba/ai-molecular-structure-identification-mass-spectra.git
cd ai-molecular-structure-identification-mass-spectra
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Step 3: Download the data** (3 GB). Kaggle → Settings → API Tokens → **Generate**, then follow the instructions shown to save the token. Then:
```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data/raw
cd data/raw && unzip '*.zip' && rm *.zip && cd ../..
```

**Step 4: Build the reference library and check the baseline:**
```bash
python -m src.data.build_reference      # ~1-3 min, writes data/processed/
python -m src.evaluation.validate       # must print  [library-cosine] MRR@25 = 0.877
```
✅ If you see `0.877`, you're set up. Post "setup done" in the group chat.
❌ If anything fails, post the **full error message** in the group chat.

No GPU on your laptop? Train on **Kaggle notebooks** (free GPU, 30 h/week) or **Google Colab**. Loading the training data needs ~5 GB RAM with `max_per_structure=3`, less with `sample=` (see below).

---

## Part 3: How to build a model

**1. Copy the template:**
```bash
cp members/_template/model_template.py members/<your-username>/model_<name>.py
```
It contains a `Scorer` class with a `score(spectra, candidates)` method. That's the only thing you must implement.

**2. What your `score` method receives:**
- `spectra`: a list of dicts, one per spectrum of the molecule. Each has `mz`, `intensity` (cleaned numpy arrays), `precursor_mz`, `adduct`, `ionization_mode`, `collision_energy`.
- `candidates`: a pandas DataFrame, one row per candidate, with `smiles`, `inchikey14`, `formula`, `mass`, `n_spectra` and `cosine` (the baseline's score, handy as an extra feature).
- **Return:** a numpy array with one score per candidate row (higher = better).

**3. Load training data** (no leakage: validation molecules are already removed):
```python
from src.data.dataset import load_training, split_by_structure
df = load_training(sample=50_000)            # quick start, ~10 s
df = load_training(max_per_structure=3)      # full run: ~740k spectra, ~30 s, ~5 GB RAM
train_df, val_df = split_by_structure(df)    # for early stopping; never splits a structure
```
Each row has: `smiles`, `inchikey14`, `adduct`, `ionization_mode`, `precursor_mz`, `collision_energy`, `mz`, `intensity`.

**4. Ready-made features** (`src/features/featurize.py`):
| Function | Gives you |
|---|---|
| `bin_spectrum(mz, intensity, max_mz=1000, bin_size=1.0)` | fixed-length vector (for MLP/CNN) |
| `peak_tensor(mz, intensity, precursor_mz, max_peaks=64)` | `(64, 3)` peaks + mask (for transformers) |
| `metadata_vector(adduct, collision_energy, precursor_mz)` | 12 numbers (adduct one-hot, CE, mass) |
| `morgan_fp(smiles)` / `morgan_fps(list)` | 2048-bit molecular fingerprint (the usual training target) |
| `tanimoto(pred, fps)` / `cosine_sim(pred, fps)` | compare a predicted fingerprint with candidate fingerprints |

**5. Validate**, the same way for everyone:
```python
from src.evaluation.validate import evaluate
evaluate(MyModel())                    # 250 validation molecules, prints MRR@25, top-1, top-10, top-25
evaluate(MyModel(), limit=20)          # quick debug on 20 molecules
evaluate(MyModel(), hide_true=True)    # only for models that propose new structures (see your task)
```
**Numbers to beat:** baseline library cosine = **0.877**, mass only = **0.824**.
A model that scores *below* the baseline is still useful: report it honestly (it counts as a "failed approach" in the report), and try **blending**: `score = a * your_score + (1 - a) * candidates.cosine.values`.

**Rules for a fair comparison:**
- Never train on the validation molecules. `load_training()` already excludes them. Don't load `enveda-np-examples` yourself.
- Fix your random seeds.
- Change one thing at a time and log every run (Part 5).

---

## Part 4: Submit to Kaggle

Submissions **must** come from a Kaggle notebook with **internet OFF** (≤ 9 h runtime).

1. **Upload your trained weights** as a private Kaggle dataset: Kaggle → **Create → New Dataset** → upload `model.pt` (or similar) → name it `casmi-<yourname>-weights`.
2. **Import the template:** Kaggle → **Create → Import Notebook** → upload [kaggle/TEMPLATE_submission.ipynb](../kaggle/TEMPLATE_submission.ipynb).
3. **Add inputs** (right panel → Add Input): the competition data, **`casmi-team-src`** (team code, shared by Anees), and your weights dataset.
4. Settings: **Internet OFF**, GPU only if needed.
5. In the "Your model" cell, paste your model class and set `model = MyModel(WEIGHTS)`.
6. **Save Version → Save & Run All.** When it finishes: open the version → **Submit to Competition**.
7. When the score appears: **take a screenshot** → `results/leaderboard_evidence/<yourname>_<model>.png`, and add a row to [submissions/submission_log.md](../submissions/submission_log.md).

**Submission slots:** each Kaggle team gets **5 per day**, shared by 4 people. **Max 1 submission per person per day**, and say "submitting now" in the chat first.

If you need a Python package that Kaggle doesn't have (e.g. `rdkit`), tell Anees; it will be added to a shared `casmi-wheels` dataset.

---

## Part 5: Logging (this is graded)

For **every** experiment, including the ones that don't work, add a row to [experiments/experiment_log.md](../experiments/experiment_log.md):

| ID | Date | Member | Model / Change | Why (hypothesis) | Local Val Score | Kaggle Score | Result / Takeaway | Code / Notebook |
|---|---|---|---|---|---|---|---|---|
| E1xx | 2026-11-18 | cyril | Binned MLP → fingerprint, Tanimoto score | Fragments predict substructures | 0.612 | | Below baseline alone; blend 0.3 → 0.889 | members/mrcyrilgoud/model_mlp.py |

**Your experiment IDs** (so they never collide): Cyril E100–199 · Nikhil E200–299 · Prakhar E300–399 · Praveen E400–499 · Srinidhi E500–599 · Vrishin E600–699 · Yash E700–799.

The course **requires at least one failed approach with an explanation**. A model that didn't beat the baseline is perfect for that. Write down *why* you think it failed.

---

## Part 6: Git workflow

```bash
git checkout main && git pull
git checkout -b <yourname>/<model>          # e.g. cyril/fingerprint-mlp
# ... work only inside members/<your-username>/ ...
git add members/<your-username> experiments/experiment_log.md
git commit -m "Add fingerprint MLP (val MRR 0.61)"
git push -u origin <yourname>/<model>
```
Then open a **Pull Request** on GitHub. Anees reviews and merges.

- **Only edit your own folder** plus your rows in the logs. If you need to change shared code in `src/`, ask Anees first.
- **Never commit** data, weights (`*.pt`, `*.pkl`), or big outputs. `members/*/weights/` is git-ignored, so keep weights there.
- Clear huge notebook outputs before committing.

---

## Part 7: Your hand-off (due Wed Nov 25, 9 PM)

Fill in `members/<your-username>/README.md` (the template is already in your folder). It needs:
1. One paragraph per model: what it is, inputs, architecture, training setup (epochs, lr, batch size, data size).
2. The results table: local MRR@25, top-1, top-10, Kaggle score.
3. **What didn't work and why** (at least one thing).
4. One figure: a training curve, or a bar chart comparing your models against the baseline. Save it to `members/<your-username>/`.
5. Your Kaggle screenshots in `results/leaderboard_evidence/`.

Anees copies this straight into the report and slides, so clear writing here = your work shows up properly.

**Stuck for more than 2 hours? Ask in the group chat.** Include what you tried and the full error.
