# Team Workflow

## Branches
- `main` holds working, reviewed code only. Do not push directly.
- Create a branch per task: `<your-name>/<short-task>` (e.g., `anees/eda-spectra`, `nidhi/gnn-ranker`).
- Open a Pull Request into `main` and ask at least one teammate to review.

## Where to Put Your Work
- **Your own models / notebooks** → `members/<your-github-username>/`
  - Every member must build **2–3 model architectures** and submit them to Kaggle (course requirement).
  - Name notebooks with a number + description: `01_fingerprint_mlp.ipynb`, `02_transformer_ranker.ipynb`.
- **Code that others will reuse** (data loaders, metric, featurizers) → `src/`
- **Shared EDA** → `notebooks/eda/`

## Logging (graded!)
1. **Every experiment** → add a row to `experiments/experiment_log.md`, including failed ones. G.3 requires at least one failed approach with analysis.
2. **Every Kaggle submission** → add a row to `submissions/submission_log.md` and save a screenshot in `results/leaderboard_evidence/`.

## Rules
- Never commit data, model weights, or large files (`.gitignore` covers common cases). Share large artifacts via the team Google Drive / Kaggle datasets.
- Clear notebook outputs that are very large before committing.
- Fix random seeds so results are reproducible.
- Add any new packages to `requirements.txt`.
