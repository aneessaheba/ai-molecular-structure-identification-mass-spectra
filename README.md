# Enveda CASMI 2026: Molecule ID From Mass Spectra

**DATA 266: Group Project (Fall 2026)**

Kaggle competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra

## Problem
Predict the 2D chemical structure of unknown small molecules from tandem mass spectrometry (MS/MS) data. Combine evidence across spectra and output a **ranked list of candidate molecular structures** for each unknown molecule.

## Repository Structure
```
├── data/                  # NOT committed (see data/README.md)
│   ├── raw/               # Original Kaggle download
│   ├── processed/         # Cleaned / featurized data
│   └── external/          # Allowed external data (check competition rules)
├── notebooks/
│   ├── eda/               # Shared exploratory data analysis (G.2)
│   └── baseline/          # Team baseline notebook (G.1 / G.2)
├── src/                   # Shared, reusable code
│   ├── data/              # Loading & preprocessing spectra
│   ├── features/          # Spectrum / molecule featurization
│   ├── models/            # Model architectures
│   ├── evaluation/        # Local validation & competition metric
│   └── utils/
├── configs/               # Experiment configs (YAML)
├── scripts/               # CLI scripts: train, predict, make submission
├── members/<github-username>/  # Each member's own models (2–3 architectures each)
├── experiments/
│   └── experiment_log.md  # Every experiment: what changed, why, result (G.3)
├── submissions/
│   ├── submission_log.md  # Every Kaggle submission and its score (G.4)
│   └── files/             # Submission CSVs (git-ignored)
├── results/
│   ├── figures/           # Plots for report / slides
│   └── leaderboard_evidence/  # Kaggle screenshots (G.4 / G.6)
└── docs/
    ├── competition_summary.md  # Task, data, metric, rules, format (G.1)
    ├── team_roles.md           # Roles + contribution table
    ├── proposal/               # One-page abstract
    ├── report/                 # Final report
    ├── presentation/           # Final slides
    └── peer_review/            # Reminder only, forms are submitted individually
```

## Deliverables → Where They Live
| Deliverable (from brief) | Location | Rubric |
|---|---|---|
| One-page proposal / abstract | `docs/proposal/` | — |
| Competition understanding & setup | `docs/competition_summary.md`, `notebooks/baseline/` | G.1 |
| EDA, preprocessing, baseline | `notebooks/eda/`, `src/`, `notebooks/baseline/` | G.2 |
| Improvement experiments (incl. ≥1 failed approach) | `members/`, `experiments/experiment_log.md` | G.3 |
| Kaggle submission history & evidence | `submissions/submission_log.md`, `results/leaderboard_evidence/` | G.4 |
| Final report & presentation | `docs/report/`, `docs/presentation/` | G.5 |
| Best score & improvement over baseline | `submissions/submission_log.md`, `docs/report/` | G.6 |
| Contribution table | `docs/team_roles.md` → copied into final report | — |

## Getting Started
```bash
git clone https://github.com/aneessaheba/ai-molecular-structure-identification-mass-spectra.git
cd ai-molecular-structure-identification-mass-spectra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download data (requires Kaggle API token in ~/.kaggle/kaggle.json)
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data/raw
unzip data/raw/*.zip -d data/raw
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team workflow.

## Team
See [docs/team_roles.md](docs/team_roles.md).
