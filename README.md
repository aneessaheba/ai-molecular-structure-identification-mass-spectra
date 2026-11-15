# Enveda CASMI 2026: Molecule ID From Mass Spectra

**DATA 266: Group Project (Fall 2026)**

Kaggle competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra

## Deadlines
| Due | What | Where |
|---|---|---|
| **Tue Dec 1, 3:00 PM** | **Group Project Report** (120 pts, IEEE/APA) | Canvas (late window closes Dec 8, 3:00 PM) |
| **Tue Dec 1, 3:00 PM** | **Group Project Presentation PPT** | Canvas (late window closes Dec 8, 3:00 PM) |
| **Tue Dec 1, 11:59 PM** | Peer Review (each member, individually) | Canvas (late window closes Dec 7, 11:59 PM) |
| Mon Dec 7, 11:59 PM UTC | Kaggle entry and team merge deadline | Kaggle |
| Mon Dec 14, 11:59 PM UTC | Kaggle final submission deadline | Kaggle |

**The report is due Dec 1, so any Kaggle scores we want in it must be in by about Nov 29.** The Kaggle Dec 14 deadline doesn't help us.

### Working plan
| Dates | Goal |
|---|---|
| Nov 14–17 | Split into Kaggle teams `GROUP_PROGRAMMING_01_a` / `_b`; assign roles; **baseline submitted for both teams**; EDA notebook |
| Nov 17–25 | Each member builds and submits 2–3 models (`members/<username>/`); log every experiment, including failures |
| Nov 25–28 | Final submissions; collect leaderboard screenshots; freeze results |
| Nov 26–30 | Write the report and slides (sections in `docs/report/` and `docs/presentation/`) |
| Dec 1 | Submit report + PPT by 3:00 PM; peer reviews by 11:59 PM |

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
