# Hand-off: @Mrnidhi

> Fill this in by **Wed Nov 25, 9 PM**. Anees copies it into the report and slides. Your task: [TASK.md](TASK.md)

## Models
| # | Model | File | Local MRR@25 | Top-1 | Top-10 | hide_true MRR (if used) | Kaggle score | Exp. ID |
|---|---|---|---|---|---|---|---|---|
| 0a | Original baseline (library cosine, 50 refs, top-64) | kaggle/library_search_baseline | 0.877 | 0.816 | 0.968 | 0.000 | | E002 |
| 0b | Tuned library search (all refs, top-32) = `CosineScorer` / `cosine` column | src/models/base.py | 0.929 | 0.892 | 0.984 | 0.000 | | E006 |
| 1 | | | | | | | | |
| 2 | | | | | | | | |
| 3 | | | | | | | | |

## Model 1: <name>
- **What it is (2–3 sentences):**
- **Input → output:**
- **Architecture:**
- **Training:** data size, epochs, optimizer, lr, batch size, hardware, training time
- **Result and interpretation:**

## Model 2: <name>
- **What it is:**
- **What changed vs Model 1, and why:**
- **Architecture / training:**
- **Result and interpretation:**

## Model 3 (optional): <name>

## What didn't work, and why (required)
- **What I tried:**
- **What happened (numbers):**
- **Why I think it failed:**
- **What I learned / what I'd do next:**

## Figure
![](figure.png)
_One plot: a training curve, or your models vs the baseline._

## Kaggle evidence
Screenshots in `results/leaderboard_evidence/`:
