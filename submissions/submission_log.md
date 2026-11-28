# Kaggle Submission Log (G.4 / G.6)

Submit regularly throughout the project. Save a screenshot of each notable result in `results/leaderboard_evidence/`.

| # | Date | Kaggle Team (_a/_b) | Member | Model / Experiment ID | Public LB Score | Private LB (if shown) | Notes | Screenshot |
|---|------|--------------------|--------|-----------------------|-----------------|-----------------------|-------|------------|
| 1 | | | | Baseline (E002) | | | | |
| 2 | 2026-11-27 | _b | mrcyrilgoud | E103: E102 fingerprint MLP + library cosine, α 0.2 | 0.137 | | Kaggle ref 59022000; notebook `mrcyrilgoud/casmi-cyril-e103` v1 (CPU, internet off, 13 min). Public rank 5,522 of 6,150 on Nov 27. Local validation 0.8955; candidates come only from training structures, so Class 2/3 test molecules score 0 | [Submissions page](../results/leaderboard_evidence/cyril_team_b_submissions.png) |
| 3 | 2026-11-27 | _b | mrcyrilgoud | E102: fingerprint MLP + neutral losses + metadata (no library cosine) | 0.098 | | Kaggle ref 59037487, submitted 06:01 UTC Nov 28 (Nov 27 Pacific); notebook `mrcyrilgoud/casmi-cyril-e102` v1 (CPU, internet off, 12 min), same as the E103 notebook with only `EXPERIMENT` changed. Local validation 0.4710. Below E103 as expected for the fingerprint model alone; team best stays 0.137 | [Submissions page](../results/leaderboard_evidence/cyril_team_b_submissions.png) |
| 4 | 2026-11-27 | _b | mrcyrilgoud | E101: binned-spectrum fingerprint MLP (no library cosine) | 0.089 | | Kaggle ref 59038326, submitted 06:26 UTC Nov 28 (Nov 27 Pacific); notebook `mrcyrilgoud/casmi-cyril-e101` v1 (CPU, internet off, 12 min), same as the E103 notebook with only `EXPERIMENT` changed. Local validation 0.4188. Below E102 (0.098), matching the local order: neutral losses and metadata helped; team best stays 0.137 | [Submissions page](../results/leaderboard_evidence/cyril_team_b_submissions.png) |
| 5 | 2026-11-27 | _b | mrcyrilgoud | Baseline (E002): library search, greedy cosine | 0.135 | | Kaggle ref 59040129, submitted 07:05 UTC Nov 28 (Nov 27 Pacific); notebook `mrcyrilgoud/casmi-cyril-baseline` v1 (CPU, internet off, 10 min), the repo's `kaggle/library_search_baseline` notebook unchanged. Same score as team _a's baseline. E103 is +0.002 over it, inside leaderboard noise; both pick the same top-1 for all 400 test molecules | [Submissions page](../results/leaderboard_evidence/cyril_team_b_submissions.png) |

## Summary (fill in for each Kaggle team; the report needs both)
| | GROUP_PROGRAMMING_01_a | GROUP_PROGRAMMING_01_b |
|---|---|---|
| Baseline score | | 0.135 (E002 notebook, row 5) |
| Best score (by course cutoff) | | 0.137 (E103, row 2; as of 2026-11-27) |
| Improvement over baseline | | +0.002, inside the ±0.02 public-LB noise; E103 and the baseline pick the same top-1 for all 400 test molecules |
| Leaderboard rank / percentile | | 5,544 of 6,173 teams on 2026-11-27, ahead of 10.2% |
