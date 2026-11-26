# Cyril: measured local results

Kaggle scores/screenshots must be added after actual submissions.

| Experiment | MRR@25 | Top-1 | Top-10 | Top-25 |
|---|---|---|---|---|
| baseline | 0.891360 | 0.8360 | 0.9760 | 0.9920 |
| E101 | 0.418836 | 0.2600 | 0.7560 | 0.8840 |
| E102 | 0.471017 | 0.2960 | 0.8120 | 0.9080 |
| E103 | 0.895501 | 0.8440 | 0.9800 | 0.9960 |

## E101: Binned-spectrum MLP

Training: 699550 spectra, 28 epochs; best epoch 25; Tesla T4; 931 seconds. Adam lr=0.001, batch=256, BCE loss; structure-disjoint 5% early stopping.
Checkpoint SHA256: `f8ae89d5ea96f8879e2300a3508f793b1cd7ef5dc353eb00741ecd0f0d541d10`.

## E102: MLP with neutral losses and metadata

Training: 699550 spectra, 30 epochs; best epoch 30; Tesla T4; 17837 seconds. Adam lr=0.001, batch=256, BCE loss; structure-disjoint 5% early stopping.
Checkpoint SHA256: `0a71ab91df104e762290135962680b2a8e18e391ed816d2d6fdf7a3104adf054`.

E103 uses E102 with alpha=0.2; selected on shared validation.

These scorers cannot recover structures absent from the candidate pool. Compare with the measured baseline before explaining any failure; validation overlap favors spectral library search.

## Experiment-log rows

| E101 | DATE | mrcyrilgoud | Binned-spectrum MLP | Substructure evidence improves ranking | 0.418836 | | Local validation only; interpretation pending | members/mrcyrilgoud/ |
| E102 | DATE | mrcyrilgoud | MLP with neutral losses and metadata | Substructure evidence improves ranking | 0.471017 | | Local validation only; interpretation pending | members/mrcyrilgoud/ |
| E103 | DATE | mrcyrilgoud | Validation-selected blend | Combine learned and library evidence | 0.895501 | | Local validation only; interpretation pending | members/mrcyrilgoud/ |
