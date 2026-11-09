# Main results (block protocol)

| Method | n | Block BA | Median | > chance | Cross-session BA | Drop | Cost | Features | Train s | Infer ms | Δ vs ga_nsga | p (Holm) | r_rb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GA-NSGA (proposed) | 22 | 0.345 ± 0.025 | 0.347 | 2/42 | 0.340 ± 0.027 | +0.002 ± 0.042 | 0.30 | 13 | 21.5 | 0.03 | – | – | – |
| FBCSP all | 22 | 0.353 ± 0.044 | 0.348 | 2/42 | 0.337 ± 0.039 | +0.027 ± 0.085 | 1.00 | 168 | 0.2 | 0.40 | -0.007 | 1.000 | -0.19 |
| GA single-obj. | 22 | 0.353 ± 0.029 | 0.347 | 3/42 | 0.328 ± 0.027 | +0.028 ± 0.045 | 0.48 | 39 | 25.4 | 0.08 | -0.008 | 1.000 | -0.28 |
| EEGNet (32 ch) | 22 | 0.341 ± 0.028 | 0.342 | 2/42 | 0.325 ± 0.029 | +0.013 ± 0.037 | – | 3107 | 14.8 | 0.12 | +0.004 | 1.000 | -0.01 |
| EEGNet (8 ch) | 22 | 0.347 ± 0.031 | 0.341 | 1/42 | 0.334 ± 0.033 | +0.004 ± 0.040 | – | 2723 | 4.8 | 0.06 | -0.002 | 1.000 | -0.00 |

Balanced accuracy, chance = 0.333. Subject level: seeds averaged, then sessions (mean ± SD over subjects).
*> chance*: recordings with permutation p < 0.05 after Holm correction over recordings.
*Drop*: block-CV BA on ses-02 minus cross-session BA (train ses-01, test ses-02).
*Δ vs ga_nsga*: mean paired difference (ga_nsga − method); Wilcoxon signed-rank over subjects, Holm-corrected over the comparisons; r_rb = rank-biserial correlation.
*Cost*: (n_ch/8 + n_bands/4 + n_win/7)/3, – = not applicable (EEGNet). *Train s*: per result file, including GA search.
Friedman test over methods: χ² = 1.20, p = 0.878, n = 22.
