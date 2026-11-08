# Main results (block protocol)

| Method | n | Block BA | Median | > chance | Cross-session BA | Drop | Cost | Features | Train s | Infer ms | Δ vs fbcsp_all | p (Holm) | r_rb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FBCSP all | 22 | 0.353 ± 0.044 | 0.348 | 2/42 | 0.337 ± 0.039 | +0.027 ± 0.085 | 1.00 | 168 | 0.2 | 0.40 | – | – | – |
| FBCSP all (viewing) | 22 | 0.435 ± 0.069 | 0.432 | 12/42 | – | – | 1.00 | 168 | 0.1 | – | -0.083 | <0.001 | -0.87 |

Balanced accuracy, chance = 0.333. Subject level: seeds averaged, then sessions (mean ± SD over subjects).
*> chance*: recordings with permutation p < 0.05 after Holm correction over recordings.
*Drop*: block-CV BA on ses-02 minus cross-session BA (train ses-01, test ses-02).
*Δ vs fbcsp_all*: mean paired difference (fbcsp_all − method); Wilcoxon signed-rank over subjects, Holm-corrected over the comparisons; r_rb = rank-biserial correlation.
*Cost*: (n_ch/8 + n_bands/4 + n_win/7)/3, – = not applicable (EEGNet). *Train s*: per result file, including GA search.
