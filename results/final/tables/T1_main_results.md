## T1. Main results (pre-registered methods)

| Method | n | Block BA (mean ± SD) | Median | Recordings > chance (Holm) | Cross-session BA (n = 20) | Drop (ses-02 − cross) | Cost | Features / params |
|---|---|---|---|---|---|---|---|---|
| FBCSP all | 22 | 0.353 ± 0.044 | 0.348 | 2/42 | 0.337 ± 0.039 | +0.027 ± 0.085 | 1.00 | 168 |
| GA-NSGA (proposed) | 22 | 0.345 ± 0.025 | 0.347 | 2/42 | 0.340 ± 0.027 | +0.002 ± 0.042 | 0.30 | 13 |
| GA single-obj. | 22 | 0.353 ± 0.029 | 0.347 | 3/42 | 0.328 ± 0.027 | +0.028 ± 0.045 | 0.48 | 39 |
| EEGNet (32 ch) | 22 | 0.341 ± 0.028 | 0.342 | 2/42 | 0.325 ± 0.029 | +0.013 ± 0.037 | – | 3107 |
| EEGNet (8 ch) | 22 | 0.347 ± 0.031 | 0.341 | 1/42 | 0.334 ± 0.033 | +0.004 ± 0.040 | – | 2723 |

Balanced accuracy, chance = 0.333. Subject level (seeds, then sessions); mean ± sample SD over subjects. Recordings > chance: label-permutation p < 0.05 after Holm over the 42 recordings. Drop = ses-02 block BA − cross-session BA. Cost = (n_ch/8 + n_bands/4 + n_win/7)/3; – = not applicable. Timings are omitted (measured on three different machines).
