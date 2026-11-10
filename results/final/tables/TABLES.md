# Final tables

## T1. Main results (pre-registered methods)

| Method | n | Block BA (mean ± SD) | Median | Recordings > chance (Holm) | Cross-session BA (n = 20) | Drop (ses-02 − cross) | Cost | Features / params |
|---|---|---|---|---|---|---|---|---|
| FBCSP all | 22 | 0.353 ± 0.044 | 0.348 | 2/42 | 0.337 ± 0.039 | +0.027 ± 0.085 | 1.00 | 168 |
| GA-NSGA (proposed) | 22 | 0.345 ± 0.025 | 0.347 | 2/42 | 0.340 ± 0.027 | +0.002 ± 0.042 | 0.30 | 13 |
| GA single-obj. | 22 | 0.353 ± 0.029 | 0.347 | 3/42 | 0.328 ± 0.027 | +0.028 ± 0.045 | 0.48 | 39 |
| EEGNet (32 ch) | 22 | 0.341 ± 0.028 | 0.342 | 2/42 | 0.325 ± 0.029 | +0.013 ± 0.037 | – | 3107 |
| EEGNet (8 ch) | 22 | 0.347 ± 0.031 | 0.341 | 1/42 | 0.334 ± 0.033 | +0.004 ± 0.040 | – | 2723 |

Balanced accuracy, chance = 0.333. Subject level (seeds, then sessions); mean ± sample SD over subjects. Recordings > chance: label-permutation p < 0.05 after Holm over the 42 recordings. Drop = ses-02 block BA − cross-session BA. Cost = (n_ch/8 + n_bands/4 + n_win/7)/3; – = not applicable. Timings are omitted (measured on three different machines).

## T2. Pre-registered tests (ANALYSIS_PLAN §2–3)

| Test | Comparison | Protocol | n | Mean Δ | Median Δ | p | p (Holm) | r_rb |
|---|---|---|---|---|---|---|---|---|
| Primary | GA-NSGA > FBCSP all (one-sided) | block | 22 | -0.007 | -0.004 | 0.788 | – | -0.19 |
| S1 | GA-NSGA (proposed) vs GA single-obj. | block | 22 | -0.008 | -0.006 | 0.256 | 1.000 | -0.28 |
| S2 | GA-NSGA (proposed) vs FBCSP all | cross-session | 20 | +0.003 | +0.008 | 0.616 | 1.000 | +0.13 |
| S3 | GA-NSGA (proposed) vs EEGNet (32 ch) | block | 22 | +0.004 | -0.001 | 0.975 | 1.000 | -0.01 |
| S4 | GA-NSGA (proposed) vs EEGNet (8 ch) | block | 22 | -0.002 | -0.001 | 1.000 | 1.000 | -0.00 |
| S2 sensitivity | S2 without sub-01 | cross-session | 19 | +0.006 | +0.008 | 0.463 | outside family | +0.20 |

Wilcoxon signed-rank over subjects. Primary: one-sided; secondary: two-sided, Holm over S1–S4. Δ = first − second method.

## T3. GA selection and stability (block)

| Method | Cost | Channels / bands / windows | Gene frequency range | Top 3 genes | Kuncheva: all / same fold / same seed |
|---|---|---|---|---|---|
| GA-NSGA (proposed) | 0.30 | 2.7 / 1.3 / 1.6 | 18–43 % | Pz 43 %, Oz 38 %, P7 35 % | 0.078 / 0.163 / 0.072 |
| GA single-obj. | 0.48 | 3.8 / 2.0 / 3.2 | 38–54 % | alpha 54 %, Pz 54 %, Oz 51 % | 0.084 / 0.254 / 0.058 |

## T4. Exploratory checks (P5)

| § | Check | Result |
|---|---|---|
| 1 | Position-only control (cross-session) | Position predicts class: 0.397 (p < 0.001); without sub-01 0.390. Shared-block subjects 0.449 vs 0.345; methods unchanged (FBCSP all 0.337 vs 0.337, p = 0.79) |
| 2 | Random chromosomes at matched cost: GA-NSGA (proposed) | GA 0.345 vs random 0.341: Δ +0.004 (p = 0.42); mean percentile 0.51 |
| 2 | Random chromosomes at matched cost: GA single-obj. | GA 0.353 vs random 0.347: Δ +0.007 (p = 0.20); mean percentile 0.52 |
| 3 | Chooser rules (§k12), block | Outer BA 0.338–0.351 across 7 rules; vs FBCSP all -0.014 to -0.002 (all p ≥ 0.28) |
| 4 | Random 5-fold vs block | FBCSP all -0.004, GA-NSGA (proposed) -0.006, GA single-obj. -0.011, EEGNet (32 ch) -0.002, EEGNet (8 ch) -0.005 (all p ≥ 0.09) |
| 4 | Viewing period: GA-NSGA vs FBCSP all | 0.410 vs 0.435: Δ -0.025 (p = 0.004) |
| 5 | Training vs test BA (block) | FBCSP all 0.999 vs 0.353; GA single-obj. 0.751 vs 0.353; GA-NSGA (proposed) 0.576 vs 0.345 |
| 5 | EEGNet training diagnostic (block, seed 0) | Training BA 0.64 (P4 recipe) → 1.00 (300 epochs); test 0.350 → 0.327 (p = 0.012) |

## T5. Training vs inner-CV / validation vs test BA (P5 §5)

| Method | Protocol | Training BA | Inner-CV BA | Test BA | Training − test |
|---|---|---|---|---|---|
| FBCSP all | block | 0.999 | 0.342 | 0.353 | +0.647 |
| GA single-obj. | block | 0.751 | 0.568 | 0.353 | +0.398 |
| GA-NSGA (proposed) | block | 0.576 | 0.512 | 0.345 | +0.231 |
| EEGNet (32 ch) (seed 0) | block | 0.639 | val 0.364 | 0.350 | +0.290 |
| EEGNet (8 ch) (seed 0) | block | 0.648 | val 0.373 | 0.350 | +0.298 |
| FBCSP all | cross-session | 0.997 | 0.361 | 0.337 | +0.660 |
| GA single-obj. | cross-session | 0.753 | 0.552 | 0.328 | +0.425 |
| GA-NSGA (proposed) | cross-session | 0.547 | 0.491 | 0.340 | +0.207 |
| EEGNet (32 ch) (seed 0) | cross-session | 0.762 | val 0.372 | 0.330 | +0.432 |
| EEGNet (8 ch) (seed 0) | cross-session | 0.638 | val 0.407 | 0.330 | +0.307 |

Training BA = resubstitution on the outer training set (FBCSP methods) or the fit part of it (EEGNet); EEGNet's middle column is its validation BA.

## T6. Viewing period vs imagery (P5 §4.2)

| Method | Imagery BA | Viewing BA | Viewing − imagery | p | Viewing recordings > chance (Holm) | Cost |
|---|---|---|---|---|---|---|
| FBCSP all | 0.353 | 0.435 ± 0.069 | +0.083 | <0.001 | 12/42 | 1.00 |
| GA-NSGA (proposed) | 0.345 | 0.410 ± 0.048 | +0.065 | <0.001 | 15/42 | 0.33 |
| GA single-obj. | 0.353 | 0.428 ± 0.074 | +0.074 | <0.001 | 20/42 | 0.54 |
