# Random-vs-block inflation (subject level, seeds 0-2)

| Method | n | Block | Random | Random − block (median) | p | Subjects random > block |
|---|---|---|---|---|---|---|
| fbcsp_all | 22 | 0.353 | 0.349 | -0.004 (-0.002) | 0.849 | 10 |
| ga_nsga | 22 | 0.345 | 0.340 | -0.006 (-0.011) | 0.305 | 6 |
| ga_single | 22 | 0.353 | 0.343 | -0.011 (-0.006) | 0.092 | 8 |
| eegnet_32ch | 22 | 0.341 | 0.339 | -0.002 (-0.008) | 0.524 | 9 |
| eegnet_8ch | 22 | 0.347 | 0.342 | -0.005 (-0.009) | 0.702 | 9 |

# Viewing period (3-7 s) vs imagery, block protocol

| Method | n | Imagery | Viewing | Viewing − imagery | p | Viewing recordings p<.05 (uncorr./Holm) | Viewing cost |
|---|---|---|---|---|---|---|---|
| fbcsp_all | 22 | 0.353 | 0.435 | +0.083 | 0.000334 | 25/12 of 42 | 1.00 |
| ga_nsga | 22 | 0.345 | 0.410 | +0.065 | 2.05e-05 | 30/15 of 42 | 0.33 |
| ga_single | 22 | 0.353 | 0.428 | +0.074 | 0.000146 | 30/20 of 42 | 0.54 |

view_ga_nsga_vs_fbcsp: Δ -0.025 (median -0.028), p = 0.004, 6/22 subjects

view_ga_single_vs_fbcsp: Δ -0.008 (median -0.006), p = 0.388, 9/22 subjects

view_nsga_vs_single: Δ -0.018 (median -0.019), p = 0.042, 5/22 subjects
