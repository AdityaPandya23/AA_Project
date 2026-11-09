# Random-chromosome control at matched cost (50 draws per fold)

| Method | Protocol | n | GA BA | Random BA (same cost) | GA − random (p two-sided / greater) | Subjects GA > random | Fold percentile: mean, ≥ 0.95, ≤ 0.05 | Random − fbcsp_all (p) |
|---|---|---|---|---|---|---|---|---|
| ga_nsga | block | 22 | 0.345 | 0.341 | +0.004 (0.424 / 0.212) | 13/22 | 0.51, 6.9%, 5.0% | -0.011 (0.137) |
| ga_single | block | 22 | 0.353 | 0.347 | +0.007 (0.198 / 0.099) | 15/22 | 0.52, 6.7%, 4.8% | -0.006 (0.305) |
| ga_nsga | cross_session | 20 | 0.340 | 0.335 | +0.005 (0.312 / 0.156) | 12/20 | 0.55, 11.7%, 6.7% | -0.002 (0.784) |
| ga_single | cross_session | 20 | 0.328 | 0.335 | -0.007 (0.143 / 0.934) | 9/20 | 0.43, 5.0%, 11.7% | -0.002 (0.956) |
