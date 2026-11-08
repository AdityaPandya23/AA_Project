# P3 report (R02 share): GA runs for sub-01..08

R02, 2026-11-08. Code: `main` @ `ef813ba` (`src/`, `experiments/` and `config.yaml`
identical to tag `p1-integrated`, so the GA operators frozen by §k2 apply). Config
`08e124167b`. 170 tests pass. Subject split per CONTRACTS §k12: R02 sub-01..08,
R01 sub-09..16, R07 sub-17..22. **These are 8 of 22 subjects; the group result
needs all three shares.**

Runs: `ga_nsga` and `ga_single`, block (16 recordings × 3 seeds × 4 folds) and
cross-session (8 subjects × 3 seeds), pop 30 × 20 generations. 432 fold jobs,
7.5 min wall time on 10 workers. Results: `results/ga_nsga/`, `results/ga_single/`
(72 JSONs each) and `results/pareto/{method}/` (side files, §k6).

Reproduce:

```
AA_HOST=R02 python -m experiments.run_ga --method ga_nsga   --subjects 01..08 --protocol block         --n-jobs 10
AA_HOST=R02 python -m experiments.run_ga --method ga_nsga   --subjects 01..08 --protocol cross_session --n-jobs 10
AA_HOST=R02 python -m experiments.run_ga --method ga_single --subjects 01..08 --protocol block         --n-jobs 10
AA_HOST=R02 python -m experiments.run_ga --method ga_single --subjects 01..08 --protocol cross_session --n-jobs 10
```

## 1. Balanced accuracy (sub-01..08, subject level, seeds then sessions averaged)

`fbcsp_all` is the P2 baseline restricted to the same 8 subjects. On these
subjects it is slightly above its 22-subject mean (0.367 vs 0.353 block,
0.360 vs 0.337 cross-session).

| Method | Block mean ± SD | Cross-session mean ± SD | Cost | Features | Recordings p < 0.05 (uncorr.) |
|---|---|---|---|---|---|
| `fbcsp_all` | **0.367 ± 0.038** | **0.360 ± 0.036** | 1.000 | 168 | 4 / 16 |
| `ga_single` | 0.354 ± 0.030 | 0.334 ± 0.020 | 0.475 | 38 | 2 / 16 |
| `ga_nsga` | 0.345 ± 0.031 | 0.328 ± 0.026 | **0.286** | **12** | 2 / 16 |

Paired Wilcoxon, n = 8 subjects (uncorrected):

| Comparison | Block Δ (p, r_rb) | Cross-session Δ (p, r_rb) |
|---|---|---|
| `ga_nsga` − `fbcsp_all` | −0.022 (0.109, −0.67) | −0.031 (0.078, −0.79) |
| `ga_single` − `fbcsp_all` | −0.013 (0.148, −0.61) | −0.026 (0.078, −0.72) |
| `ga_nsga` − `ga_single` | −0.009 (0.547, −0.28) | −0.005 (0.844, +0.11) |

Per subject (block): `ga_nsga` beats `fbcsp_all` on 2 of 8 subjects (03, 04),
and `ga_single` beats it on 3 of 8 (01, 06, 08). The two clearly decodable
recordings, 02/02 and 03/01, stay significant for both GAs (p = 0.001,
Holm over R02's 16 recordings p = 0.016).

**Reading:** on R02's share, neither GA beats the full baseline; both are
somewhat worse (not significant at n = 8). `ga_nsga` achieves this with about
7 % of the features (12 vs 168) and a 16× faster prediction (0.04 vs 0.59 ms
per trial). So the trade-off is accuracy against cost, not a stability gain.

## 2. Why: the inner CV is optimistic and the selection is close to random

For the chosen chromosome of each fold, its inner-CV BA from the stored front
compared with the outer test BA:

| Method, protocol | Inner BA (chosen) | Outer BA | Optimism | Front size |
|---|---|---|---|---|
| `ga_nsga`, block | 0.512 | 0.345 | **+0.17** | 18.9 |
| `ga_single`, block | 0.567 | 0.354 | **+0.21** | 30 |
| `ga_nsga`, cross-session | 0.487 | 0.328 | +0.16 | 23.3 |
| `ga_single`, cross-session | 0.553 | 0.334 | +0.22 | 30 |

* About 600 evaluations against a 3-fold inner CV on about 90 trials:
  the GA finds chromosomes that fit the inner folds by chance. The
  outer-fold results do not carry that over. The selection is properly nested
  (the outer test is touched once), so the outer numbers are honest; the inner
  numbers must not be reported as accuracy.
* **Seed variability:** the SD of a recording's BA across the 3 seeds is 0.037
  (`ga_nsga`) and 0.029 (`ga_single`), about the size of the gap to the baseline.
* **Selection frequency (block, 192 folds):** almost flat. `ga_nsga` selects
  every channel and band in 23–36 % of folds (cost pressure pulls them all
  down evenly). The only window above the rest is **1.5–2.5 s** (37 % vs
  16–26 %). `ga_single` is at about 40–55 % everywhere, close to the
  Bernoulli(0.5) start.
* **Selection stability across folds and seeds (block):** Kuncheva 0.083
  (`ga_nsga`), 0.073 (`ga_single`). Jaccard 0.23 and 0.36. Near 0 = no
  more agreement than chance.

The GA is fitting noise in the inner CV. That fits P2: with a weak signal
(0.356 group BA), there is little that 3-fold CV over 90 trials can
reliably select.

## 3. Points for P3 integration and P5

1. **§k12 "within 1 SD" rule:** with inner SDs of this size, the rule
   mostly picks low-cost points. That is why `ga_nsga` has cost 0.29. Since
   the inner accuracy is so optimistic, a different rule would not fix
   outer accuracy. Re-examine it on all 22 subjects from the side files,
   as planned.
2. **Proposed control (P5):** random valid chromosomes at the same cost as
   `ga_nsga`, scored the same way. If they match `ga_nsga`, the search adds
   nothing beyond the cost reduction.
3. **Analysis glitch:** `analysis.load_results("results")` raises on
   `results/sanity/sanity_summary.json` (not a result record), so the GA dirs
   cannot be loaded from the results root together with `sanity/`. For this
   report the GA dirs were loaded from a copy. Either pass a dir that holds
   only method folders, or let `load_results` skip `*_summary.json` (R01's call).
4. sub-01 cross-session (P2 §3 flag): 0.406 for `fbcsp_all`, 0.366 for
   `ga_nsga` and 0.318 for `ga_single`. The GA does not reproduce the
   sub-01 effect.

## 4. Status / next

* R02 share of P3: done. Results committed with this report.
* Waiting for R01 (sub-09..16) and R07 (sub-17..22). Then: group tables with
  `reports.make_tables` (reference `ga_nsga`), Friedman over the three methods,
  and the k12 re-examination.
