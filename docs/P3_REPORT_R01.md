# P3 report (R01 share): GA runs for sub-09..16

R01 share, 2026-11-08. Code: `main` @ `6a38c4b`. The GA path (`src/ga.py`, `src/fbcsp.py`,
`src/cv.py`, `src/chromosome.py`, `experiments/run_ga.py`, `experiments/common.py`,
`experiments/_common.py`, `config.yaml`) is identical to `ef813ba`, which R02 used,
and to tag `p1-integrated`, so the GA operators frozen by §k2 apply. Config
`08e124167b`. 180 tests pass, 1 skipped (EEGNet/torch). Subject split per CONTRACTS
open item 12: R02 sub-01..08, R01 sub-09..16, R07 sub-17..22. **These are 8 of 22
subjects; the group result needs all three shares.**

The runs were done on Aditya's machine on behalf of R01 (Windows 11, Python 3.12,
8 logical CPUs). `AA_HOST=R01` was set, so `host` = `R01` labels the subject
share, not the hardware. P2_REVIEW §1 showed that the pipeline is bit-identical
across machines given the same cache.

Runs: `ga_nsga` and `ga_single`, block (14 recordings × 3 seeds × 4 folds;
sub-09 and sub-10 have ses-01 only) and cross-session (sub-11..16 × 3 seeds),
pop 30 × 20 generations. 372 fold jobs, 8.0 min wall time on 8 workers.
Results: `results/ga_nsga/`, `results/ga_single/` (60 JSONs each) and
`results/pareto/{method}/` (side files, §k6).

Reproduce (PowerShell: `$env:AA_HOST = "R01"`):

```
AA_HOST=R01 python -m experiments.run_ga --method ga_nsga   --subjects 09..16 --protocol block         --n-jobs 8   # 178 s
AA_HOST=R01 python -m experiments.run_ga --method ga_nsga   --subjects 09..16 --protocol cross_session --n-jobs 8   #  68 s
AA_HOST=R01 python -m experiments.run_ga --method ga_single --subjects 09..16 --protocol block         --n-jobs 8   # 182 s
AA_HOST=R01 python -m experiments.run_ga --method ga_single --subjects 09..16 --protocol cross_session --n-jobs 8   #  51 s
```

## 1. Balanced accuracy (sub-09..16, subject level, seeds then sessions averaged)

`fbcsp_all` is the P2 baseline restricted to the same subjects. On these subjects
it is **at chance**, below its 22-subject mean: 0.331 vs 0.353 block (n = 8),
0.319 vs 0.337 cross-session (n = 6). So the R01 share is the mirror image of
R02's, where the baseline was above its mean.

| Method | Block mean ± SD (n = 8) | Cross-session mean ± SD (n = 6) | Cost | Features | Infer ms/trial | Block recordings p < 0.05 (uncorr. / Holm, of 14) |
|---|---|---|---|---|---|---|
| `fbcsp_all` | 0.331 ± 0.045 | 0.319 ± 0.026 | 1.000 | 168 | 0.31 | 3 / 0 |
| `ga_single` | **0.347 ± 0.035** | 0.327 ± 0.036 | 0.466 | 37 | 0.08 | 3 / 0 |
| `ga_nsga` | 0.337 ± 0.023 | **0.349 ± 0.026** | **0.293** | **13** | **0.03** | 0 / 0 |

SD = sample SD over subjects. Paired Wilcoxon, two-sided, uncorrected (descriptive only; the
pre-registered tests in ANALYSIS_PLAN.md need all 22 subjects):

| Comparison | Block Δ (p, r_rb), n = 8 | Cross-session Δ (p, r_rb), n = 6 |
|---|---|---|
| `ga_nsga` − `fbcsp_all` | +0.006 (0.844, +0.11) | +0.030 (0.062, +0.90) |
| `ga_single` − `fbcsp_all` | +0.016 (0.461, +0.33) | +0.008 (0.562, +0.33) |
| `ga_nsga` − `ga_single` | −0.010 (0.484, −0.31) | +0.022 (0.844, +0.14) |

Per subject (block): `ga_nsga` beats `fbcsp_all` on 4 of 8 subjects (09, 10, 12,
15), and `ga_single` beats it on 4 of 8 (09, 10, 15, 16). The largest gain is sub-10
(ses-01 only: 0.278 → 0.374 / 0.392). The three recordings where `fbcsp_all`
is significant before correction (11/02, 13/02, 14/02) all lose accuracy
under `ga_nsga`, and no `ga_nsga` recording reaches p < 0.05. No recording
survives Holm over the 14 recordings for any method.

Cross-session: `ga_nsga` is higher than `fbcsp_all` on 5 of 6 subjects
(0.349 vs 0.319). With n = 6 this is not significant (p = 0.062), and the
baseline's 0.319 on these subjects is itself below chance.

**Reading:** on R01's share, neither GA clearly beats the full baseline. Both
are a little higher on average, but the differences (+0.006 to +0.030) are
the size of the seed-to-seed SD of a recording (0.036–0.041) and none is
significant. The baseline is at chance on these subjects, so the GA cannot
lose much here. Its small advantage looks more like a weak baseline than a
working selection: the recordings that are decodable with `fbcsp_all` get
worse under `ga_nsga`. As on R02, the real gain is cost: `ga_nsga` uses
about 8 % of the features (13 vs 168) and predicts about 12× faster
(0.026 vs 0.308 ms per trial) at no loss of accuracy on this share.

## 2. Why: the inner CV is optimistic and the selection is close to random

For the chosen chromosome of each fold, its inner-CV BA from the stored front
compared with the outer test BA:

| Method, protocol | Inner BA (chosen) | Outer BA | Optimism | Front size |
|---|---|---|---|---|
| `ga_nsga`, block | 0.513 | 0.337 | **+0.18** | 19.7 |
| `ga_single`, block | 0.569 | 0.347 | **+0.22** | 30 |
| `ga_nsga`, cross-session | 0.488 | 0.349 | +0.14 | 21.8 |
| `ga_single`, cross-session | 0.545 | 0.327 | +0.22 | 30 |

* Practically the same overfitting gap as R02 (+0.17 / +0.21 block). About
  600 evaluations against a 3-fold inner CV on about 90 trials find chromosomes
  that fit the inner folds by chance. The outer test is touched once per fold,
  so the outer numbers are honest; the inner numbers must not be reported as
  accuracy.
* **Seed variability:** the SD of a recording's BA across the 3 seeds is 0.036
  (`ga_nsga`) and 0.041 (`ga_single`) for block, which is larger than every
  block difference in §1.
* **Mean selection (block):** `ga_nsga` 2.6 channels, 1.3 bands, 1.7 windows;
  `ga_single` 3.7 channels, 2.0 bands, 3.0 windows. Cross-session is similar.
* **Selection frequency (block, 168 folds):** almost flat. `ga_nsga` selects
  every gene in 14–47 % of folds. The only clearly preferred gene is **Pz
  (47 %)**, then Oz (39 %). Bands are 27–35 %, windows 14–30 % with no
  preferred window. R02 saw a preference for 1.5–2.5 s, which does not
  replicate here. `ga_single` is at 34–55 % everywhere, close to the
  Bernoulli(0.5) start. Top 5: `ga_nsga` Pz 47 %, Oz 39 %, alpha 35 %,
  beta 34 %, P4 32 %; `ga_single` Oz 55 %, lowgamma 55 %, alpha 54 %, 0.0 s 54 %, Pz 52 %.
* **Selection stability of `chosen_bits` (Kuncheva / Jaccard, mean over recordings):**

  | Method, protocol | Folds and seeds (all pairs) | Same fold, different seed | Same seed, different fold |
  |---|---|---|---|
  | `ga_nsga`, block | **0.082** / 0.23 | 0.156 / 0.29 | 0.065 / 0.22 |
  | `ga_single`, block | **0.081** / 0.36 | 0.275 / 0.51 | 0.049 / 0.34 |
  | `ga_nsga`, cross-session (seeds only) | 0.062 / 0.25 | | |
  | `ga_single`, cross-session (seeds only) | 0.269 / 0.53 | | |

  Block values match R02 (0.083 / 0.073). Near 0 = no more agreement than
  chance. Stability across seeds on the same training data is higher than
  across folds, so what the GA picks depends more on which trials it is
  trained on than on its random seed. Being stability-aware does not make
  `ga_nsga`'s selection more stable than `ga_single`'s.

Same conclusion as R02: with a weak signal, the GA is fitting noise in the inner CV.

## 3. Points for P3 integration and P5

1. **Shares differ in direction:** R02's share has GA − `fbcsp_all` < 0
   (−0.022 block for `ga_nsga`), and R01's has > 0 (+0.006). The two shares
   also differ in baseline level (0.367 vs 0.331). Both point to a group
   effect near zero. The pre-registered primary test (ANALYSIS_PLAN §2) needs
   all 22 subjects; don't read either share on its own.
2. **Random-chromosome control (R02 §3.2):** supported. A control at matched
   cost would show whether `ga_nsga` adds anything beyond the cost reduction.
3. **Analysis glitch confirmed:** `analysis.load_results("results")` raises on
   `results/sanity/sanity_summary.json`. Passing `results/ga_nsga` does not
   work either, because `load_results` globs `{dir}/*/*.json`. This report
   loaded a scratch copy holding only `ga_nsga/`, `ga_single/` and
   `pareto/`. A fix in `load_results` (skip `*_summary.json`) is R01's call.
4. **Checkpoints are kept:** `run_ga` does not delete
   `results/raw/partial/*` after writing the final JSON (it only clears them
   with `--overwrite`). All 372 checkpoints of this share match their final
   records, and none is orphaned. They are git-ignored and can be deleted by
   hand. The P3 prompts' "nothing left in `raw/partial`" check does not
   hold by design.
5. `git_commit` is `6a38c4b` here and `ef813ba` for R02. The GA path is identical
   between the two (empty diff), so the shares are comparable. `analysis`
   should not treat the two commits as different code.

## 4. Status / next

* R01 share of P3: done. 120/120 results and 120/120 side files, integrity
  check clean (§3.4 aside). Results committed with this report.
* Waiting for R07 (sub-17..22). Then: group tables and the pre-registered
  tests (ANALYSIS_PLAN.md), and the k12 re-examination.
