# Analysis plan (pre-registered before P3)

R02, 2026-11-08. Written on branch `results-r02` from `main` @ `6a38c4b`
(`6a38c4ba9f658051ae84d8d2cd8cfb4abb76ac22`), config `08e124167b`, tag
`p1-integrated` in history. At the time of writing **no P3 GA result exists**:
there are none on R02 and none on any branch of `origin` (R07's single-recording
budget check on sub-17 is not part of the analysed set). This plan is committed
and pushed on its own, before the runs, so that its timestamp comes before the
results.

Methods and result locations follow CONTRACTS §i and §k: `ga_nsga`, `ga_single`
in `results/{method}/`, `fbcsp_all` in `results/baseline_all/fbcsp_all/`,
`eegnet_32ch` / `eegnet_8ch` from P4. Metric: balanced accuracy (BA), chance
1/3.

## 1. Unit of analysis

One score per subject (CONTRACTS §k4): average the seeds (0, 1, 2), then the
sessions, i.e. `analysis.subject_level`. Block protocol: n = 22.
Cross-session (train ses-01 → test ses-02): n = 20 (sub-09 and sub-10 have no
ses-02). `fbcsp_all` is deterministic and has seed 0 only.

Completeness rule: a subject enters a comparison only if all expected files of
both methods exist (block: 2 sessions × 3 seeds, or 1 session for sub-09/10;
cross-session: 3 seeds). Missing files are listed in RESULTS.md and are not
imputed. A subject is not dropped for any other reason (no outlier removal).

## 2. Primary hypothesis

**`ga_nsga` > `fbcsp_all`, block protocol, subject level (n = 22).**

* Test: Wilcoxon signed-rank on the paired differences `ga_nsga − fbcsp_all`,
  **one-sided** (`alternative="greater"`), `zero_method="wilcox"`, α = 0.05.
* `analysis.paired_test` is two-sided, so the one-sided p is computed with
  `scipy.stats.wilcoxon(d, zero_method="wilcox", alternative="greater")` on the
  same `subject_level` differences. The statistic, mean and median difference
  and rank-biserial r are taken from `paired_test`.
* Reported: n, mean and median difference, one-sided p, rank-biserial r.
* Sensitivity: P2_REVIEW §5 estimates a minimum detectable difference of
  about **0.023 BA** (80 % power, one-sided α = 0.05, assuming SD of the
  differences ≈ 0.044). The realised SD of the differences is reported next to
  it. A non-significant result means "no evidence of a gain of this size", not
  "no gain".

## 3. Secondary comparisons (one Holm family, m = 4)

Two-sided Wilcoxon signed-rank (`analysis.paired_test`), subject level,
Holm-corrected together (`analysis.holm`):

| # | Comparison | Protocol | n |
|---|---|---|---|
| S1 | `ga_nsga` vs `ga_single` | block | 22 |
| S2 | `ga_nsga` vs `fbcsp_all` | cross-session | 20 |
| S3 | `ga_nsga` vs `eegnet_32ch` | block | 22 |
| S4 | `ga_nsga` vs `eegnet_8ch` | block | 22 |

* S2 is also reported **without sub-01** (n = 19), because sub-01 is the only
  subject whose two sessions share the trial sequence (DATA_NOTES §4,
  P2_REPORT §3). This is a sensitivity analysis outside the Holm family.
* If P4 does not deliver an EEGNet variant, its test is dropped and the family
  shrinks accordingly. This is stated in RESULTS.md.

## 4. Descriptive (no tests)

* BA mean ± SD (sample SD) per method and protocol; per-recording
  above-chance counts (`analysis.above_chance`, uncorrected / Holm over
  recordings).
* Cost (`(n_ch/8 + n_bands/4 + n_win/7)/3`), `n_features`, and the mean number of
  selected channels, bands and windows per method.
* Gene-selection frequency (`analysis.selection_frequency`).
* Selection stability of `chosen_bits` (`analysis.selection_stability`):
  Jaccard and Kuncheva, `by="all"`, `"seed"` and `"fold"`, for `ga_nsga` vs
  `ga_single`.
* Cross-session drop with `analysis.cross_session_drop(within_session="02")`
  (ses-02 block BA − cross-session BA, P2_REVIEW §3.2), with and without
  sub-01.
* Random-vs-block inflation: for `fbcsp_all` it is in P2 (−0.004). The frozen GA
  and EEGNet runners only support `block` and `cross_session`, so this cannot be
  estimated for them in P3/P4. It is reported as a gap and not added.
* Position-only control for cross-session (R07, P5).

## 5. Final chromosome rule

CONTRACTS §g as written, with the §k8 tie-break: `b*` is the highest inner-CV
`ba_mean`. The candidates are the Pareto entries with
`ba_mean ≥ b*.ba_mean − b*.ba_std`, and the lowest-cost candidate is chosen. All
results above use this rule. Any revision of the 1-SD rule (§k12), e.g.
re-choosing from the stored Pareto fronts, is **exploratory only**. It is
labelled as such and never replaces the primary or secondary results.

## 6. Deviations

Any deviation from this plan is listed in RESULTS.md with its reason. Analyses
not named here are exploratory.
