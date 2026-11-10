# Results: pre-registered P3/P4 analysis (all 22 subjects)

2026-11-09. This follows `docs/ANALYSIS_PLAN.md` (R02, pushed 2026-11-08 22:49 +0530,
`main` @ `6a38c4b`) section by section. Data: `results/ga_nsga/`, `results/ga_single/`
(R02 sub-01..08, R01 sub-09..16, R07 sub-17..22), `results/eegnet_32ch/`,
`results/eegnet_8ch/` (R07), `results/baseline_all/fbcsp_all/` (P2). Config
`08e124167b` for every file. Code for all GA and EEGNet runs is identical to
`p1-integrated`.

Reproduce:

```
python -m reports.p3_numbers <scratch_dir>      # every number in this file
python -m reports.make_tables --results <scratch_dir>/ana --out results/tables/p3 --reference ga_nsga
python -m src.analysis --results <scratch_dir>/ana --out results/figures/p3 --protocol block
python -m src.analysis --results <scratch_dir>/ana --out results/figures/p3 --protocol cross_session
```

`reports/p3_numbers.py` copies the method folders into `<scratch_dir>/ana`, because
`analysis.load_results("results")` raises on `results/sanity/sanity_summary.json`
(R01 and R02 reports, §3).

## 1. Unit of analysis and completeness

One score per subject: seeds (0, 1, 2) averaged, then sessions
(`analysis.subject_level`). Block n = 22, cross-session n = 20 (sub-09 and sub-10
have no ses-02).

**Completeness rule: no file is missing** for any of the five methods (block:
42 recordings × 3 seeds, `fbcsp_all` seed 0 only; cross-session: 20 subjects ×
3 seeds). No subject was dropped, and nothing was imputed.

## 2. Primary hypothesis: `ga_nsga` > `fbcsp_all`, block, n = 22

| n | Mean Δ | Median Δ | W⁺ | One-sided p | r_rb | Subjects with Δ > 0 |
|---|---|---|---|---|---|---|
| 22 | **−0.007** | −0.004 | 102 | **0.788** | −0.19 | 10 / 22 |

Δ = `ga_nsga` − `fbcsp_all`, BA. One-sided Wilcoxon signed-rank
(`alternative="greater"`, `zero_method="wilcox"`) as specified in the plan.

**The primary hypothesis is not supported.** `ga_nsga` is not more accurate than
the full FBCSP baseline; the mean difference is slightly negative.

Sensitivity: the plan assumed an SD of the differences of 0.044, giving a minimum
detectable difference of about 0.023 BA (80 % power, one-sided α = 0.05). **The
realised SD is 0.045**, so the MDD is about 0.024. So: no evidence of a gain of
0.024 BA or more. *Exploratory, not in the plan:* the paired t-interval for the
mean difference is −0.027 to +0.013 (95 %). A loss of up to about 0.03 BA cannot
be excluded either.

## 3. Secondary comparisons (one Holm family, m = 4)

Two-sided Wilcoxon signed-rank (`analysis.paired_test`), Holm over the four
(`analysis.holm`). Both EEGNet variants were delivered, so the family is complete.

| # | Comparison | Protocol | n | Mean Δ | Median Δ | p | p (Holm) | r_rb |
|---|---|---|---|---|---|---|---|---|
| S1 | `ga_nsga` vs `ga_single` | block | 22 | −0.008 | −0.006 | 0.256 | 1.000 | −0.28 |
| S2 | `ga_nsga` vs `fbcsp_all` | cross-session | 20 | +0.003 | +0.008 | 0.616 | 1.000 | +0.13 |
| S3 | `ga_nsga` vs `eegnet_32ch` | block | 22 | +0.004 | −0.001 | 0.975 | 1.000 | −0.01 |
| S4 | `ga_nsga` vs `eegnet_8ch` | block | 22 | −0.002 | −0.001 | 1.000 | 1.000 | −0.00 |

Δ = first − second method. **No secondary comparison is significant.**
S2 without sub-01 (sensitivity, outside the family): n = 19, Δ = +0.006,
p = 0.463, r_rb = +0.20.

## 4. Descriptive (no tests)

### 4.1 Accuracy, cost and above-chance recordings

| Method | Block BA mean ± SD (n = 22) | Cross-session BA (n = 20) | Cost | Features / params | Ch / bands / windows | Block recordings p < 0.05 (uncorr. / Holm, of 42) | Cross-session (of 20) |
|---|---|---|---|---|---|---|---|
| `fbcsp_all` | **0.353 ± 0.044** | 0.337 ± 0.039 | 1.000 | 168 | 8 / 4 / 7 | 10 / 2 | 2 / 0 |
| `ga_single` | **0.353 ± 0.029** | 0.328 ± 0.027 | 0.478 | 38.5 | 3.8 / 2.0 / 3.2 | 6 / 3 | 0 / 0 |
| `ga_nsga` | 0.345 ± 0.025 | **0.340 ± 0.027** | **0.295** | **13.3** | 2.7 / 1.3 / 1.6 | 5 / 2 | 2 / 0 |
| `eegnet_32ch` | 0.341 ± 0.028 | 0.325 ± 0.029 | – | 3107 params | 32 ch | 6 / 2 | 1 / 0 |
| `eegnet_8ch` | 0.347 ± 0.031 | 0.334 ± 0.033 | – | 2723 params | 8 ch | 8 / 1 | 2 / 0 |

SD = sample SD over subjects; chance = 0.333. Per-recording tests are label
permutations on the stored predictions (`analysis.above_chance`, 1000
permutations, seed 0), Holm over the recordings of each method. All five methods
are within 0.012 BA of each other on block, and no group mean is more than
0.02 above chance. Full table: `results/tables/p3/main_results.md`; per subject:
`results/tables/p3/per_subject.csv`. Note: the `p (Holm)` column in that table is
`make_tables`' own family (each method vs `ga_nsga`, block). It is descriptive;
the pre-registered p-values are the ones in §2 and §3 above.

Figures: `results/figures/p3/` (method comparison, cost vs accuracy, confusion
matrices, gene selection heatmaps and frequencies, stability), for block and
cross-session.

### 4.2 Gene selection

Selection frequency over all block folds (504 per method):

* `ga_nsga`: 18–43 %. Top 5: **Pz 43 %**, Oz 38 %, P7 35 %, P4 35 %, theta 34 %.
  Bottom 3: windows 2.5 s 22 %, 1.0 s 19 %, 0.5 s 18 %.
* `ga_single`: 38–54 %, close to the Bernoulli(0.5) start. Top 5: alpha 54 %,
  Pz 54 %, Oz 51 %, 0.0 s 51 %, lowgamma 51 %.

The only consistent preference is the midline parietal-occipital pair Pz/Oz.

### 4.3 Selection stability of `chosen_bits` (Kuncheva / Jaccard, mean over recordings)

| Method | Block, all pairs | Block, same fold, different seed | Block, same seed, different fold | Cross-session (seeds) |
|---|---|---|---|---|
| `ga_nsga` | **0.078** / 0.23 | 0.163 / 0.30 | 0.072 / 0.22 | 0.051 / 0.22 |
| `ga_single` | **0.084** / 0.37 | 0.254 / 0.50 | 0.058 / 0.35 | 0.278 / 0.53 |

Kuncheva near 0 means no more agreement than random subsets of the same size.
**The stability-aware GA is not more stable than the accuracy-only GA**, and both
are close to chance. Selections agree more across seeds on the same training
data than across folds, so the training trials drive the choice more than the GA's
random seed does.

### 4.4 Inner-CV optimism (chosen chromosome, from the stored fronts)

| Method, protocol | Inner BA | Outer BA | Optimism |
|---|---|---|---|
| `ga_nsga`, block | 0.512 | 0.346 | +0.17 |
| `ga_single`, block | 0.568 | 0.354 | +0.21 |
| `ga_nsga`, cross-session | 0.491 | 0.340 | +0.15 |
| `ga_single`, cross-session | 0.552 | 0.328 | +0.22 |

The fitness signal is mostly noise. The search finds chromosomes that fit the
three inner folds by chance, and the outer test does not confirm them.

### 4.5 Cross-session drop (ses-02 block BA − cross-session BA, n = 20)

| Method | All | Without sub-01 (n = 19) |
|---|---|---|
| `fbcsp_all` | +0.027 | +0.031 |
| `ga_single` | +0.028 | +0.026 |
| `ga_nsga` | **+0.002** | **+0.003** |
| `eegnet_32ch` | +0.013 | +0.014 |
| `eegnet_8ch` | +0.004 | +0.006 |

`ga_nsga` loses almost nothing across sessions. Its block BA is lower to begin
with, so this is a smaller drop, not a higher cross-session accuracy (S2 n.s.).

### 4.6 Random-vs-block inflation and position-only control

As stated in the plan: for `fbcsp_all` it is in P2 (−0.004). The frozen GA and
EEGNet runners support only `block` and `cross_session`, so the inflation is
**not estimated** for them. The position-only control for cross-session is a P5
task (R07).

### 4.7 EEGNet training (context for S3/S4)

In 32 % (`eegnet_32ch`) and 38 % (`eegnet_8ch`) of block folds, early stopping
restores the weights after the first epoch (validation set of about 18 trials).
Those folds score 0.329 / 0.338, against 0.350 / 0.351 for the others. S3 and S4
compare `ga_nsga` with EEGNet *as trained by the frozen §k2 recipe*. See
`docs/P3_REPORT_R07.md` §3.

## 5. Final chromosome rule

CONTRACTS §g as written, with the §k8 tie-break. Every result above uses it. No
re-choice from the stored fronts was made. The §k12 re-examination is a separate,
exploratory step.

## 6. Deviations from the plan

1. **Results that predated the plan existed on one machine.** On 2026-11-08,
   18:30–20:12 +0530, R07 ran the full GA and EEGNet grid locally. That was before
   the plan was pushed (22:49), so the plan's statement that "no P3 GA result
   exists" was not true for R07's local disk. Those results were never committed
   or pushed, and R02 could not see them. Every file analysed here comes from runs
   made after the plan: R02 and R01's GA shares, and R07's fresh runs started
   23:10. R07's analysed files are bit-identical to its earlier local run, and
   R01/R02's files are bit-identical to R07's local reruns of their subjects
   (`docs/P3_REPORT_R07.md` §4.2–4.3). So the earlier runs could not have
   influenced any number here.
2. **Not in the plan, labelled exploratory:** the 95 % t-interval in §2, the
   Friedman test over the five methods in `results/tables/p3/main_results.md`
   (χ² = 1.20, p = 0.878, n = 22), and that table's own p-values.
3. Nothing else: no subject excluded, no file missing, the Holm family as planned
   (m = 4), and the chooser unchanged.

*Erratum (2026-11-10, P6):* two cells in the §3 table were rounded twice from 3-decimal output. S2 r_rb is +0.13 (exact 0.1345, was +0.14) and S3 median Δ is −0.001 (exact −0.00147, was −0.002). No p-value or conclusion changes; `results/final/tables/T2_preregistered_tests.md` is generated directly from the results.

## 7. Conclusion

On 22 subjects, selecting channels, bands and windows with a stability-aware
NSGA-II **does not improve accuracy** over the full FBCSP baseline (−0.007 BA,
one-sided p = 0.79). It is also not better than accuracy-only selection or
EEGNet. Every method is within about 0.02 BA of chance at the group level. What
`ga_nsga` does deliver is **the same accuracy at about 30 % of the cost** (13 of
168 features, 2.7 of 8 channels) and the smallest cross-session drop. Its
selections are not stable, though (Kuncheva 0.08), and the inner CV overestimates
accuracy by about 0.17. So the reduced subset should not be read as a recovered
neural signature.

Next (P5): the random-chromosome control at matched cost (R02 §3.2), the
position-only cross-session control (R07), and the §k12 re-examination from the
stored Pareto fronts. All three are exploratory relative to this plan.
