# P5 report: robustness checks and controls

R07, started 2026-11-09. **Everything here is exploratory relative to
`docs/ANALYSIS_PLAN.md`.** Nothing in this report changes the pre-registered results
in `docs/RESULTS.md`. Code: `main` @ `a267a66` (tag `p4-results` + PR #16) plus the
commits on branch `p5`. Config `08e124167b`. All P5 runs are on R07 (i5-13450HX,
RTX 3050 6 GB, `AA_HOST=R07`), because R01 and R02 are offline.

| § | Check | Status |
|---|---|---|
| 1 | Position-only control (cross-session, with and without sub-01) | done |
| 2 | Random-chromosome control at matched cost | done |
| 3 | "Within 1 SD" chooser re-examination (§k12) | done |
| 4 | Random-vs-block inflation (GA, EEGNet); viewing-period GA | done |
| 5 | Training accuracy; EEGNet early-stopping diagnostic | pending |

## 1. Position-only control

**Question** (P2_REPORT §3, P2_REVIEW §7, DATA_NOTES §4): can the class be predicted
from where a trial sits in the presentation sequence, without any EEG? If yes, a
method could score above chance by picking up a position-locked signal (fatigue
within a block, time since a pause, slow drift) instead of shape information. This
matters most for sub-01, the only subject whose ses-01 and ses-02 use the same
120-trial sequence. sub-01 is also the best cross-session `fbcsp_all` result.

**Method** (`experiments/position_control.py`). The classifier is a lookup table:
position key → majority class of the training trials with that key. Ties and keys
missing from training (rejected trials) are resolved by a seeded RNG, and test labels
are never used. It runs on exactly the outer splits and test trials of `fbcsp_all`
and the GA (`fbcsp` cache):

* `position_block30`: key = position within the 30-trial block (`trial_id % 30`),
  the control specified in DATA_NOTES §4. Block and cross-session protocols.
* `position_seq120`: key = position in the 120-trial session (`trial_id`).
  Cross-session only, because a held-out block's positions never occur in training.
  This is what a pure time-in-session signal could exploit.

Results are contract JSONs (82 files, seed 0) in `results/p5/position_control/`.
Per-recording p-values come from `analysis.above_chance` (label permutation within
fold, 1000 permutations, seed 0). Group tests are one-sided Wilcoxon of subject BA
against 1/3.

### 1.1 Results

Cross-session per subject (BA, permutation p). `shared` = number of ses-02 blocks
whose 30-trial label order also occurs in ses-01 (§1.2).

| Subject | shared | `position_block30` | `position_seq120` | `fbcsp_all` | `ga_nsga` | `ga_single` | `eegnet_32ch` | `eegnet_8ch` |
|---|---|---|---|---|---|---|---|---|
| **01** | **4** | **0.536 (0.001)** | **0.940 (0.001)** | **0.406 (0.043)** | 0.366 (0.082) | 0.318 (0.739) | 0.337 (0.428) | 0.363 (0.190) |
| 02 | 3 | **0.533 (0.001)** | 0.333 (0.528) | 0.350 (0.343) | 0.342 (0.387) | 0.339 (0.445) | 0.317 (0.723) | 0.328 (0.591) |
| 03 | 3 | **0.500 (0.001)** | 0.300 (0.801) | 0.358 (0.315) | 0.358 (0.208) | 0.347 (0.343) | 0.336 (0.473) | 0.347 (0.306) |
| 04 | 1 | 0.317 (0.717) | 0.325 (0.629) | 0.308 (0.754) | 0.317 (0.764) | 0.325 (0.646) | 0.319 (0.687) | 0.314 (0.727) |
| 05 | 0 | 0.358 (0.325) | 0.342 (0.453) | **0.400 (0.049)** | 0.292 (0.978) | 0.369 (0.080) | 0.294 (0.889) | **0.389 (0.012)** |
| 06 | 0 | 0.384 (0.135) | 0.317 (0.632) | 0.357 (0.303) | 0.300 (0.896) | 0.342 (0.285) | 0.337 (0.439) | 0.336 (0.465) |
| 07 | 1 | **0.408 (0.048)** | 0.333 (0.533) | 0.383 (0.063) | 0.328 (0.621) | 0.317 (0.702) | 0.303 (0.846) | 0.319 (0.673) |
| 08 | 0 | 0.308 (0.745) | 0.242 (0.985) | 0.317 (0.736) | 0.325 (0.649) | 0.311 (0.844) | 0.333 (0.524) | 0.256 (0.999) |
| 11 | 0 | 0.378 (0.155) | 0.386 (0.126) | 0.288 (0.884) | 0.341 (0.389) | 0.357 (0.203) | 0.356 (0.188) | 0.361 (0.142) |
| 12 | 0 | 0.277 (0.925) | 0.319 (0.666) | 0.297 (0.871) | 0.318 (0.726) | 0.336 (0.477) | 0.305 (0.908) | 0.322 (0.685) |
| 13 | 0 | 0.379 (0.177) | 0.370 (0.231) | 0.339 (0.453) | 0.356 (0.182) | 0.354 (0.051) | 0.343 (0.345) | 0.336 (0.481) |
| 14 | 3 | **0.466 (0.004)** | 0.320 (0.626) | 0.304 (0.789) | 0.353 (0.201) | 0.307 (0.745) | **0.389 (0.033)** | 0.327 (0.533) |
| 15 | 0 | 0.367 (0.252) | 0.308 (0.777) | 0.342 (0.460) | **0.394 (0.010)** | 0.264 (0.986) | 0.292 (0.952) | 0.344 (0.353) |
| 16 | 2 | **0.491 (0.001)** | 0.287 (0.878) | 0.346 (0.367) | 0.333 (0.519) | 0.345 (0.343) | 0.325 (0.633) | 0.343 (0.390) |
| 17 | 2 | **0.449 (0.006)** | 0.356 (0.280) | 0.324 (0.639) | 0.364 (0.189) | 0.325 (0.647) | 0.305 (0.813) | 0.339 (0.404) |
| 18 | 1 | **0.432 (0.006)** | 0.356 (0.323) | 0.270 (0.941) | 0.345 (0.342) | 0.322 (0.594) | 0.386 (0.065) | **0.400 (0.021)** |
| 19 | 0 | 0.298 (0.759) | 0.307 (0.704) | 0.296 (0.840) | 0.303 (0.859) | 0.371 (0.140) | 0.333 (0.498) | 0.337 (0.450) |
| 20 | 0 | 0.400 (0.065) | 0.367 (0.267) | 0.333 (0.552) | 0.333 (0.524) | 0.303 (0.885) | 0.286 (0.970) | 0.283 (0.957) |
| 21 | 0 | 0.300 (0.796) | 0.267 (0.964) | 0.400 (0.073) | **0.383 (0.025)** | 0.314 (0.816) | 0.294 (0.942) | 0.344 (0.367) |
| 22 | 1 | 0.361 (0.251) | 0.336 (0.427) | 0.317 (0.624) | 0.350 (0.280) | 0.289 (0.927) | 0.300 (0.839) | 0.291 (0.925) |

Group (subject level; one-sided Wilcoxon against 1/3; recordings with permutation
p < 0.05, uncorrected / Holm):

| Method | Cross-session, all (n = 20) | Cross-session, without sub-01 (n = 19) | Block (n = 22) |
|---|---|---|---|
| `position_block30` | **0.397 ± 0.079, p = 0.001**, 8 / 4 | **0.390 ± 0.074, p = 0.002**, 7 / 3 | 0.329 ± 0.038, p = 0.69, 2 / 1 of 42 |
| `position_seq120` | 0.356 ± 0.142, p = 0.65, 1 / 1 | 0.325 ± 0.036, p = 0.80, 0 / 0 | – |
| `fbcsp_all` | 0.337 ± 0.039, p = 0.41, 2 / 0 | 0.333 ± 0.037, p = 0.56, 1 / 0 | 0.353 ± 0.044, p = 0.027 |
| `ga_nsga` | 0.340 ± 0.027, p = 0.15, 2 / 0 | 0.339 ± 0.027, p = 0.22, 2 / 0 | 0.345 ± 0.025, p = 0.021 |
| `ga_single` | 0.328 ± 0.027, p = 0.81, 0 / 0 | 0.328 ± 0.027, p = 0.76, 0 / 0 | 0.353 ± 0.029, p = 0.001 |
| `eegnet_32ch` | 0.325 ± 0.029, p = 0.91, 1 / 0 | 0.324 ± 0.030, p = 0.92, 1 / 0 | 0.341 ± 0.028, p = 0.083 |
| `eegnet_8ch` | 0.334 ± 0.033, p = 0.40, 2 / 0 | 0.332 ± 0.033, p = 0.50, 2 / 0 | 0.347 ± 0.031, p = 0.038 |

(The block p-values of the methods are one-sided tests against chance. They are
descriptive and are not the pre-registered method comparisons.)

### 1.2 New finding: 30-trial block orders are reused across sessions

DATA_NOTES §4 found that only sub-01 repeats the whole 120-trial sequence across
sessions. That holds, but **whole 30-trial blocks are reused much more often**,
usually in a shifted block order. Example: sub-02's ses-02 blocks 1–3 have exactly
the label order of its ses-01 blocks 0–2. 10 of the 20 cross-session subjects have
at least one ses-02 block identical to a ses-01 block (`shared` above: 4 for sub-01;
3 for 02, 03, 14; 2 for 16, 17; 1 for 04, 07, 18, 22). The within-block position →
class mapping therefore transfers across sessions for half of the subjects, not only
for sub-01. That is why `position_block30` is above chance cross-session even without
sub-01 (0.390). `position_seq120` stays at chance without sub-01 (0.325), because
shifted blocks do not keep their session position.

Within a session, blocks are distinct except in one recording: **sub-13 ses-01,
blocks 0 and 3 are identical.** This is the only Holm-significant recording of the
block-protocol control (0.517, p_holm = 0.042). The leave-one-block-out protocol is
otherwise unaffected (group 0.329).

### 1.3 Do the methods exploit it?

**No sign that they do.** If a method used a position-locked signal, its
cross-session BA would be higher where position carries class information. It is not:

| Method | ≥ 1 shared block (n = 10) | Same, without sub-01 (n = 9) | No shared block (n = 10) | Mann–Whitney p | ρ with `shared` |
|---|---|---|---|---|---|
| `position_block30` | **0.449** | 0.440 | 0.345 | **0.004** | **+0.80** |
| `fbcsp_all` | 0.337 | 0.329 | 0.337 | 0.79 | +0.20 |
| `ga_nsga` | 0.346 | 0.343 | 0.335 | 0.27 | +0.36 |
| `ga_single` | 0.323 | 0.324 | 0.332 | 0.47 | −0.11 |
| `eegnet_32ch` | 0.332 | 0.331 | 0.317 | 0.38 | +0.29 |
| `eegnet_8ch` | 0.337 | 0.334 | 0.331 | 0.97 | +0.12 |

* Across the 20 subjects, the methods' cross-session BA correlates weakly and not
  significantly with the control's BA: Spearman ρ = +0.33 (`fbcsp_all`, p = 0.16),
  +0.40 (`ga_nsga`, p = 0.08), −0.03 (`ga_single`), +0.33 (`eegnet_32ch`),
  +0.26 (`eegnet_8ch`).
* Trial-level agreement between each method's predictions and the control's
  predictions is at chance: Cohen's κ between −0.02 and +0.01 (mean over the 19
  subjects other than sub-01). When the control is wrong, the methods pick the
  control's class at the rate expected under independence (0.32–0.34 vs 0.33–0.34).
* The methods do not reach the information that position offers: the control scores
  0.449 on the shared-block subjects, the best method 0.346.

`ga_nsga`'s ρ = +0.40 is the largest. It is not significant, and the group
differences behind it are small (0.346 vs 0.335). It is worth a mention, not a claim.

### 1.4 sub-01

For sub-01, class and position are confounded by design: `position_seq120` predicts
ses-02 at 0.940, which is below 1.0 only because rejected trials differ between
sessions. A position-locked EEG signal and a shape signal therefore make *identical*
predictions on sub-01, so no analysis of these data can separate them. What can be
said:

* sub-01's `fbcsp_all` 0.406 (p = 0.043) is the only result that drives the
  cross-session mean above chance (0.337 → 0.333 without it). It is one subject,
  uncorrected, and not significant after Holm over the 20 subjects (P2).
* Its errors do not follow the within-block position prediction (`fbcsp_all`
  follows the wrong `position_block30` class on 0.377 of trials vs 0.357 expected,
  κ = +0.11). The GA and EEGNet are not elevated on sub-01 (0.318–0.366).
* Subjects 02, 03 and 14 share three of four blocks and get no comparable gain
  (`fbcsp_all` 0.350, 0.358, 0.304).

### 1.5 Conclusion

**Trial position does not explain the cross-session results.** The methods are at
chance cross-session (0.325–0.340, all p ≥ 0.15). They are not higher on the
subjects whose sessions share block orders, and their predictions do not follow the
position control. **For sub-01 the question cannot be answered** with these data,
since position and class are identical there. Its 0.406 stays flagged, and
cross-session results should keep being reported with and without sub-01, as
RESULTS.md S2 already does (Δ +0.003 → +0.006, both n.s.).

Implications:

1. **DATA_NOTES §4 should be extended:** block orders (not only whole sequences)
   are reused across sessions for 10 of 20 subjects, and sub-13 ses-01 repeats a
   block within the session. Any future analysis that can see within-block position
   (cross-session, cross-subject, or the `random` protocol) inherits this confound.
2. The leave-one-block-out protocol remains sound (control at chance, 0.329).
3. A decoder that learned a position-within-block signal would gain on half the
   subjects in cross-session. These methods show no such gain, which is weak
   evidence that they are not sensitive to position.

Reproduce (about 2 min on R07):

```
AA_HOST=R07 python -m experiments.position_control --subjects 01..22 --out results/p5/position_control
```

Outputs: `results/p5/position_control/{position_block30,position_seq120}/*.json`,
`per_subject.csv`, `summary.json` (every number in this section) and `summary.md`.

## 2. Random-chromosome control at matched cost

**Question** (P3_REPORT_R02 §3.2): does the GA's *search* find better components than
chance, or does it only deliver the cost reduction? If a random subset of the same
size scores as well as the GA's chosen subset, the search adds nothing.

**Method** (`experiments/random_control.py`). For every outer fold of `ga_nsga` and
`ga_single`, 50 random chromosomes are drawn with exactly the same number of
channels, bands and windows as that fold's `chosen_bits`. That gives the same cost
and validity; within each gene group the subset is uniform. Draws are seeded per
fold with `default_rng([method, subject, session, seed, fold])`. Each draw is scored
once on the outer test fold exactly like `run_ga`'s final step,
`fbcsp.evaluate(X, y, bits, [(train_idx, test_idx)])`: a fresh FBCSP+LDA on the full
outer training set. The GA's own chromosome is re-scored the same way and must
reproduce the stored fold BA. It does, in all 1128 folds (504 block + 60
cross-session per method), so the comparison runs on an identical pipeline.
Block: 22 subjects × seeds 0–2. Cross-session: 20 subjects × seeds 0–2.
Subject level = fold mean per file, then seeds, then sessions (`analysis.subject_level`).

| Method, protocol | n | GA BA | Random BA, same cost | GA − random (Wilcoxon p: two-sided / GA > random) | Subjects GA > random | GA percentile among the 50 draws: mean (subject-level p vs 0.5) | Folds with GA ≥ 95th / ≤ 5th percentile | Random − `fbcsp_all` (p) |
|---|---|---|---|---|---|---|---|---|
| `ga_nsga`, block | 22 | 0.345 | 0.341 | +0.004 (0.42 / 0.21) | 13 | 0.51 (0.54) | 6.9 % / 5.0 % | −0.011 (0.14) |
| `ga_single`, block | 22 | 0.353 | 0.347 | +0.007 (0.20 / 0.10) | 15 | 0.52 (0.14) | 6.7 % / 4.8 % | −0.006 (0.31) |
| `ga_nsga`, cross-session | 20 | 0.340 | 0.335 | +0.005 (0.31 / 0.16) | 12 | 0.55 (0.28) | 11.7 % / 6.7 % | −0.002 (0.78) |
| `ga_single`, cross-session | 20 | 0.328 | 0.335 | −0.007 (0.14 / 0.93) | 9 | 0.43 (0.14) | 5.0 % / 11.7 % | −0.002 (0.96) |

Mean cost: `ga_nsga` 0.30, `ga_single` 0.48 (block). Under "the GA is a random draw",
the percentile averages 0.5 and 5 % of folds fall in each 5 % tail.

* **The GA's subset is indistinguishable from a random subset of the same size.**
  The GA's percentile averages 0.51 / 0.52 (block), the tails are at their
  nominal 5 %, and the GA − random differences (+0.004 / +0.007) are not
  significant. The SD of the subject-level differences is 0.021 (`ga_nsga`, block),
  so with n = 22 a search gain of about **0.011 BA** would have been detected
  (80 % power, one-sided α = 0.05). There is no evidence of a gain that large.
* **Cutting components costs little accuracy.** Random subsets at 30 % of the cost
  score 0.011 below `fbcsp_all` (block, p = 0.14). At 48 % of the cost they score
  0.006 below (p = 0.31). Cross-session, the gap is 0.002.
* **The test fold is noisy.** Within a fold, the 50 random subsets spread with an
  SD of 0.083 BA (block, about 30 test trials per fold). This is far larger than
  any systematic difference between the methods. Selecting on a 3-fold inner CV of
  about 90 trials cannot beat this noise, which matches the inner-CV optimism of
  +0.17 / +0.21 (RESULTS.md §4.4).

**Conclusion: the GA search adds nothing beyond the cost reduction.** What
`ga_nsga` delivers, the same accuracy at about 30 % of the cost, a random subset
of the same size delivers too. Its selections are as unstable as random ones
(RESULTS.md §4.3, Kuncheva 0.08), and this control shows they are also no more
accurate. With this little class information in the signal, BA is roughly flat
across component subsets, so any subset of moderate size is about as good as any
other. The GA's cost advantage is real but is not evidence of a meaningful
selection. A stability-aware GA could only show a benefit on data with a stronger,
more localised signal (e.g. the viewing period, §4).

Reproduce (20 s on 8 workers, R07):

```
AA_HOST=R07 python -m experiments.random_control --n-draws 50 --n-jobs 8 --out results/p5/random_control
```

Outputs: `results/p5/random_control/{ga_nsga,ga_single}/*.json` (per fold: GA bits
and BA, the 50 random bits and BAs, percentile), `fold_level.csv`,
`subject_level.csv`, `summary.json` (every number in this section) and `summary.md`.

## 3. "Within 1 SD" chooser re-examination (CONTRACTS §k12)

**Exploratory (ANALYSIS_PLAN §5): this never replaces the stored `ga_nsga` results.
CONTRACTS §g is unchanged and `ga_nsga` was not re-run.**

**Question:** the contract picks the cheapest Pareto entry whose inner BA is within
1 SD of the best (`b*`). With 3 small inner folds the SD is large, so the rule may
pick chromosomes that are too small (CONTRACT_ISSUES #11). Would another rule do
better?

**Method** (`experiments/pareto_thresholds.py`, ported from R07's pre-plan WIP and
extended). The rules are applied to the **stored** per-fold `ga_nsga` fronts
(`results/pareto/`), with no new search:

| Rule | Picks |
|---|---|
| `0sd` | best inner BA (`b*`; ties → cheapest) |
| `0.5sd`, `1sd`, `2sd` | cheapest entry with inner BA ≥ `b*` − k·SD (`1sd` = contract rule) |
| `se` | cheapest entry within one standard error (SD/√3) |
| `knee` | knee of the 2-D (cost, inner BA) front: largest distance above the chord from the cheapest to the most accurate entry, axes scaled to [0, 1] (fewer than 3 points → `b*`) |
| `mincost` | cheapest entry of the front |

Each rule's chromosome is re-scored once on the outer test fold with the same
FBCSP+LDA (`fbcsp.evaluate` on the full outer training set). `1sd` reproduces every
stored `chosen_bits` (asserted), and every row that overlaps R07's pre-plan run of
this script is identical. Stability is Kuncheva / Jaccard over all (fold, seed)
pairs per recording.

| Rule | Block BA (n = 22) | Cost | Channels | Inner BA | Inner − outer | Kuncheva / Jaccard | − `fbcsp_all` (p) | − `1sd` (p) | Cross-session BA (n = 20) | − `fbcsp_all` (p) |
|---|---|---|---|---|---|---|---|---|---|---|
| `0sd` (best inner) | **0.351** | 0.415 | 3.4 | 0.545 | +0.194 | 0.066 / 0.31 | −0.002 (0.87) | +0.005 (0.25) | 0.338 | +0.001 (0.87) |
| `0.5sd` | 0.348 | 0.345 | 3.0 | 0.533 | +0.185 | 0.076 / 0.26 | −0.004 (0.63) | +0.003 (0.15) | 0.336 | −0.001 (0.86) |
| `se` | 0.344 | 0.334 | 2.9 | 0.529 | +0.186 | 0.074 / 0.25 | −0.009 (0.29) | −0.002 (0.66) | 0.338 | +0.001 (0.72) |
| **`1sd` (contract)** | 0.345 | 0.295 | 2.6 | 0.512 | +0.167 | 0.078 / 0.23 | −0.007 (0.44) | – | 0.340 | +0.003 (0.60) |
| `knee` | 0.347 | 0.282 | 2.5 | 0.510 | +0.162 | 0.070 / 0.22 | −0.005 (0.61) | +0.002 (0.87) | 0.334 | −0.003 (0.90) |
| `2sd` | 0.338 | 0.247 | 2.3 | 0.486 | +0.147 | 0.077 / 0.20 | −0.014 (0.28) | −0.007 (0.14) | 0.341 | +0.004 (0.55) |
| `mincost` | 0.343 | **0.214** | 2.0 | 0.463 | +0.119 | 0.074 / 0.18 | −0.009 (0.50) | −0.002 (0.73) | **0.342** | +0.005 (0.65) |
| *`ga_single` (stored)* | 0.353 | 0.476 | 3.8 | 0.568 | +0.215 | 0.084 / 0.37 | +0.001 (0.80) | +0.008 (0.26) | 0.328 | −0.009 (0.35) |

`fbcsp_all`: 0.353 block, 0.337 cross-session. Two-sided Wilcoxon over subjects
(seeds, then sessions averaged).

* **No rule beats `fbcsp_all`.** All rules lie within −0.014 to +0.005 BA of it,
  with no p below 0.27. No rule differs significantly from the contract rule
  either (all p ≥ 0.14).
* **There is a weak cost-accuracy slope on block and none on cross-session.** On
  block, looser rules (cheaper chromosomes) lose a little: from 0.351 (`0sd`,
  cost 0.42) to 0.338 (`2sd`, cost 0.25). On cross-session the order reverses
  (`mincost` is highest). The full range is 0.013 BA, below the seed-to-seed SD of
  a recording (about 0.035), and consistent with BA being flat across subset size
  (§2).
* **Inner-CV optimism falls with cost** (+0.19 for `0sd` to +0.12 for `mincost`):
  picking the best inner score from about 600 evaluations selects lucky
  chromosomes (winner's curse), and less selection pressure on accuracy means less
  luck. Outer BA does not change with it.
* **Stability is near chance for every rule** (Kuncheva 0.066–0.078). No chooser
  makes the selection reproducible, as expected given §2.

**Picking a rule by these numbers would be selection on the test set.** The outer
test folds were used to score every rule, so choosing the rule with the best outer
BA (e.g. `0sd` on block, `mincost` on cross-session) would leak test information
into the method and bias that rule's score upward. The rules would have to be
compared on held-out data. Here the differences are smaller than the noise anyway,
so there is nothing to choose.

**Conclusion:** the §k12 rule is not the problem. No chooser recovers accuracy that
the search did not find; the fronts contain no reliably better chromosome. **No
change to CONTRACTS §g is proposed.**

Reproduce (about 3 min on R07):

```
python -m experiments.pareto_thresholds --results-dir results --out results/p5/chooser_rules
```

Outputs: `results/p5/chooser_rules/chooser_rules{,_cross_session}.csv` (this
table), `pareto_thresholds_folds*.csv` (one row per fold and rule) and
`pareto_thresholds_summary*.{csv,md}`.

## 4. Random-vs-block inflation and the viewing-period GA

**Exploratory.** ANALYSIS_PLAN §4.6 listed both as gaps for P3/P4, because the frozen
runners only supported `block` and `cross_session`. R07's runner extension
(`--protocol random`, `--cache-kind view`) was written before the plan and kept
as uncommitted WIP. It is ported here, and the block and cross-session outputs are
unchanged: GA (both methods) and EEGNet (32ch block, 8ch cross-session) were re-run
on sample recordings and are bit-identical to the committed P3/P4 files. Every new
file was then produced from committed `p5` code with `AA_HOST=R07` and config
`08e124167b`. The GA `random` and view files (504) and the EEGNet 32ch `random`
files (126) are **bit-identical to R07's pre-plan local runs**. EEGNet 8ch `random`
(126 files) had not been run before. Split / GA seeds 0–2 for every method; the
`fbcsp_all` reference is P2's runs restricted to seeds 0–2 (P2 reported −0.004 with
seeds 0–4).

### 4.1 Random 5-fold vs leave-one-block-out (subject level, n = 22)

| Method | Block | Random | Random − block (median) | Wilcoxon p | Subjects random > block |
|---|---|---|---|---|---|
| `fbcsp_all` | 0.353 | 0.349 | −0.004 (−0.002) | 0.85 | 10 |
| `ga_nsga` | 0.345 | 0.340 | −0.006 (−0.011) | 0.31 | 6 |
| `ga_single` | 0.353 | 0.343 | −0.011 (−0.006) | 0.09 | 8 |
| `eegnet_32ch` | 0.341 | 0.339 | −0.002 (−0.008) | 0.52 | 9 |
| `eegnet_8ch` | 0.347 | 0.342 | −0.005 (−0.009) | 0.70 | 9 |

**There is no random-split inflation for any method.** DATA_NOTES §4 expected
random > block, because random folds put temporally adjacent trials (often the same
class, in runs of 3–5) into training and test, so slow drifts become class-predictive.
That does not happen here: every method is slightly *lower* under random splits, and
none significantly. For EEGNet the random splits also carve its validation set from
interleaved trials, which could have helped early stopping; they don't. So the block
protocol does not hide a drift-driven gain, and the near-chance block results are not
an artefact of strict splitting. There is little class information to inflate.

### 4.2 Viewing period (3–7 s, the shape on screen) vs imagery (block)

| Method | Imagery | Viewing | Viewing − imagery (p) | Viewing recordings p < .05 (uncorr. / Holm, of 42) | Viewing cost |
|---|---|---|---|---|---|
| `fbcsp_all` | 0.353 | **0.435 ± 0.069** | +0.083 (0.0003) | 25 / 12 | 1.00 |
| `ga_single` | 0.353 | 0.428 ± 0.074 | +0.074 (0.0001) | 30 / 20 | 0.54 |
| `ga_nsga` | 0.345 | 0.410 ± 0.048 | +0.065 (0.00002) | 30 / 15 | 0.33 |

Viewing vs the full baseline (paired Wilcoxon, n = 22): `ga_nsga` **−0.025 (p = 0.004,
better on 6/22)**; `ga_single` −0.008 (p = 0.39); `ga_nsga` vs `ga_single` −0.018
(p = 0.04).

* **Every method decodes the viewing period** well above imagery (+0.065 to +0.083),
  with 12–20 recordings significant after Holm. This confirms the pipeline works
  when the signal is there (P2_REPORT §2(c)).
* **With a real signal, `ga_nsga`'s cost pressure hurts.** It is significantly worse
  than the full baseline and than `ga_single`, at about a third of the cost. On
  imagery the same comparison was flat (RESULTS.md §2/§3), because there was almost
  nothing to lose.
* **The chooser rule matters on the viewing data** (the §3 analysis repeated on the
  view fronts, `results/p5/chooser_rules/pareto_thresholds_summary_view.md`):
  best inner BA (`0sd`) 0.421 at cost 0.45, `1sd` 0.410 at 0.33, `knee` 0.398,
  `2sd` 0.393, `mincost` 0.377 at 0.21. Accuracy now falls steadily with cost,
  unlike the flat imagery result in §3. The 1-SD rule gives up about 0.01 BA
  relative to `0sd`, and about 0.025 relative to `fbcsp_all`, for a 67 % cost
  reduction. (Same caveat as §3: these are test-set scores.)
* The GA selections still pass Holm on more recordings than `fbcsp_all` (15–20 vs
  12) despite the lower mean. The smaller models seem to help the weak recordings a
  little and cost the strong ones more.

**Conclusion:** with a decodable signal, the stability-aware GA trades accuracy for
cost in the expected way. It never matches the full baseline, and the 1-SD rule sits
at a point that costs a significant 0.025 BA. On imagery this trade-off is invisible
only because there is no accuracy to trade.

Reproduce:

```
AA_HOST=R07 python -m experiments.run_ga --method ga_nsga   --subjects 01..22 --protocol random --results-dir results/p5/drift
AA_HOST=R07 python -m experiments.run_ga --method ga_single --subjects 01..22 --protocol random --results-dir results/p5/drift
AA_HOST=R07 python -m experiments.run_ga --method ga_nsga   --subjects 01..22 --cache-kind view --results-dir results/p5/drift
AA_HOST=R07 python -m experiments.run_ga --method ga_single --subjects 01..22 --cache-kind view --results-dir results/p5/drift
AA_HOST=R07 python -m experiments.run_eegnet --channels 32 8 --subjects 01..22 --protocol random --results-dir results/p5/drift
python -m experiments.pareto_thresholds --results-dir results/p5/drift --out results/p5/chooser_rules --view --protocol block --no-compare
python -m reports.p5_drift --results results --out results/p5/drift
```

Outputs: `results/p5/drift/{ga_nsga,ga_single,eegnet_32ch,eegnet_8ch}/*_random.json`,
`results/p5/drift/{ga_nsga,ga_single}_view/*.json`, the Pareto side files in
`results/p5/drift/pareto/`, and `summary.{json,md}`.
