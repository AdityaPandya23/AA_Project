# P5 report: robustness checks and controls

R07, started 2026-11-09. **Everything here is exploratory relative to
`docs/ANALYSIS_PLAN.md`.** Nothing in this report changes the pre-registered results
in `docs/RESULTS.md`. Code: `main` @ `a267a66` (tag `p4-results` + PR #16) plus the
commits on branch `p5`. Config `08e124167b`. All P5 runs are on R07 (i5-13450HX,
RTX 3050 6 GB, `AA_HOST=R07`), because R01 and R02 are offline.

| § | Check | Status |
|---|---|---|
| 1 | Position-only control (cross-session, with and without sub-01) | done |
| 2 | Random-chromosome control at matched cost | pending |
| 3 | "Within 1 SD" chooser re-examination (§k12) | pending |
| 4 | Random-vs-block inflation (GA, EEGNet); viewing-period GA | pending |
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
