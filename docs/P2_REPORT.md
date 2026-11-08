# P2 report: FBCSP baseline and sanity checks

R02, 2026-11-08. Code: `main` @ `4f6109f`. The PR #4 (`ga`) and #7 merges do not
touch `src/fbcsp.py`, `src/cv.py`, `experiments/baseline.py`, `experiments/sanity.py`
or `config.yaml`, so these numbers also hold for the integrated tree. Config
`08e124167b`. Cache: 42 recordings; `check_manifest` OK.

Method: `fbcsp_all` = FBCSP + shrinkage LDA with all 19 genes on (8 channels × 4
bands × 7 windows, 168 features). Chance = 0.333. Results:
`results/baseline_all/fbcsp_all/` (62 JSONs) and `results/sanity/` (shuffled,
random-split and viewing runs + `sanity_summary.{json,md}`). SDs are sample SDs
(`ddof=1`), as in `make_tables`; the committed `sanity_summary.*` still show
population SDs, which are up to 0.001 lower (P2_REVIEW §3.1).

Reproduce (about 4 min on 8 threads):

```
python -m experiments.baseline --n-jobs -1
python -m experiments.sanity  --n-jobs -1
```

## 1. Balanced accuracy

| Protocol | n | Mean ± SD | Median | Range | Above chance, p < 0.05 (uncorrected / Holm) |
|---|---|---|---|---|---|
| Block (leave-one-block-out), per recording | 42 | **0.356 ± 0.061** | 0.350 | 0.233–0.476 | 10 / 2 |
| Block, per subject (sessions averaged) | 22 | 0.353 ± 0.044 | | | group Wilcoxon > 1/3: p = 0.027 |
| Cross-session (train ses-01 → test ses-02) | 20 | **0.337 ± 0.039** | 0.336 | 0.270–0.406 | 2 / 0 |
| Cross-session **without sub-01** | 19 | 0.333 | | | |

* Per-recording test: label permutation on the stored fold predictions
  (`analysis.above_chance`, 1000 permutations, one-sided). By chance alone,
  2.1 of 42 recordings would reach p < 0.05; 10 do. The recordings significant
  before correction are sub-02 (both sessions), 03/01, 05/01, 11/02, 13/02,
  14/02, 20/02 and sub-22 (both sessions). After Holm correction, 02/02 and
  20/02 remain (both 0.475).
* The refit-based permutation test in the sanity run (200 permutations, the
  whole pipeline refitted) is more conservative: 6/42 at p < 0.05.
* ses-01 0.349 vs ses-02 0.364. For the 20 cross-session subjects the mean is
  0.360 within session and 0.337 across sessions (drop 0.023).

**Reading:** imagery decoding with the full FBCSP baseline is weak. It is
above chance at group level (p = 0.027), with a minority of clearly decodable
recordings. Cross-session transfer is at chance. This is the bar the GA (P3)
has to beat.

## 2. Sanity checks

| Check | Result | Verdict |
|---|---|---|
| (a) Shuffled labels (within-block permutation, full refit) | 0.333 ± 0.050 | ✅ no leakage in the pipeline |
| (b) Random stratified 5-fold vs block | 0.353 vs 0.356, inflation **−0.004** (18/42 higher) | ✅ no inflation for this baseline |
| (c) Viewing (3–7 s) vs imagery | **0.438 ± 0.076** vs 0.356, +0.082 (32/42 higher) | ✅ perception decodes better, as expected |

* (b) DATA_NOTES §4 predicted that random splits would be optimistic, because
  neighbouring trials end up in both train and test. That does not happen with
  this baseline: CAR plus band-limited log-variance apparently carries little
  slowly drifting information. Re-check it for the GA and EEGNet, which may
  pick up drift more easily.
* (c) This largely settles the DATA_NOTES open point about the viewing window.
  The 3–7 s window contains clearly decodable perceptual signal, so the timeline
  alignment (marker = imagery onset) is correct.

## 3. ⚠ Flag: sub-01 cross-session

sub-01 is the **only** subject whose ses-01 and ses-02 use the identical trial
sequence (DATA_NOTES §4). It is also the **best** cross-session result (0.406,
p = 0.043, one of only two below 0.05), although its within-session results are
unremarkable (0.300 and 0.361). Without it, cross-session is exactly at chance
(0.333).

This fits the position confound predicted in DATA_NOTES, but one subject
proves nothing. Action for P5 (R07 robustness task): run the position-only
control (predict class from `trial_id % 30`) and report cross-session results
with and without sub-01.

## 4. Status / next

* P2 baseline and sanity: done, results committed with this report.
* R01: P2 review (`make_tables`, figures, `docs/P2_REVIEW.md`).
* R07: full-budget GA check on real data. R07 already reports ~1.5–3 s per
  fold at pop 30 × 20 generations (PR #4); R02 measured 8.8 s for sub-01 block,
  both sessions, 8 folds. No budget change is needed, so `config_hash` can stay
  `08e124167b`.
