# P3/P4 report (R07 share): GA runs for sub-17..22, EEGNet for all subjects

R07, 2026-11-09. Code: `main` @ `6a38c4b`, run from a clean detached checkout (no
local changes). The GA and EEGNet paths (`src/ga.py`, `src/eegnet.py`, `src/fbcsp.py`,
`src/cv.py`, `src/chromosome.py`, `experiments/run_ga.py`, `experiments/run_eegnet.py`,
`experiments/common.py`, `experiments/_common.py`, `config.yaml`) are identical to tag
`p1-integrated` and to `db5c9b7` (current `main`), so §k2 applies. Config `08e124167b`
for all 480 files. 176 tests pass, 7 skipped (real-data tests; no `data/` in the
checkout). Subject split per CONTRACTS open item 12: R02 sub-01..08, R01 sub-09..16,
R07 sub-17..22, and R07 runs all EEGNet. **With this PR all three GA shares are in.**
The pre-registered group tests are in a separate PR (`docs/RESULTS.md`).

Machine: R07 (i5-13450HX, 16 logical CPUs, RTX 3050 6 GB Laptop), Windows 11,
Python 3.11.4, torch 2.14.1+cu130, braindecode 2.0.0, scikit-learn 1.9.1, numpy 2.4.6.
`AA_HOST=R07`.

Runs:

* **GA:** `ga_nsga` and `ga_single`, block (12 recordings × 3 seeds × 4 folds) and
  cross-session (sub-17..22 × 3 seeds). Pop 30 × 20 generations. 324 fold jobs,
  98 s wall time on 8 workers. Output: `results/ga_nsga/` and `results/ga_single/`
  (54 JSONs each) and `results/pareto/{method}/`.
* **EEGNet (P4):** `eegnet_32ch` and `eegnet_8ch`, all 22 subjects, block
  (42 recordings × 3 seeds) and cross-session (20 subjects × 3 seeds). 372 runs,
  49 min on the GPU. Output: `results/eegnet_32ch/` and `results/eegnet_8ch/`
  (186 JSONs each).

Reproduce (PowerShell: `$env:AA_HOST = "R07"`):

```
AA_HOST=R07 python -m experiments.run_ga --method ga_nsga   --subjects 17..22 --protocol block         --n-jobs 8   # 37 s
AA_HOST=R07 python -m experiments.run_ga --method ga_nsga   --subjects 17..22 --protocol cross_session --n-jobs 8   #  9 s
AA_HOST=R07 python -m experiments.run_ga --method ga_single --subjects 17..22 --protocol block         --n-jobs 8   # 42 s
AA_HOST=R07 python -m experiments.run_ga --method ga_single --subjects 17..22 --protocol cross_session --n-jobs 8   # 10 s
AA_HOST=R07 python -m experiments.run_eegnet --channels 32 8 --subjects 01..22 --protocol block           # 42 min
AA_HOST=R07 python -m experiments.run_eegnet --channels 32 8 --subjects 01..22 --protocol cross_session   #  7 min
```

**Do not pass `--device` to `run_eegnet`** (§4.1): the device is then hashed into
`config_hash`. CUDA is used by default when available.

## 1. GA balanced accuracy (sub-17..22, subject level, seeds then sessions averaged)

`fbcsp_all` is the P2 baseline restricted to the same subjects. On this share it is
close to its 22-subject mean for block (0.362 vs 0.353) and below it for
cross-session (0.323 vs 0.337).

| Method | Block mean ± SD (n = 6) | Cross-session mean ± SD (n = 6) | Cost | Features | Block recordings p < 0.05 (uncorr. / Holm, of 12) |
|---|---|---|---|---|---|
| `fbcsp_all` | **0.362 ± 0.047** | 0.323 ± 0.044 | 1.000 | 168 | 3 / 2 |
| `ga_single` | 0.361 ± 0.020 | 0.321 ± 0.028 | 0.494 | 41 | 2 / 1 |
| `ga_nsga` | 0.356 ± 0.018 | **0.347 ± 0.027** | **0.310** | **15** | 3 / 0 |

SD = sample SD over subjects. Paired Wilcoxon, two-sided, uncorrected. These are
descriptive only; the pre-registered tests use all 22 subjects (`docs/RESULTS.md`).

| Comparison | Block Δ (p, r_rb), n = 6 | Cross-session Δ (p, r_rb), n = 6 |
|---|---|---|
| `ga_nsga` − `fbcsp_all` | −0.006 (1.000, −0.05) | +0.023 (0.188, +0.73) |
| `ga_single` − `fbcsp_all` | −0.001 (0.844, +0.14) | −0.003 (1.000, −0.05) |
| `ga_nsga` − `ga_single` | −0.005 (0.844, −0.14) | +0.026 (0.312, +0.52) |

Per subject (block): `ga_nsga` beats `fbcsp_all` on 4 of 6 subjects (17, 18, 19, 21).
It loses clearly on the two subjects where the baseline is strongest:
sub-22 (0.443 → 0.358) and sub-20 (0.393 → 0.339). `ga_single` beats the baseline on
4 of 6. The baseline's decodable recordings are 20/02, 22/01 and 22/02. Of these,
only 22/01 stays significant under `ga_nsga`, which instead reaches p < 0.05 on 19/01
and 21/02, where the baseline does not. As on R01's share, the GA is better where the
baseline is weak and worse where it works.

Cross-session: `ga_nsga` is higher than `fbcsp_all` on 4 of 6 subjects, tied on
sub-20 and lower on sub-21 (0.347 vs 0.323). With n = 6 this is not significant (p = 0.19).

**Reading:** R07's share lies between R02's (GA below the baseline) and R01's (GA
slightly above). The block differences are within ±0.006, much smaller than the
seed-to-seed SD of a recording (0.035). The cost gain is the same as on the other
shares: `ga_nsga` uses about 9 % of the features (15 vs 168), and on R07 it predicts
in 0.011 ms per trial. The `fbcsp_all` timings were measured in P2, not in this
run, so no timing ratio is reported here.

## 2. Why: the inner CV is optimistic and the selection is close to random

For the chosen chromosome of each fold, its inner-CV BA from the stored front
compared with the outer test BA:

| Method, protocol | Inner BA (chosen) | Outer BA | Optimism | Front size |
|---|---|---|---|---|
| `ga_nsga`, block | 0.511 | 0.356 | **+0.16** | 21.0 |
| `ga_single`, block | 0.570 | 0.361 | **+0.21** | 30 |
| `ga_nsga`, cross-session | 0.501 | 0.347 | +0.16 | 22.8 |
| `ga_single`, cross-session | 0.557 | 0.321 | +0.24 | 30 |

* The overfitting gap is the same as on R02 (+0.17 / +0.21) and R01 (+0.18 / +0.22).
* **Seed variability:** the SD of a recording's BA across the 3 seeds is 0.035
  (`ga_nsga`) and 0.034 (`ga_single`) for block.
* **Mean selection (block):** `ga_nsga` 2.7 channels, 1.4 bands, 1.7 windows;
  `ga_single` 4.0 channels, 2.0 bands, 3.5 windows.
* **Selection frequency (block, 144 folds):** flat. `ga_nsga` selects every gene in
  17–47 % of folds. Top 5: Pz 47 %, theta 42 %, lowgamma 39 %, P4 38 %, P7 38 %.
  `ga_single` is at 29–60 %; top 5: Pz 60 %, 0.0 s 58 %, 2.5 s 57 %, Oz 56 %,
  P7 56 %. Pz is also R01's top gene, and the top gene over all 22 subjects
  (43 %, `docs/RESULTS.md`), but R02 had no preferred channel. Nothing else
  replicates: R02's preferred window (1.5–2.5 s) is not preferred here.
* **Selection stability of `chosen_bits` (Kuncheva / Jaccard, mean over recordings):**

  | Method, protocol | Folds and seeds (all pairs) | Same fold, different seed | Same seed, different fold |
  |---|---|---|---|
  | `ga_nsga`, block | **0.065** / 0.23 | 0.138 / 0.28 | 0.085 / 0.23 |
  | `ga_single`, block | **0.101** / 0.40 | 0.272 / 0.53 | 0.081 / 0.39 |
  | `ga_nsga`, cross-session (seeds only) | 0.109 / 0.27 | | |
  | `ga_single`, cross-session (seeds only) | 0.349 / 0.55 | | |

  This is the same picture as R01 and R02: near chance, and more stable across seeds
  than across folds. `ga_nsga` is not more stable than `ga_single`.

## 3. EEGNet (P4, all 22 subjects)

| Method | Block mean ± SD (n = 22) | Cross-session mean ± SD (n = 20) | Trainable params | Infer ms/trial (batch / batch 1) | Block recordings p < 0.05 (uncorr. / Holm, of 42) |
|---|---|---|---|---|---|
| `fbcsp_all` (P2) | **0.353 ± 0.044** | **0.337 ± 0.039** | 168 features | 0.40 (P2) | 10 / 2 |
| `eegnet_32ch` | 0.341 ± 0.028 | 0.325 ± 0.029 | 3107 | 0.121 / 0.87 | 6 / 2 |
| `eegnet_8ch` | 0.347 ± 0.031 | 0.334 ± 0.033 | 2723 | 0.056 / 0.89 | 8 / 1 |

Paired Wilcoxon, two-sided, uncorrected (descriptive):

| Comparison | Block Δ (p, r_rb), n = 22 | Cross-session Δ (p, r_rb), n = 20 |
|---|---|---|
| `eegnet_32ch` − `fbcsp_all` | −0.011 (0.353, −0.23) | −0.012 (0.247, −0.30) |
| `eegnet_8ch` − `fbcsp_all` | −0.005 (0.775, −0.08) | −0.003 (0.498, −0.18) |
| `eegnet_8ch` − `eegnet_32ch` | +0.006 (0.799, +0.07) | +0.009 (0.083, +0.45) |

* EEGNet does not beat the FBCSP baseline, and **8 channels are not worse than 32**
  (if anything, slightly better). This fits the parietal-occipital channels holding
  whatever signal there is, and 32 channels giving more parameters to overfit on
  about 90 training trials.
* **Early stopping often keeps the first epoch.** In 32 % (`32ch`) and 38 % (`8ch`)
  of block folds, the lowest validation loss is at epoch 0, so the restored weights
  are after one epoch of training. Those folds score 0.329 / 0.338, against
  0.350 / 0.351 for the other folds. The median best epoch is 18 (32ch) and 10 (8ch),
  and the median run length is 62–69 epochs (patience 50). The validation set is 20 %
  of about 90 trials (about 18 trials, 6 per class), so the stopping signal is very
  noisy. EEGNet here is "EEGNet with the frozen §k2 training recipe"; whether a better
  recipe helps is a P5 question, not a reason to change P4.
* Seed SD of a recording's BA is 0.033 (32ch) and 0.036 (8ch), as large as for the GA.
* Cross-session drop (ses-02 block − cross-session): `eegnet_32ch` +0.013,
  `eegnet_8ch` +0.004 (n = 20), smaller than `fbcsp_all` (+0.027).

## 4. Points for integration and P5

1. **`run_eegnet.py --device` changes `config_hash` (CONTRACT_ISSUES R07-11).**
   `main()` writes `cfg["eegnet"]["device"]` before `config_hash(cfg)`. So
   `--device cuda` gives `3b7bc9400a` instead of `08e124167b`, which contradicts §k9.
   R07's first EEGNet pass did this. Its predictions were bit-identical, but it was
   discarded and re-run without the flag. The committed files all have `08e124167b`.
   The fix (hash before applying CLI overrides, or drop `device` in `config_hash`)
   changes frozen code, so it waits until after P4.
2. **GA is bit-identical across machines.** Before this share was run, R07 had also
   run sub-01..16 locally (see 3). All 264 R01/R02 GA files on `main` match those
   R07 runs exactly, fold by fold (BA, `chosen_bits`, `y_pred`, `test_trial_id`,
   `n_evals`). With P2_REVIEW §1, the whole pipeline is reproducible across R01, R02
   and R07 given the same cache. EEGNet on CUDA was also bit-identical between two
   R07 passes (`cudnn.deterministic`).
3. **Disclosure (ANALYSIS_PLAN §6):** R07 ran the full GA and EEGNet grid (all
   subjects, plus `random`-protocol and viewing-period runs from local, uncommitted
   runner changes) on 2026-11-08 between 18:30 and 20:12 +0530. That was before the
   analysis plan was pushed (22:49), which states that no P3 GA result existed.
   R02 could not have seen these results: they were never committed or pushed. The
   committed files are a fresh run started at 23:10, after the plan, and are
   bit-identical to the earlier run (point 2). The earlier local files are not used
   in any analysis.
4. **Checkpoints:** as R01 noted, `results/raw/partial/` keeps all fold checkpoints
   (git-ignored, not committed).
5. **Timings are host-specific:** `infer_time_ms` and `train_time_s` come from three
   machines (`fbcsp_all` from P2, GA from three hosts, EEGNet from R07's GPU).
   Compare them within a host only.

## 5. Status / next

* R07 share of P3 and all of P4: done. GA 108/108 results with 108/108 side files.
  EEGNet 372/372. One config hash, completeness check clean for all five methods.
* Next: the pre-registered group analysis (`docs/RESULTS.md`, separate PR), then
  the P5 items (position-only control, random-chromosome control at matched cost,
  §k12 re-examination).
