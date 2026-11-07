# Project: stability-aware channel, band and window selection for EEG visual imagery

## Goal

Classify **imagined shapes** (circle, square, pentagram; 3 classes, chance 33.3 %) from EEG.

## Data

Gao et al., *Visual Imagery* dataset, figshare DOI
[10.6084/m9.figshare.30227503](https://doi.org/10.6084/m9.figshare.30227503).

* 22 subjects, up to 2 sessions each. BIDS layout, `.bdf` files.
* 32 channels at 1000 Hz.
* **Figure task**: 120 trials per session (40 per shape) in 4 blocks.
* Trial timeline (s): fixation 0–3 · viewing 3–7 · mask 7–9 · **imagery 9–13** · rest 13–17.

## Fixed design

| Item | Value |
|---|---|
| Resampling | 250 Hz |
| Imagery epoch | 4 s = 1000 samples, from imagery onset (trial t = 9 s) |
| Candidate channels | 8 parietal-occipital, placeholder `[O1, Oz, O2, Pz, P3, P4, P7, P8]` (R02 to confirm against `channels.tsv`) |
| Bands | theta 4–7, alpha 8–12, beta 13–30, lowgamma 30–40 Hz |
| Windows | 1.0 s long (250 samples), starts 0.0, 0.5, …, 3.0 s after imagery onset, so 7 windows |
| Chromosome | 19 bits = 8 channel + 4 band + 7 window genes. Valid if channels ≥ 2, bands ≥ 1, windows ≥ 1 (otherwise repaired) |
| Cost | `(n_ch/8 + n_bands/4 + n_win/7) / 3` |
| FBCSP | one-vs-rest, 1 filter pair per class per band-window, so 6 log-variance features per selected band-window |
| Classifier | LDA, `solver="lsqr"`, `shrinkage="auto"` |
| GA | NSGA-II (pymoo), pop 30, 20 generations, inner CV 3 folds, seeds 0, 1, 2 |

## Methods

* **Proposed (`ga_nsga`).** A stability-aware multiobjective GA (NSGA-II) chooses
  channels, bands and windows. Its objectives are inner-CV balanced accuracy
  (maximise), inner-CV SD (minimise) and cost (minimise). The final chromosome is
  the lowest-cost Pareto solution within 1 SD of the best accuracy. It is then
  evaluated with FBCSP + shrinkage LDA.
* **Baseline 1 (`fbcsp_all`).** FBCSP + LDA on all components (all 19 bits on).
* **Baseline 2 (`ga_single`).** GA on accuracy only.
* **Comparison (`eegnet`).** EEGNet (braindecode) on broadband 1–40 Hz, all 32
  channels, using the same splits.

## Evaluation

* **Within-session:** leave-one-block-out CV (4 folds). GA selection is nested
  inside the training folds only, using an inner 3-fold CV over the 3 training blocks.
* **Cross-session:** train on ses-01, test on ses-02.
* **Sanity check:** the same FBCSP pipeline on the viewing period (3–7 s).
* Metric: balanced accuracy. Statistics are computed at subject level after
  averaging over GA seeds.

Interfaces between modules are fixed in [CONTRACTS.md](CONTRACTS.md).
