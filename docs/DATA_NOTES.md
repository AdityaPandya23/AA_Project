# Data notes: Gao et al. Visual Imagery dataset (figure task)

Owner: R02. Source: figshare article 30227503 (DOI
[10.6084/m9.figshare.30227503](https://doi.org/10.6084/m9.figshare.30227503)),
version 4, 22 subject zips + top-level BIDS files. Paper: Gao et al., *Sci Data*
13, 194 (2026), doi:10.1038/s41597-025-06512-5.

Download with `python -m scripts.download` (resumable, MD5-checked). The data
lands in `$AA_DATA_DIR`, default `data/raw/`, which is the BIDS root
(`config.yaml: paths.data_dir`). The zips stay in `data/raw/_zips/`.

## 1. BIDS layout

```
data/raw/
  README  dataset_description.json  participants.tsv/.json  electrodes.tsv
  coordsystem.json  task-{AVI,FVI,OVI}_events.json
  sub-XX/ses-YY/eeg/sub-XX_ses-YY_task-<T>_{eeg.bdf, eeg.json, channels.tsv, events.tsv}
```

* Tasks: `AVI` = animals, **`FVI` = figures (our task)**, `OVI` = objects.
  `config.yaml: dataset.bids_task: FVI`; `src.io` maps `"figure"` → `"FVI"`.
* There is no `stimuli/` or `code/` folder in the BIDS tree. On figshare these
  are loose files (`Figure_*.jpg`, `Preprocess.py`, …); they are not downloaded.
* `dataset_description.json` declares BIDS 1.1.1. `mne_bids.read_raw_bids` works
  but warns about three things. (a) channels.tsv has 32 rows but the BDF has 33
  channels (the extra one is `Status`), so mne_bids does not apply the TSV and
  `src.io.read_raw` sets channel types itself. (b) The coordinate frame
  `standard_1020` and unit `millimetres` are not BIDS-valid, so electrodes.tsv
  is skipped. (c) participants.tsv has extra columns. All three are harmless.
* Amplifier: Neuracle NeuSenW32, converted to BDF in MATLAB. Fs = 1000 Hz.
  Recorded reference **CPz**, ground Fpz. Power line 50 Hz. No software
  filters were applied.

## 2. Channels (all 32 are EEG; no EOG/ECG/EMG)

BDF / channels.tsv order (this is the EEGNet cache `ch_names` order):

```
Fpz Fp1 Fp2 Fz F3 F4 F7 F8 FCz FC3 FC4 FT7 FT8 Cz C3 C4
T7 T8 CP3 CP4 TP7 TP8 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2
```

There is also a `Status` stim channel (index 32), which `preprocess_raw` drops.
In sub-01, channels.tsv marks every channel `good`.

### Candidate channels (chromosome bits 0–7)

`[O1, Oz, O2, Pz, P3, P4, P7, P8]` (unchanged from the placeholder, so the
config hash did not change).

| Wanted | Present? | Used | Distance (MNE 10-05 template) |
|---|---|---|---|
| Oz, O1, O2, P3, P4 | yes | same | 0 |
| POz | **no** | **Pz** | 38.3 mm (PO3 39.1, PO4 39.3, Oz 38.1 but already used) |
| PO7 / P7 | PO7 no, P7 yes | P7 | 0 (PO7 → O1 30.1, P7 30.3) |
| PO8 / P8 | PO8 no, P8 yes | P8 | 0 |

Pz was chosen over PO3/PO4 because it is the nearest *unused* electrode to POz
and keeps the set midline-symmetric. Taking one of PO3/PO4 would bias the set to
one hemisphere. PO3/PO4 stay available in the 32-channel EEGNet cache.

## 3. Events, class and imagery onset

`events.tsv` has one row per trial, 120 rows per recording:

| Column | Content |
|---|---|
| `onset` | s from file start: **imagery onset** (trial t = 9 s) |
| `duration` | always 4 (= the imagery period) |
| `type` | `pentagram` / `square` / `circle` |
| `value` | 1 = pentagram, 2 = square, 3 = circle |
| `latency` | `onset` in samples at 1000 Hz |
| `trial` | 1..120, presentation order |

* The same marker is in the BDF `Status` channel: a 1-sample pulse with value
  1/2/3. The baseline is 131071 (0x1FFFF) in some files and 0 in others.
  `io.read_trials(raw=...)` checks that Status and events.tsv agree. In sub-01
  they agree to the sample. The BDF has no annotations.
* **There is only one marker per trial.** Fixation, viewing, mask and rest have
  no triggers of their own.
* Why the marker is the *imagery* onset and not the trial onset: (i)
  `duration` = 4 s = the imagery length; (ii) the authors' `Preprocess.py`
  epochs `tmin=-0.2, tmax=4.0` around it; (iii) occipital evoked transients in
  sub-01 (median over trials, CAR, 2–30 Hz) occur at marker −9 s (fixation
  onset, t = 0), −2 s (mask, t = 7), 0 s (imagery cue, t = 9) and +4 s (rest,
  t = 13). These are the phase changes the timeline predicts if the marker
  is at t = 9.
* So `read_trials` returns `imagery_onset_s` = marker and `onset_s` = marker −
  `trial_phases_s.imagery[0]` (9.0). Imagery epochs are cut at the marker. The
  viewing epoch is cut at marker − 6 s (t = 3). Neither is a hard-coded trial grid.
* Inter-marker interval within a block is 17.11–17.13 s (nominal 17 s).
* ⚠ **Open point (viewing window).** I see no clear evoked onset at marker −6 s,
  where the picture should appear (t = 3). There is one at about −8 s instead,
  which could be the stimulus appearing at t ≈ 1 or a blink/saccade after
  fixation onset. I could not read the paper (paywalled). The view cache uses
  the documented t = 3–7 s. Whoever runs the viewing sanity check: if
  viewing does not beat imagery, check this first.

### Blocks

There are no block markers. Blocks are separated by self-paced pauses of
165–350 s, against ~17 s between trials. `io.read_trials` starts a new block at
any gap > 2 × the median interval and requires exactly 4 blocks. Every
recording checked has 4 blocks × 30 trials.

## 4. Trial order within blocks

* Each block holds **10 circle + 10 square + 10 pentagram, randomly
  interleaved**, not grouped. The number of class runs per block is 20–24,
  against 21.0 ± 2.5 for a random permutation of 10/10/10. Runs of up to 5
  identical shapes occur (e.g. `ppppscpc…`).
* The order is a fixed pseudo-random sequence chosen per recording, but
  **sequences are reused**. Among the first 20 recordings there are only 10
  distinct sequences. One sequence (sub-01's) appears in 9 recordings: sub-01
  ses-01 *and* ses-02, plus ses-02 of subjects 02, 03, 04, 05, 06, 08, 11.
  sub-03/07 ses-01 share another, as do sub-06/10 ses-01. The full table is
  produced by `tests/test_real_data.py` → see §6.

**Leakage implications**

* *Leave-one-block-out* (our within-session protocol) is sound. Every block is
  class-balanced and has the same structure, and test trials are never temporal
  neighbours of training trials.
* *Random stratified k-fold* (the authors' protocol, our `random`) puts
  adjacent trials (17 s apart, often the same class in runs of 3–5) into train and test.
  Slow drifts (electrode impedance, alertness, alpha level) then become
  class-predictive. Expect `random` > `block`, and report `random` only for
  comparison with the literature.
* *Cross-session* when both sessions share the same sequence (sub-01 so far):
  class is confounded with trial position in exactly the same way in train and
  test. Any position-locked effect (e.g. first trials after a pause, fatigue
  over a block) can then transfer as spurious "shape" information. The effect
  is probably small, but a position-only control is cheap: predict class from
  `trial_id % 30` or `block` only. The ses-01→ses-02 pairs that share a sequence
  are listed in §6.
* Because sequences are shared across subjects too, any future cross-subject
  analysis has the same confound.

## 5. Trials per class / session, and missing data

* Raw: every figure-task recording has **120 trials = 40 per class**, 4 blocks
  of 30 (10/10/10).
* **sub-09 and sub-10 have only ses-01** (the dataset README says so, and the
  files agree). They are skipped for cross-session, so there are 42 recordings
  in total.
* The README also excludes sub-08 ses-02 for the *animal* task only. Its
  figure task is present and is used.

## 6. Preprocessing (as implemented in `src/preprocess.py`)

1. Read the BDF (V), keep the 32 EEG channels, and interpolate channels marked bad in
   channels.tsv (none in sub-01).
2. Notch at 50, 100, …, 450 Hz (MNE FIR notch, at 1000 Hz).
3. Band-pass 1–40 Hz on the continuous data (MNE zero-phase FIR).
4. Common average reference over the 32 channels.
5. Resample to 250 Hz.
6. Epochs (µV): imagery = marker + [−1, 5) s, viewing = marker − 6 + [−1, 5) s
   (1 s padding for the per-band filters).
7. **Rejection:** peak-to-peak of the unpadded broadband imagery epoch on the 8
   candidate channels > **150 µV** (`preprocess.reject`). The trial is dropped
   from all three caches. Rejection uses the candidate channels, not all 32,
   because there is no EOG and no ICA: blinks on Fp1/Fp2/Fpz exceed 150 µV in
   most trials. On sub-01 an all-channel rule at 150 µV would drop 47/120
   (ses-01) and 107/120 (ses-02) trials. Consequence: EEGNet sees blink
   activity on frontal channels in kept trials.
8. FBCSP / view caches: Butterworth order 4 `sosfiltfilt` per band on the
   padded epoch, crop to [0, 4) s, 7 windows of 250 samples. EEGNet cache: crop
   of the 1–40 Hz epoch (no second filter).

`cache/manifest.csv` has per-recording counts. Extra npz keys:
`rejected_trial_id`, `n_rejected`, `n_out_of_range`, `reject_ptp_uv`, `ptp_uv`.

### Per-recording result

_Filled in after the full build; see `cache/manifest.csv`._
