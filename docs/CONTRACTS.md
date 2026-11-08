# Interface contracts

This document fixes everything that more than one machine (R02, R01, R07) codes
against. **If you need to change anything here, open a PR that edits this file
first and get agreement from the other two machines.** The module docstrings in
`src/` are copied from this file. If the two disagree, this file wins.

Contents: [Conventions](#0-conventions) · [a. FBCSP cache](#a-fbcsp-cache) ·
[b. EEGNet cache](#b-eegnet-cache) · [c. Viewing cache](#c-viewing-cache) ·
[d. chromosome](#d-srcchromosomepy) · [e. cv](#e-srccvpy) · [f. fbcsp](#f-srcfbcsppy) ·
[g. ga](#g-srcgapy) · [h. eegnet](#h-srceegnetpy) · [i. Results JSON](#i-results-json) ·
[j. analysis](#j-srcanalysispy) · [Supporting modules](#supporting-modules) · [k. Decisions after P1](#k-decisions-after-p1-binding-from-tag-p1-integrated) ·
[Open items](#open-items-status-at-p1-integrated)

---

## 0. Conventions

| Item | Convention |
|---|---|
| Subject label | 2-digit zero-padded string, `"01"`..`"22"`. Synthetic subjects: `"91"`,`"92"`,`"93"`. |
| Session label | `"01"` or `"02"`. Cross-session results use `"01to02"` (see §i). |
| File stem | `sub-{subject}_ses-{session}` |
| Class labels | `circle=0, square=1, pentagram=2` (`config.yaml: classes`) |
| Trial order | Rows are in original presentation order. `trial_id` = 0-based index of the trial in the session's original 120-trial sequence, so it stays meaningful after trials are rejected. |
| Block | `int64` in `0..3` = block index within the session. |
| Integer arrays | `int64` |
| String arrays in `.npz` | NumPy unicode (`<U…`), **never** object arrays. All caches must load with `np.load(path, allow_pickle=False)`. |
| Signal units | microvolts (µV), `float32` |
| Sampling rate | 250 Hz after resampling. |
| Metric | Balanced accuracy, `sklearn.metrics.balanced_accuracy_score`. Chance = 1/3. |
| Index arrays | `train_idx` and `test_idx` are always `int64` **absolute** indices into the trial axis of the arrays passed in, never positions relative to a subset. |
| Randomness | Functions that take `seed` or `rng` must be deterministic given it. `rng` is a `numpy.random.Generator`. |
| Unknown keys | Producers may add extra keys to dicts, npz files and JSON. Consumers must ignore keys they do not know. Required keys may not be removed or renamed. |
| Config | Every function that takes `cfg` takes the dict returned by `src.config.load_config()`. |

Trial timeline (s from trial onset): fixation 0–3, viewing 3–7, mask 7–9,
**imagery 9–13**, rest 13–17. "Imagery onset" = trial onset + 9.0 s. "Viewing
onset" = trial onset + 3.0 s.

---

## a. FBCSP cache

`{cache_dir}/fbcsp/sub-XX_ses-YY.npz`. Written by `src/preprocess.py`, read
through `src.cache.load_cache("fbcsp", ...)`.

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `X` | float32 | `(4, 7, n_trials, 8, 250)` | `[band, window, trial, channel, sample]`. Already band-passed per band. |
| `y` | int64 | `(n_trials,)` | class label 0/1/2 |
| `block` | int64 | `(n_trials,)` | 0..3 |
| `trial_id` | int64 | `(n_trials,)` | see §0. Unique and strictly increasing. |
| `ch_names` | `<U` | `(8,)` | = `config.channels.candidates`, same order |
| `band_names` | `<U` | `(4,)` | = `config.bands.names`, same order |
| `band_edges` | float64 | `(4, 2)` | Hz, `[low, high]` per band |
| `window_starts` | float64 | `(7,)` | seconds relative to **imagery onset**: `0.0, 0.5, …, 3.0` |
| `sfreq` | float64 | `()` | 250.0 |

How it is produced (per band): band-pass the imagery epoch **including
`preprocess.filter.pad_s` of padding on each side** (zero-phase), crop to imagery
onset + `[0, 4)` s (1000 samples), then cut window `w` as samples
`[start_w*250, start_w*250 + 250)`. Windows overlap by 50 %. Filtering happens
before windowing, never per window.

`n_trials` ≤ 120 (trials may be rejected; `trial_id` records which survived).
The 8-channel axis lists **only** the candidate channels.

## b. EEGNet cache

`{cache_dir}/eegnet/sub-XX_ses-YY.npz`

| Key | dtype | Shape | Meaning |
|---|---|---|---|
| `X` | float32 | `(n_trials, 32, 1000)` | broadband 1–40 Hz imagery epoch (imagery onset + `[0, 4)` s) |
| `y` | int64 | `(n_trials,)` | |
| `block` | int64 | `(n_trials,)` | |
| `trial_id` | int64 | `(n_trials,)` | Same trials, same order as the FBCSP cache for that file stem. |
| `ch_names` | `<U` | `(32,)` | all EEG channels in the raw file's order |
| `sfreq` | float64 | `()` | 250.0 |

Invariant: `y`, `block` and `trial_id` are identical to those in
`fbcsp/<same stem>.npz`, so both pipelines see exactly the same trials and
splits. Each candidate channel in `config.channels.candidates` appears in
`ch_names`. Use `src.cache.candidate_channel_index(ch_names, cfg)` to get the
`ch_idx` for EEGNet on the GA-selected channels.

## c. Viewing cache

`{cache_dir}/view/sub-XX_ses-YY.npz`. **Same keys, dtypes and shapes as (a)**,
but the epoch is the viewing period (viewing onset + `[0, 4)` s, i.e. 3–7 s of
the trial), and `window_starts` are relative to viewing onset. Used only as a
sanity check, because perception should decode better than imagery. It has the
same trials as (a).

---

## d. `src/chromosome.py`

**Implemented** (shared, small). The 19-bit layout follows config order:

```
bit  0..7   channels  config.channels.candidates  [O1, Oz, O2, Pz, P3, P4, P7, P8]
bit  8..11  bands     config.bands.names          [theta, alpha, beta, lowgamma]
bit 12..18  windows   config.windows.starts_s     [0.0, 0.5, ..., 3.0]
```

`bits` is a 1-D array-like of 0/1 with length 19. Functions accept list, tuple,
bool array or int array, and return `np.uint8` arrays where they return bits.

| Function | Signature | Contract |
|---|---|---|
| `decode` | `decode(bits) -> (ch_mask, band_mask, win_mask)` | bool arrays of shapes (8,), (4,), (7,) |
| `encode` | `encode(ch_mask, band_mask, win_mask) -> bits` | inverse of decode |
| `is_valid` | `is_valid(bits) -> bool` | channels ≥ 2 and bands ≥ 1 and windows ≥ 1 |
| `repair` | `repair(bits, rng) -> bits` | Returns a **new** valid array. If a group is below its minimum, it switches on uniformly random *off* genes in that group until the minimum is met. It never switches genes off. A valid input comes back unchanged. |
| `cost` | `cost(bits) -> float` | `(n_ch/8 + n_bands/4 + n_win/7) / 3`, in `(0, 1]`. All on = 1.0. |
| `random_valid` | `random_valid(rng, p=0.5) -> bits` | each bit ~ Bernoulli(p), then `repair` |
| `select` | `select(X, bits) -> Xsel` | `X` = FBCSP cache array `(4, 7, n, 8, 250)` → `Xsel` `(n_trials, n_bw, n_ch_sel, 250)`. `n_bw = n_bands_sel * n_win_sel`, ordered **band-major** (`bw = i_band_sel * n_win_sel + i_win_sel`). Raises `ValueError` on invalid bits. Returns a copy. |
| `n_features` | `n_features(bits) -> int` | `6 * n_bw` (3 classes × 1 pair × 2 filters per band-window) |
| `to_str` / `from_str` | `to_str(bits) -> str` / `from_str(s) -> bits` | 19-character `"0"/"1"` string. **This is the JSON representation.** |
| `all_on` | `all_on() -> bits` | the baseline-1 chromosome (19 ones) |
| `describe` | `describe(bits, cfg=None) -> dict` | `{"channels": [...], "bands": [...], "windows": [...]}` names, for logs and plots |

Constants: `N_CH=8, N_BAND=4, N_WIN=7, N_BITS=19, MIN_CH=2, MIN_BAND=1, MIN_WIN=1`.

## e. `src/cv.py`

All functions return `list[tuple[np.ndarray, np.ndarray]]` of absolute
`int64` `(train_idx, test_idx)`, with sorted indices and no overlap.

| Function | Signature | Contract |
|---|---|---|
| `block_splits` | `block_splits(block) -> list` | Leave-one-block-out. One split per distinct block value, in ascending block order. Fold `k` tests block `k`. |
| `inner_splits` | `inner_splits(y, block, train_idx, n=3, seed=0) -> list` | Inner CV used **only** inside a training set, for GA fitness. It uses `StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)` on `train_idx`, with `groups = block[train_idx]` and stratified by `y[train_idx]`. When `train_idx` spans exactly `n` blocks (the block protocol: 3 training blocks), this is leave-one-block-out over those blocks. All returned indices ⊆ `train_idx`. |
| `random_splits` | `random_splits(y, n=5, seed=0) -> list` | `StratifiedKFold(n, shuffle=True, random_state=seed)` over all trials. "random" protocol: it ignores blocks, is optimistic, and exists only for comparison with the literature. |
| `cross_session` | `cross_session(n_train, n_test) -> list` | Exactly one split, `[(arange(n_train), arange(n_train, n_train+n_test))]`. The caller concatenates ses-01 then ses-02 along the trial axis (axis 2 for FBCSP X, axis 0 for EEGNet X). |
| `concat_sessions` | `concat_sessions(d1, d2, kind) -> dict` | Concatenates two loaded cache dicts along the trial axis. `block` of the second session is offset by `+4` so blocks stay unique. Other keys are taken from `d1` after checking they are equal. |

**Nesting rule (non-negotiable):** the GA sees only `train_idx` of the outer
fold. It may call `inner_splits(y, block, train_idx, …)` and nothing else. The
outer test fold is touched exactly once, by the final fit on `train_idx` with
`chosen_bits`.

## f. `src/fbcsp.py`

```python
class FBCSPLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_filter_pairs: int = 1, lda_solver: str = "lsqr",
                 lda_shrinkage: str | float | None = "auto", cov_reg: float = 0.0): ...
    def fit(self, Xsel: np.ndarray, y: np.ndarray) -> "FBCSPLDA": ...
    def predict(self, Xsel: np.ndarray) -> np.ndarray: ...
    def transform(self, Xsel: np.ndarray) -> np.ndarray: ...  # (n_trials, 6*n_bw) features
```

* `Xsel` has shape `(n_trials, n_bw, n_ch_sel, 250)`, the output of `chromosome.select`.
* For each band-window and each class `c` (one-vs-rest), it fits CSP from the
  class-`c` covariance against the rest covariance and keeps 1 filter pair
  (largest and smallest eigenvalue). Features are the log-variance of the 2
  filtered signals. That gives 3 classes × 2 = 6 features per band-window, so
  `transform` returns `(n_trials, 6 * n_bw)`.
* Classifier: `LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")` on all
  features together, multiclass.
* It works with `n_ch_sel = 2` (1 pair = both filters).
* After fit it has `classes_`, `filters_` with shape `(n_bw, 3, 2, n_ch_sel)`, and `lda_`.

```python
def evaluate(X: np.ndarray, y: np.ndarray, bits, splits) -> dict
```

* `X` is the **full** FBCSP cache array `(4, 7, n, 8, 250)`. `evaluate` calls `select`.
* For each split it fits a fresh `FBCSPLDA` on train and predicts test.
* Returns `{"ba_mean": float, "ba_std": float, "per_fold": list[float], "fit_time_s": float}`.
  `ba_std` is the population SD (`np.std`, ddof=0) over folds. `fit_time_s` is the
  total wall time for all folds (fit + predict).

## g. `src/ga.py`

```python
def run_nsga(X, y, block, train_idx, cfg, seed) -> dict
def run_single_objective(X, y, block, train_idx, cfg, seed) -> dict
def choose_from_pareto(pareto: list[dict]) -> np.ndarray
```

* `X` is the full FBCSP cache array. `y` and `block` are full-length.
  `train_idx` are the absolute indices the GA may use (§e nesting rule).
* Fitness of `bits` = `fbcsp.evaluate(X, y, bits, inner_splits(y, block, train_idx, n=cfg["ga"]["inner_folds"], seed=seed))`.
  The inner splits are computed **once** per run, so every individual is scored
  on the same folds.
* Binary genome of 19 bits, with `repair` applied after sampling, crossover and
  mutation. pymoo implementation, `pop_size` and `n_generations` from `cfg["ga"]`.
  `seed` seeds both pymoo and `inner_splits`.
* `run_nsga` objectives (minimised): `[-ba_mean, ba_std, cost]`.
* `run_single_objective` objective: `-ba_mean` (pymoo `GA`).
* Return value, the same shape for both:
  ```python
  {
    "pareto": [{"bits": np.ndarray(uint8, 19), "ba_mean": float, "ba_std": float, "cost": float}, ...],
    "chosen_bits": np.ndarray(uint8, 19),
    "n_evals": int,          # unique chromosomes evaluated
    "time_s": float,
  }
  ```
  * `run_nsga`: `pareto` = the non-dominated set of the final population (unique bits).
    `chosen_bits = choose_from_pareto(pareto)`.
  * `run_single_objective`: `pareto` = the unique individuals of the final
    population. `chosen_bits` = highest `ba_mean`. Ties go to lower `cost`, then
    lexicographically smallest `to_str(bits)`. It does **not** use
    `choose_from_pareto`, because this is the accuracy-only baseline.
* `choose_from_pareto(pareto)`: let `b*` = entry with the highest `ba_mean`.
  Candidates are the entries with `ba_mean >= b*.ba_mean - b*.ba_std`. Return the
  candidate with the lowest `cost`. Ties go to higher `ba_mean`, then lower
  `ba_std`, then lexicographically smallest `to_str(bits)`.

## h. `src/eegnet.py`

```python
def evaluate_eegnet(X, y, splits, ch_idx=None, cfg=None) -> dict
```

* `X` = EEGNet cache array `(n_trials, 32, 1000)`. `ch_idx` = optional channel
  indices into axis 1. `None` means all 32. For method `eegnet_gach`, pass the
  GA-chosen candidate channels mapped via `src.cache.candidate_channel_index`.
* `splits` are exactly the same split objects used for FBCSP on the same file
  stem, so the comparison is fair.
* Early-stopping validation is carved **from the training fold only**
  (`cfg["eegnet"]["val_fraction"]`, stratified). The test fold is never used for
  model selection.
* Hyperparameters come from `cfg["eegnet"]`. `cfg=None` → `load_config()`.
* Returns `{"per_fold": list[float], "ba_mean": float, "ba_std": float, "train_time_s": float}`.
  It may add `"infer_time_ms"` (mean per-trial inference latency).

## i. Results JSON

Path: `{results_dir}/{method}/sub-XX_ses-YY_seed-Z.json`. Use
`src.results.result_path()` and `src.results.write_result()`. The writer
validates the record against the schema below.

* `method` ∈ `fbcsp_all` (baseline 1, all 19 bits on), `ga_single` (baseline 2,
  accuracy only), `ga_nsga` (proposed), `eegnet` (all 32 channels), `eegnet_gach`
  (EEGNet on the GA-chosen channels, optional). Suffix `_view` (e.g.
  `fbcsp_all_view`) marks runs on the viewing cache. Add new methods to
  `config.evaluation.methods` by PR.
* `ses-YY` for `protocol=block` and `protocol=random` is the evaluated session.
  For `protocol=cross_session` it is `01to02` (train ses-01, test ses-02), so the
  files never collide.
* `protocol=random` files add the suffix `_random` before `.json`, i.e.
  `sub-01_ses-01_seed-0_random.json`, so they do not overwrite block results.
* Deterministic methods (`fbcsp_all`) are still written with `seed-0`.

| Field | Type | Meaning |
|---|---|---|
| `method` | str | see above |
| `subject` | str | `"01"` |
| `session` | str | `"01"`, `"02"`, `"01to02"` |
| `seed` | int | |
| `protocol` | str | `block` \| `cross_session` \| `random` |
| `per_fold` | list[obj] | one per outer fold, see below |
| `ba_mean` | float | mean of `per_fold[].ba` |
| `ba_std` | float | SD (ddof=0) of `per_fold[].ba` |
| `train_time_s` | float | total wall time across folds, including GA search |
| `infer_time_ms` | float \| null | mean per-trial prediction latency |
| `config_hash` | str | `src.config.config_hash(cfg)` (10 hex chars, ignores `paths`) |

`per_fold[]` objects:

| Field | Type | Meaning |
|---|---|---|
| `fold` | int | 0-based. For block protocol, fold k tests block k. |
| `ba` | float | balanced accuracy on the outer test fold |
| `chosen_bits` | str \| null | 19-char bit string (`chromosome.to_str`). `null` for `eegnet`. |
| `cost` | float \| null | `chromosome.cost(chosen_bits)`. `null` for `eegnet`. |
| `n_features` | int \| null | `chromosome.n_features` for FBCSP methods, number of trainable parameters for EEGNet |

Optional extra keys: `pareto` (list of `{bits:str, ba_mean, ba_std, cost}`) per
fold, `n_evals`, `git_commit`, `host` (`R01`/`R02`/`R07`).

## j. `src/analysis.py`

All stubs. Signatures are fixed:

```python
def load_results(results_dir) -> pd.DataFrame
    # one row per JSON file: method, subject, session, seed, protocol, ba_mean, ba_std,
    # train_time_s, infer_time_ms, config_hash, mean_cost, mean_n_features, per_fold (list), path
def explode_folds(df) -> pd.DataFrame            # one row per (file, fold)
def aggregate_seeds(df) -> pd.DataFrame          # mean over seeds per (method, subject, session, protocol)
def paired_test(df, method_a, method_b, protocol="block", metric="ba_mean", test="wilcoxon") -> dict
def friedman_test(df, methods, protocol="block", metric="ba_mean") -> dict
def holm(pvalues) -> np.ndarray                   # statsmodels multipletests(method="holm")
def above_chance(df, method, protocol="block", n_perm=..., ...) -> pd.DataFrame
def selection_frequency(df, method) -> pd.DataFrame  # per-gene frequency of chosen_bits
def plot_method_comparison(df, protocol="block", ax=None)
def plot_pareto(pareto, chosen_bits=None, ax=None)
def plot_selection_frequency(freq_df, ax=None)
def plot_cost_vs_accuracy(df, protocol="block", ax=None)
```

Subject-level statistics are computed **after** `aggregate_seeds` (n = subjects),
never across seeds as if they were independent samples.

---

## Supporting modules

| Module | Status | Purpose |
|---|---|---|
| `src/config.py` | implemented | `load_config(path=None)`, `config_hash(cfg)`, `repo_root()` |
| `src/cache.py` | implemented | `cache_path`, `save_fbcsp_cache`, `save_eegnet_cache`, `load_cache`, `validate_fbcsp`, `validate_eegnet`, `candidate_channel_index` |
| `src/results.py` | implemented | `result_path`, `validate_result`, `write_result`, `make_fold_record` |
| `src/synth.py` | implemented | fake caches for subjects 91–93 (see README) |
| `src/io.py` | stub | BIDS reading: `list_subject_sessions`, `read_raw`, `read_trials`, `read_channels_tsv` |
| `src/preprocess.py` | stub | `preprocess_raw`, `epoch_period`, `make_fbcsp_array`, `build_caches`, `build_all` |

`read_trials` returns a DataFrame with columns
`trial_id:int, onset_s:float (trial onset in raw time), shape:str, y:int, block:int`.

---

## k. Decisions after P1 (binding from tag `p1-integrated`)

Settled by R02 (contract owner) at P1.5, 2026-11-08, from the items in
`docs/CONTRACT_ISSUES.md`. Where this section and §a–§j differ, this section
wins. **`config.yaml` is unchanged, so `config_hash` stays `08e124167b`** and the
P2 results stay valid.

| # | Topic | Decision |
|---|---|---|
| k1 | EEGNet method names (§i; CONTRACT_ISSUES R01 #12 / R07-1) | Keep what `run_eegnet.py` writes. **`eegnet_32ch`** = the contract's `eegnet` (all 32 channels); `analysis` treats the two as one method. **`eegnet_8ch`** = new method: EEGNet on the 8 fixed candidate channels. `eegnet_gach` (GA-chosen channels) stays allowed but is **not planned** for P4. The method list in §i is authoritative; `config.evaluation.methods` is informational and is not edited (that would change `config_hash`). |
| k2 | GA operators and EEGNet training details (R07-4, R07-8) | **Stay in code, frozen at `p1-integrated`.** They are not moved into `config.yaml`. The values are those in CONTRACT_ISSUES R07-4 and R07-8: binary random sampling p = 0.5 + repair, two-point crossover, bit-flip mutation 1/19 per gene, duplicate elimination, offspring = pop; EEGNet early stopping on validation cross-entropy with best weights restored, per-channel z-scoring from the fit part, `final_layer_with_constraint=True`. Every result JSON records `git_commit`, which pins them. Changing any of them after the tag requires a new tag and re-running the affected methods. |
| k3 | `inner_splits` with fewer than `n` blocks (§e; R01 #2) | Accepted as implemented: `RuntimeWarning` + `StratifiedKFold(n, shuffle=True, random_state=seed)` on `train_idx`. It never triggers in the planned protocols (3 training blocks; 4 ses-01 blocks for cross-session). |
| k4 | Unit of paired tests (§j; R01 #5) | n = subjects: seeds averaged, then sessions averaged within subject. Single-session analyses via `analysis.subject_level(..., sessions=[...])`. |
| k5 | Per-fold prediction keys (§i; R01 #6) | `y_true`, `y_pred`, `n_test`, `test_trial_id`, `n_train` and `test_blocks` are **expected** in every `per_fold[k]` for all methods (all runners already write them). `above_chance` uses the label-permutation test on them. |
| k6 | Where `pareto` lives (§i; R01 #7, R07-7) | Both: `per_fold[k]["pareto"]` in the result JSON and the side file `{results_dir}/pareto/{method}/…`. `{results_dir}/raw/partial/` holds crash-resume checkpoints (git-ignored). |
| k7 | `seed` under `protocol=random` (§i; R01 #8) | The seed is the `random_splits` split seed (`seed-0` … `seed-4`). |
| k8 | Tie for `b*` in `choose_from_pareto` (§g; R07-3) | `b*` = highest `ba_mean`, then lower `ba_std`, then lower `cost`, then smallest `to_str(bits)`. |
| k9 | EEGNet seed and device (§h; R07-2) | Optional `cfg["eegnet"]["seed"]` and `cfg["eegnet"]["device"]`, set per run on a copy of the config by `run_eegnet.py`. Not in `config.yaml`; excluded from `config_hash`. |
| k10 | `infer_time_ms` (§i; R07-5) | Batch prediction of the test fold ÷ `n_test`. EEGNet also writes `infer_time_ms_batch1` (online-BCI latency). |
| k11 | `train_time_s` for the GA (§i; R07-6) | Sum of the per-fold wall times (GA search + final fit). `ga_time_s` holds the search alone. |
| k12 | "Within 1 SD" rule (§g; open item 8, R01 #11) | **Unchanged for P3.** Re-examine after P3 using the stored Pareto fronts; any change is a new contract decision and a re-run of `ga_nsga`. |
| k13 | Results location of the baseline | `experiments/baseline.py` writes `results/baseline_all/fbcsp_all/`, and `experiments/sanity.py` writes `results/sanity/`. GA and EEGNet runners write `results/{method}/`. `analysis.load_results` takes a list of directories. |

---

## Open items (status at `p1-integrated`)

1. **Candidate channels:** ✅ `[O1, Oz, O2, Pz, P3, P4, P7, P8]` confirmed (DATA_NOTES §2).
2. **All 32 channels EEG?** ✅ Yes, there is no EOG; EEGNet stays at 32 (DATA_NOTES §2).
3. **Block and trial-onset derivation:** ✅ the single marker is the imagery onset; blocks come from the pauses between markers (DATA_NOTES §3).
4. **Reference and artifact handling:** ✅ CAR, plus peak-to-peak > 150 µV on the candidates (DATA_NOTES §6, `config.preprocess.reject`).
5. **Filter design:** ✅ as in `config.preprocess` (FIR 1–40 Hz on continuous data; Butterworth order 4 per band on padded epochs).
6. **CSP details:** ✅ recorded in CONTRACT_ISSUES §4.
7. **"Stability" = `ba_std` across inner folds:** ✅ unchanged. Selection stability (Jaccard/Kuncheva) is reported in P5 but is not an objective.
8. **"Within 1 SD":** ⏳ see k12 (after P3).
9. **EEGNet hyperparameters and budget:** ✅ `config.eegnet` + k2; `eegnet_gach` not planned (k1).
10. **Cross-session GA:** ✅ `StratifiedGroupKFold` over the 4 ses-01 blocks with n = 3.
11. **Subjects with only ses-01:** ✅ sub-09 and sub-10 (20 cross-session subjects).
12. **Machine ownership:** ✅ R02 data/integration, R01 cv/fbcsp/analysis, R07 ga/eegnet. P3 subject split: R02 sub-01..08, R01 sub-09..16, R07 sub-17..22; R07 runs all EEGNet.
