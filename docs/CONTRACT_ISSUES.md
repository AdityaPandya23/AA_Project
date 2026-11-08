# Contract issues found while implementing P1b / P1d

Found while implementing `src/cv.py`, `src/fbcsp.py` and `src/analysis.py`
(branch `features`). **No contract was changed.** Each item says what the code
does now and what I propose. Items marked *decision needed* need agreement before
anyone edits `CONTRACTS.md`.

## 1. `concat_sessions` output fails the cache validator (§e vs `cache.py`)

`concat_sessions` offsets ses-02 blocks by +4 (blocks 4..7), but
`cache._check_labels` requires `block ∈ 0..3`. The concatenated dict can
therefore never be passed through `validate_fbcsp`/`validate_eegnet`. Its
`trial_id` is also not strictly increasing, because ses-02 restarts at 0.

*Now:* documented in the `concat_sessions` docstring ("do not validate it").
`concat_sessions` also adds an extra key `session_idx` (0/1).
*Proposal:* state in §e that the concatenated dict is not a cache and is not validated.

## 2. `inner_splits` when `train_idx` spans fewer than `n` blocks (§e)

`StratifiedGroupKFold(n)` raises if there are fewer than `n` groups. This can
happen if a block is missing for a subject, or with a non-block outer protocol.

*Now:* warns (`RuntimeWarning`) and falls back to
`StratifiedKFold(n, shuffle=True, random_state=seed)` on `train_idx`. The splits
stay inside `train_idx`, but blocks are not kept intact. The normal cases (3
training blocks; 4 ses-01 blocks for cross-session) use `StratifiedGroupKFold`
exactly as specified.
*Proposal:* add this fallback to §e. **Decision needed** if the GA owner prefers an error.

## 3. `evaluate` "calls `select`" (§f) — implemented in the covariance domain

For speed, `evaluate` does not copy `select(X, bits)`. It computes per-trial
covariances of the full cache once per `X` object (cached), and then slices
channels, bands and windows. This is mathematically identical, and
`tests/test_fbcsp.py::test_evaluate_equals_public_fit_predict` checks it against
`FBCSPLDA().fit(select(X, bits)[tr]).predict(select(X, bits)[te])` per split.
Invalid bits still raise `ValueError`.

Consequences for **`src/ga.py`** (not mine; please read):

* Pass the **same `X` array object** on every call (don't copy or slice it per
  call), otherwise the cache misses and calls are about 5× slower.
* Don't modify `X` in place between calls, or call `fbcsp.clear_cache()` if you do.
* `evaluate` limits BLAS to 1 thread internally (multi-threaded BLAS made these
  tiny matrices ~5× slower). Parallelise across individuals, folds or subjects
  with joblib instead.
* `evaluate` returns one extra key, `y_pred` (per-split predictions). This is
  allowed by §0 "unknown keys".

*Proposal:* reword §f to "applies `chromosome.select` semantics to `X`".

## 4. Open item 6 (CSP details): choices made

Recorded in the `src/fbcsp.py` module docstring:

* trial covariance = `Xc Xc^T / T` with the channel mean removed. Class and rest
  covariances are plain means of unnormalised trial covariances (= MNE's
  `cov_est="concat"`). There is no per-trial trace normalisation.
* `cov_reg = 0.0` by default (the contract default). It adds
  `cov_reg·trace/n_ch·I` to the class and rest covariances. If the composite is
  numerically singular, a `1e-6·trace/n_ch` ridge is added automatically for that
  band-window only.
* Features = raw `log(var)`, not normalised by the sum over filters (with only
  2 filters per class, normalising would make the pair redundant).
* A `StandardScaler` is applied before LDA (as requested for P1b). With
  `solver="lsqr", shrinkage="auto"`, sklearn already standardises internally for
  the Ledoit-Wolf estimate, so predictions are invariant to the scaler. It is
  harmless and makes `lda_.coef_` comparable across features.
* The LDA maths (`lsqr` + `auto`/float/None shrinkage) is re-implemented in NumPy,
  because sklearn's validation overhead dominated the run time (41 ms → <2 ms
  per fit). `lda_` is still a real `LinearDiscriminantAnalysis` object with
  `coef_`, `intercept_`, `means_`, `priors_` and `covariance_` set. Tests check
  coefficients and predictions against sklearn. Other solvers fall back to plain sklearn.

*Proposal:* the contract owner copies these into "Open items → 6" as settled.

## 5. `paired_test` unit of analysis (§j) — *decision needed*

The `src/analysis.py` stub docstring says "over subjects (and sessions)". §j
says "n = subjects". Two sessions of the same subject are not independent.

*Now:* seeds are averaged, then **sessions are averaged within subject**, so
n = subjects (§j wins, because CONTRACTS.md wins over docstrings).
`analysis.subject_level(..., sessions=["01"])` allows a single-session analysis.

## 6. Per-trial predictions needed for permutation tests and confusion matrices (§i / §j)

`above_chance` (a per-subject permutation test) and the confusion-matrix plot
need per-trial predictions, which §i does not require.

*Now:* the FBCSP experiments add optional per-fold keys
`y_true`, `y_pred` (lists of int, aligned with the fold's test trials),
`test_trial_id`, `n_train`, `n_test` and `test_blocks`. `above_chance` uses
`y_true`/`y_pred` if present. Otherwise it falls back to a one-sided binomial
test on `n_test`, and otherwise returns `p = NaN` with `test = "none"`.
*Proposal:* list `y_true`, `y_pred` and `n_test` under "Optional extra keys" in §i
and ask **R07 (EEGNet) and the GA owner to write them too**, so all methods get
the same test and confusion matrices.

## 7. Where `pareto` lives in a result JSON (§i)

§i says "`pareto` (...) per fold" but not whether it sits inside each `per_fold`
entry or as a top-level list of per-fold lists.

*Now:* `analysis.load_results` accepts both. *Proposal:* put it inside each
`per_fold[k]` entry (`per_fold[k]["pareto"] = [{bits, ba_mean, ba_std, cost}, ...]`).

## 8. `seed` for the random protocol (§i)

"Deterministic methods (`fbcsp_all`) are still written with seed-0." Under
`protocol="random"`, however, the seed selects the `random_splits` partition, so
the results differ by seed.

*Now:* `experiments/sanity.py` writes one `_random` file per split seed
(`seed-0` … `seed-4`). *Proposal:* state that for `protocol=random` the seed is
the split seed.

## 9. Sanity-check outputs that are not results JSONs

The shuffled-label null distribution has no method name in
`config.evaluation.methods`, so it is not written as a results JSON. It goes into
`results/sanity/sanity_summary.json` / `.md` together with the block-vs-random
and viewing-vs-imagery summaries. The viewing runs are written as
`fbcsp_all_view` (allowed by the `_view` suffix rule).

## 10. Small additions (no conflict)

* `load_results` also accepts a list of directories, and adds the columns
  `n_folds` and `synthetic`. It warns when synthetic and real results are mixed.
* `analysis.py` adds `subject_level`, `confusion`, `selection_matrix`,
  `selection_stability`, `jaccard`, `kuncheva`, `cross_session_drop`,
  `pareto_front`, `plot_subject_bars`, `plot_confusion`,
  `plot_selection_heatmap`, `plot_stability` and `make_figures`. The contract's
  signatures are unchanged.
* Results produced from the synthetic cache carry `"synthetic": true`.

---

# R07 (GA / EEGNet)

Found while implementing `src/ga.py`, `src/eegnet.py`, `experiments/run_ga.py`
and `experiments/run_eegnet.py` (branch `ga`). **No contract was changed.**
R01's items above are unchanged. R07 follows them: items 3 (same-`X` rule, no
in-place edits, BLAS pinned to 1 thread → joblib across folds, subjects and seeds), 6
(per-fold `y_true`, `y_pred`, `n_test`, `test_trial_id`, `n_train`,
`test_blocks`) and 7 (`pareto` inside each `per_fold[k]`).

## R07-1. EEGNet method names do not match §i — *decision needed*

The task specified `results/eegnet_32ch/` and `results/eegnet_8ch/`. §i only
defines `eegnet` (all 32 channels) and `eegnet_gach` (GA-chosen channels). EEGNet
on the fixed 8 FBCSP candidate channels is not a contract method, and neither
name is in `config.evaluation.methods`.

*Now:* `run_eegnet.py` writes `eegnet_32ch` / `eegnet_8ch` by default.
`--method-name eegnet --channels 32` writes the contract name instead.
*Proposal:* add `eegnet_8ch` to §i and `config.evaluation.methods`, and either
rename `eegnet_32ch` to `eegnet` or list both.
**Request for R01 (`src/analysis.py`):** whichever names are agreed, add them to
`METHOD_ORDER`, `METHOD_LABELS` and `METHOD_MARKERS`. Unknown methods already
work (they sort last and show the raw name), so nothing is broken in the meantime.

## R07-2. `evaluate_eegnet` has no seed argument (§h)

The signature `evaluate_eegnet(X, y, splits, ch_idx=None, cfg=None)` is fixed
by `tests/test_contracts.py`, but EEGNet is run with seeds 0, 1 and 2.

*Now:* the seed is read from `cfg["eegnet"].get("seed", 0)`, and the device from
`cfg["eegnet"].get("device")`. `run_eegnet.py` sets the seed on a per-run copy.
`config_hash` is computed on the design config without these keys, so all seeds
share one hash. *Proposal:* list both as optional keys in §h.

## R07-3. `choose_from_pareto`: tie for `b*` is unspecified (§g)

If several entries share the highest `ba_mean` but have different `ba_std`,
the threshold `b*.ba_mean - b*.ba_std` depends on which one is `b*`.
*Now:* `b*` is chosen by highest `ba_mean`, then lower `ba_std`, then lower `cost`,
then smallest `to_str`. This matches the order of the final tie-break.
*Proposal:* write this into §g.

## R07-4. GA operators are not in the contract or `config.yaml`

They affect the results but not `config_hash`.
*Now:* `BinaryRandomSampling` (each bit Bernoulli(0.5), then repair, which is
the same as `random_valid(p=0.5)`), `TwoPointCrossover`, `BitflipMutation` with
per-gene probability 1/19, `eliminate_duplicates=True`, and offspring =
`pop_size`. That is about 600 evaluations per run, about 570 of them unique.
*Proposal:* add `ga.crossover`, `ga.mutation_prob_var` and `ga.sampling_p` to
`config.yaml`.

## R07-5. `infer_time_ms` definition (§i)

*Now:* aligned with R01's FBCSP experiments, i.e. the batch prediction of the
test fold divided by `n_test`. The GA uses `experiments.common.infer_latency_ms`
directly. EEGNet also includes the host→device copy and `cuda.synchronize`.
EEGNet also writes `infer_time_ms_batch1` (batch size 1, which is what an online BCI sees).
*Proposal:* state "batch prediction / n_test" in §i.

## R07-6. `train_time_s` for the GA under parallel folds (§i)

Folds run as separate processes, so `train_time_s` is the **sum** of the
per-fold wall times (GA plus final fit), not the elapsed time of the run. The extra
key `ga_time_s` (top level and per fold) holds the GA search alone.

## R07-7. Pareto side files and crash-resume checkpoints

The fronts are stored in `per_fold[k]["pareto"]` (R01 item 7) **and** in a side
file `{results_dir}/pareto/{method}/sub-XX_ses-YY_seed-Z.json`.
Checkpoints are in `{results_dir}/raw/partial/{method}/..._fold-k.json`
(git-ignored). `analysis.load_results` globs `*/*.json`, so it reads neither,
which is correct. *Proposal:* §j should reserve `pareto/` and `raw/` as
non-method directories.

## R07-8. EEGNet training details not in config (open item 9)

They are recorded in the `src/eegnet.py` docstring:
* early stopping on the lowest validation cross-entropy (stratified
  `val_fraction` taken from the training fold only), with the best weights restored;
* per-channel z-scoring with statistics from the fit part of the training fold;
* `final_layer_with_constraint=True` (pinned, because braindecode 2.0 warns its default will change);
* braindecode 2.0 renamed `EEGNetv4` to `EEGNet`, and the code imports either.

*Proposal:* move these to `config.eegnet` so they change `config_hash`.

## R07-9. GA runtime with R01's `fbcsp.evaluate`

Measured on R07 (i5-13450HX) with the synthetic cache (the same shapes as real data):
about 0.004 s per 3-fold `evaluate`, so about 2.2 s per GA fold at the config budget
(pop 30, 20 generations). The full GA grid (22 subjects × 2 sessions × 2 methods ×
3 seeds, plus cross-session) is about 1,200 fold jobs, which is minutes on one
machine. **No smaller budget is needed.** There is even room to raise `pop_size` or
`n_generations` if the Pareto fronts look unconverged on real data.

## R07-10. Subject labels on the CLI

The CLIs accept `17`, `sub-17`, `17..22`, `sub-17..sub-22` and `17-22`, and
always write the contract's 2-digit label.

## Requests for R01 (no change needed in `cv.py` or `fbcsp.py`)

* `src/analysis.py`: method names from R07-1, once agreed.
* Nothing else. `cv.inner_splits`, `cv.concat_sessions` and `fbcsp.evaluate`
  (including `y_pred` and the covariance cache) work as they are, and
  `tests/test_ga.py::test_real_fbcsp_cache_is_hit` checks that one GA run
  computes the covariances once.
