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
