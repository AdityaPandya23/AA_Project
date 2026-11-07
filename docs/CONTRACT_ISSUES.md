# Contract issues (raised from branch `ga`)

Concerns found while implementing `src/ga.py`, `src/eegnet.py`,
`experiments/run_ga.py` and `experiments/run_eegnet.py`. **CONTRACTS.md was
not changed.** Each item says what the code does now and what change is proposed.

## 1. EEGNet method names do not match §i

The task asked for `results/eegnet_32ch/` and `results/eegnet_8ch/`. §i only
defines `eegnet` (all 32 channels) and `eegnet_gach` (GA-chosen channels). An
EEGNet on the fixed 8 FBCSP candidate channels is not a contract method, and
neither name is in `config.evaluation.methods`.

* **Now:** `run_eegnet.py` writes `eegnet_32ch` / `eegnet_8ch` by default.
  `--method-name eegnet --channels 32` writes the contract name instead.
* **Proposal:** add `eegnet_8ch` to §i and `config.evaluation.methods`, and
  either rename `eegnet_32ch` to `eegnet` or list both. Whoever writes
  `analysis.py` needs to know which one is used.

## 2. `evaluate_eegnet` has no seed argument (§h)

Seeds 0, 1 and 2 are run for EEGNet, but the fixed signature
`evaluate_eegnet(X, y, splits, ch_idx=None, cfg=None)` has no `seed`.

* **Now:** the seed is read from `cfg["eegnet"].get("seed", 0)`, and
  `cfg["eegnet"].get("device")` picks the device. `run_eegnet.py` sets the seed on a
  per-run copy of the config. `config_hash` is computed on the design config
  without that key, so all seeds share one hash.
* **Proposal:** document `cfg["eegnet"]["seed"]` and `["device"]` as optional
  keys in §h.

## 3. `src/cv.py` and `src/fbcsp.py` are still stubs

* **Now:** while `cv.*` raises `NotImplementedError`, `ga._inner_splits` and
  `experiments/_common.py` fall back to implementations that follow §e literally,
  and warn. When the real `cv` lands, it is used automatically.
* `fbcsp`: `experiments/_standin_fbcsp.py` follows §f (trace-normalised
  covariances, `eigh(Ca, Ca+Cb)`, raw log-variance). It is used only with
  `--allow-standin-fbcsp`, and every result it produces is tagged
  `"fbcsp_impl": "standin"`. Without the flag, `run_ga.py` refuses to run.
  Results made with the stand-in must be deleted before the paper runs.
* **For R01:** the GA budget estimate assumes `fbcsp.evaluate` with 3 inner
  folds takes about 0.05 s (NSGA-chosen chromosomes) to 0.08 s (larger
  accuracy-only ones) per call on 90 training trials. A slower implementation
  scales the GA runtime linearly. See the benchmark in the PR description.

## 4. Where the Pareto side files and checkpoints live (§i / §j)

* **Now:** the per-fold Pareto fronts go to
  `{results_dir}/pareto/{method}/sub-XX_ses-YY_seed-Z.json`. Crash-resume
  checkpoints go to `{results_dir}/raw/partial/{method}/..._fold-k.json`
  (`results/raw/` is git-ignored). The main JSON stays exactly as in §i.
* **Risk:** if `analysis.load_results` uses a recursive glob (`rglob("*.json")`),
  it would read these files as results. **Proposal:** §j should say that
  `load_results` reads only `{results_dir}/{method}/*.json` for
  `method` in `config.evaluation.methods` (or one directory level and no deeper),
  and should reserve `pareto/` and `raw/`.

## 5. `choose_from_pareto`: tie for `b*` is unspecified (§g)

If several entries share the highest `ba_mean` but have different `ba_std`,
the threshold `b*.ba_mean - b*.ba_std` depends on which one is `b*`.

* **Now:** `b*` is chosen by highest `ba_mean`, then lower `ba_std`, then lower
  `cost`, then smallest `to_str`. This is the same order as the final tie-break,
  and in a tie it gives the stricter threshold.
* **Proposal:** write this order into §g.

## 6. GA operators are not in the contract or config

pymoo operators change the results but not `config_hash`.

* **Now:** `BinaryRandomSampling` (each bit Bernoulli(0.5), then repair, which is
  the same as `random_valid(p=0.5)`), `TwoPointCrossover`, `BitflipMutation`
  with per-gene probability 1/19, `eliminate_duplicates=True`. The
  `chromosome.repair` RNG is seeded from `seed`. The default offspring count is
  `pop_size`, so the run is about 30 + 19 × 30 = 600 evaluations before
  memoisation and about 570 unique evaluations.
* **Proposal:** add `ga.crossover`, `ga.mutation_prob_var` and `ga.sampling_p` to
  `config.yaml` so the hash covers them.

## 7. Meaning of `infer_time_ms` (§i)

"Mean per-trial prediction latency" could mean a batch prediction divided by
n, or a batch of size 1.

* **Now:** a batch of size 1, averaged over up to 50 test trials after warm-up.
  For FBCSP it is `predict` on one `Xsel` trial (selection already done). For
  EEGNet it includes the host-to-device copy and `cuda.synchronize`.
* **Proposal:** state "batch size 1" in §i so FBCSP and EEGNet are comparable.

## 8. `train_time_s` for the GA under parallel folds (§i)

§i says "total wall time across folds, including GA search". Folds run in
parallel, so `train_time_s` is the **sum** of the per-fold wall times (GA plus
final refit), not the elapsed time of the whole run. EEGNet folds run
sequentially, so the two numbers are the same there. The extra key `ga_time_s`
holds the GA part alone.

## 9. EEGNet training details not in config (open item 9)

They are recorded in the `src/eegnet.py` docstring:
* early stopping on the lowest validation cross-entropy, with the best weights restored;
* per-channel z-scoring with statistics from the fit part of the training fold;
* `final_layer_with_constraint=True`, pinned because braindecode 2.0 warns that
  its default will change;
* braindecode 2.0 renamed `EEGNetv4` to `EEGNet`, and the code imports either.

**Proposal:** move these to `config.eegnet` so they change `config_hash`.

## 10. Subject labels on the CLI

The task uses `sub-17..sub-22`. The contract label is `"17"`. The CLIs accept
`17`, `sub-17`, `17..22`, `sub-17..sub-22` and `17-22`, and always write the
2-digit label.
