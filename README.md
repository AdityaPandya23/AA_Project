# AA_Project: EEG visual-imagery shape decoding

This project classifies imagined shapes (circle, square, pentagram) from EEG. The
proposed method is a stability-aware NSGA-II that selects channels, bands and
windows, followed by FBCSP and shrinkage LDA. The baselines are FBCSP on all
components and an accuracy-only GA. EEGNet is the deep-learning comparison.

**Result in one line:** on 22 subjects, the stability-aware GA does not beat FBCSP on all
components (pre-registered primary test: Δ −0.007 BA, one-sided p = 0.79). Every method
is within 0.02 of chance on imagery. See [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md).

* **Final report:** [docs/FINAL_REPORT.md](docs/FINAL_REPORT.md). Final tables and figures
  are in [results/final/](results/final/) (`tables/TABLES.md`, `figures/CAPTIONS.md`).
* Pre-registered analysis: [docs/ANALYSIS_PLAN.md](docs/ANALYSIS_PLAN.md) →
  [docs/RESULTS.md](docs/RESULTS.md). Exploratory checks: [docs/P5_REPORT.md](docs/P5_REPORT.md).
* Project summary: [docs/PROJECT.md](docs/PROJECT.md)
* **Interfaces everyone codes against:** [docs/CONTRACTS.md](docs/CONTRACTS.md).
  Change these only by PR.
* All design constants: [config.yaml](config.yaml)

## Setup

Python 3.11+ (R07 used 3.11.4; R01 3.12).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-gpu.txt   # only on the machine running EEGNet
pytest
```

`requirements-lock.txt` is a `pip freeze` of the reference environment.

## Paths

`config.yaml: paths` uses paths relative to the repo root. You can override
them per machine:

| Env var | Default | Content |
|---|---|---|
| `AA_DATA_DIR` | `data/` | BIDS root of the Gao dataset |
| `AA_CACHE_DIR` | `cache/` | preprocessed `.npz` caches (`fbcsp/`, `eegnet/`, `view/`) |
| `AA_RESULTS_DIR` | `results/` | result JSONs `results/{method}/...` |

`data/`, `cache/` and `results/raw/` are git-ignored.

## Building the cache on another machine

There is no shared data folder. Every machine downloads the raw data from
figshare and builds its own cache from the same commit, then checks it against
the reference manifest built on R02 (`docs/manifest_reference.csv`).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m scripts.download --subjects <ids>                 # e.g. 09 10 11 12 13 14 15 16; omit for all
python -m scripts.build_cache --subjects <ids> --n-jobs 3   # 2-3 workers on an 8 GB machine
python -m scripts.check_manifest --subjects <ids>           # must print OK; 'all' for all 22
```

| Machine | Subjects | Raw download | Cache |
|---|---|---|---|
| R02 | all (reference) | ~44 GB | 2.7 GB |
| R01 | 09–16 | ~16 GB | ~0.9 GB |
| R07 | all (EEGNet needs every subject) | ~44 GB | 2.7 GB |

Building takes about 1–2.5 min per recording with 3 workers on R02, and about 10 s per
recording on R07 (measured).

The download is resumable and MD5-checked. The zips in `data/raw/_zips/` are
about a quarter of it (12 GB of 44 GB for all subjects, 4 GB of 16 GB for
09–16), and they can be deleted once `check_manifest` prints OK. The extracted
`data/raw/sub-*` folders are the rest. Keep them only if you may need to
rebuild the cache after a change to `src/io.py`, `src/preprocess.py` or
`config.yaml`.

**Shortcut:** instead of downloading the raw data, unzip a copy of R02's
`cache/` folder (2.7 GB) into the repo root and run
`python -m scripts.check_manifest --subjects all`. Nothing after preprocessing
needs the raw data.

`check_manifest` compares, per local recording, the trials kept per class and
block, the rejected trial ids and the config hash. Without `--subjects` it checks
only the recordings that are present, so a recording that was never built goes
unnoticed. With `--subjects` it also reports missing recordings. If it reports mismatches,
first check that you are on the same commit as the reference and that
`config.yaml` is unchanged.

## Developing without real data

```bash
python -m src.synth --out cache_synth     # fake subjects 91, 92, 93 × sessions 01, 02
AA_CACHE_DIR=cache_synth pytest
```

The synthetic data has the exact cache formats from CONTRACTS.md (a)–(c). The
classes differ slightly in **alpha (8–12 Hz) power on O1 and O2**, and the
viewing cache has a stronger effect than imagery. Subject 93 ses-01 has 3
rejected trials, so code must not assume 120 trials.

## Reproducing everything on one machine

All commands run from the repo root with the venv active. Set a host label first
(`export AA_HOST=R07`, PowerShell: `$env:AA_HOST = "R07"`). It is written into every
result file. Timings were **measured on R07**: i5-13450HX (10 cores / 16 threads),
RTX 3050 6 GB laptop GPU, 16 GB RAM, Windows 11, Python 3.11.4, torch 2.14.1+cu130.

| # | Step | Command | Time on R07 |
|---|---|---|---|
| 1 | Download raw data (all 22 subjects, ~44 GB) | `python -m scripts.download` | network-bound (not timed) |
| 2 | Build caches (42 recordings) | `python -m scripts.build_cache --n-jobs 3` | ~7 min (measured 39 s for 4 recordings) |
| 3 | Check caches against the reference | `python -m scripts.check_manifest --subjects all` | ~1 s, must print OK |
| 4 | Baseline `fbcsp_all` (+ P2 sanity checks) | `python -m experiments.baseline --n-jobs 8` then `python -m experiments.sanity` | 9 s (baseline) |
| 5 | GA, both methods × both protocols | `python -m experiments.run_ga --method {ga_nsga,ga_single} --subjects 01..22 --protocol {block,cross_session} --n-jobs 8` (4 runs) | 2.0 + 0.4 + 2.3 + 0.5 min = **5.3 min** |
| 6 | EEGNet, 32 + 8 channels (GPU) | `python -m experiments.run_eegnet --channels 32 8 --subjects 01..22 --protocol block`, then `--protocol cross_session` | 42 + 7 min = **49 min** |
| 7 | Pre-registered analysis (RESULTS.md) | `python -m reports.p3_numbers <scratch_dir>` | 19 s |
| 8 | P5 exploratory checks | see the reproduce block at the end of [docs/P5_REPORT.md](docs/P5_REPORT.md) | ~2 h GPU + CPU (EEGNet diagnostic); the rest ~20 min |
| 9 | Final tables and figures | `python -m reports.make_final` | 8 s |
| – | Test suite | `pytest` | ~70 s |

Notes:

* Steps 4–5 and 7 are deterministic. On R07 they reproduce the committed files
  bit-for-bit (re-checked for this README: 62 baseline and 372 GA files). EEGNet on
  CUDA is deterministic too (`cudnn.deterministic`).
* **Don't pass `--device` to `run_eegnet` on code older than PR #16** (it changed
  `config_hash`, CONTRACT_ISSUES R07-11). It is fixed now.
* On this laptop GPU, more than about 2 EEGNet processes add no throughput, and about 8
  CUDA processes hit a driver limit. CPU workers for EEGNet need `CUDA_VISIBLE_DEVICES=-1`
  (P5_REPORT §5.2).
* Without the raw data, step 2 can be replaced by unzipping a copy of another machine's
  `cache/` and running step 3.
* Step 5 per machine: the P3 subject split was R02 01–08, R01 09–16, R07 17–22. On one
  machine, use `--subjects 01..22`.

## Where the results are

| Path | Content |
|---|---|
| `results/{fbcsp_all → baseline_all/fbcsp_all, ga_nsga, ga_single, eegnet_32ch, eegnet_8ch}/` | P2–P4 result JSONs (CONTRACTS §i), config `08e124167b` |
| `results/pareto/` | GA Pareto fronts per fold |
| `results/sanity/` | P2 sanity checks (random split, viewing period) |
| `results/p5/` | exploratory P5 outputs (each subfolder has a `summary.*`) |
| `results/final/` | paper tables (`tables/`) and figures with captions (`figures/`) |
| `results/tables/`, `results/figures/` | P2 and P3 working tables and figures |

## Layout

```
src/            config, chromosome, cache, results, synth, io, preprocess, cv,
                fbcsp, ga, eegnet, analysis
experiments/    baseline, sanity, run_ga, run_eegnet (P2-P4); position_control,
                random_control, pareto_thresholds, eegnet_diag (P5)
reports/        make_tables, p3_numbers, train_accuracy, p5_drift,
                make_p5_figures, make_final
scripts/        download, build_cache, check_manifest
tests/          pytest suite (synthetic data; real-data tests skip without data/)
docs/           PROJECT, CONTRACTS, CONTRACT_ISSUES, DATA_NOTES, ANALYSIS_PLAN,
                RESULTS, P2/P3/P5 reports, FINAL_REPORT
```

Run modules as `python -m src.<module>` from the repo root. Never run them as
`python src/<module>.py`, because `src/io.py` would then shadow the standard library `io`.
