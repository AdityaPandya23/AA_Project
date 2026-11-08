# AA_Project: EEG visual-imagery shape decoding

This project classifies imagined shapes (circle, square, pentagram) from EEG. The
proposed method is a stability-aware NSGA-II that selects channels, bands and
windows, followed by FBCSP and shrinkage LDA. The baselines are FBCSP on all
components and an accuracy-only GA. EEGNet is the deep-learning comparison.

* Project summary: [docs/PROJECT.md](docs/PROJECT.md)
* **Interfaces everyone codes against:** [docs/CONTRACTS.md](docs/CONTRACTS.md).
  Change these only by PR.
* All design constants: [config.yaml](config.yaml)

## Setup

Python 3.11+ (the venv was created with 3.12).

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
python -m scripts.check_manifest                            # must print OK
```

| Machine | Subjects | Raw download | Cache |
|---|---|---|---|
| R02 | all (reference) | ~44 GB | 2.7 GB |
| R01 | 09–16 | ~16 GB | ~1 GB |
| R07 | all (EEGNet needs every subject) | ~44 GB | 2.7 GB |

The download is resumable and MD5-checked. About 12 GB of it is the zips in
`data/raw/_zips/`, which can be deleted once `check_manifest` prints OK. Keep
`data/raw/sub-*` only if you may need to rebuild the cache after a change to
`src/io.py`, `src/preprocess.py` or `config.yaml`.

`check_manifest` compares, per local recording, the trials kept per class and
block, the rejected trial ids and the config hash. If it reports mismatches,
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

## Layout

```
src/
  config.py       load_config, config_hash           (implemented)
  chromosome.py   19-bit genome                      (implemented)
  cache.py        .npz cache read/write/validate     (implemented)
  results.py      results JSON schema/writer         (implemented)
  synth.py        fake caches                        (implemented)
  io.py           BIDS reading                       (stub)
  preprocess.py   raw -> caches                      (stub)
  cv.py           splits                             (stub)
  fbcsp.py        FBCSPLDA, evaluate                 (stub)
  ga.py           NSGA-II / single-objective GA      (stub)
  eegnet.py       EEGNet evaluation                  (stub)
  analysis.py     results -> stats, plots            (stub)
tests/
scripts/          entry-point scripts go here
docs/
```

Run modules as `python -m src.<module>` from the repo root. Never run them as
`python src/<module>.py`, because `src/io.py` would then shadow the standard library `io`.
