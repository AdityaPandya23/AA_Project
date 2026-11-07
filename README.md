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
