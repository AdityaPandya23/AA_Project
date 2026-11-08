"""Check that this machine's cache matches the reference cache built on R02.

Compares ``<cache_dir>/manifest.csv`` (written by ``scripts.build_cache``) with
``docs/manifest_reference.csv`` for the recordings present locally. For each
recording it checks the status, the trials kept per class and per block,
``n_rejected``, ``n_out_of_range``, the rejected trial ids and the config hash.

Usage (from the repo root)::

    python -m scripts.check_manifest
    python -m scripts.check_manifest --local cache/manifest.csv --reference docs/manifest_reference.csv

Prints ``OK`` or one line per mismatch. Exit code 0 = OK, 1 = mismatch or no
local recordings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "docs" / "manifest_reference.csv"
COMPARE_COLS = ["status", "n_trials", "n_circle", "n_square", "n_pentagram", "n_rejected",
                "n_out_of_range", "n_block0", "n_block1", "n_block2", "n_block3",
                "rejected_trial_ids", "config_hash"]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["subject"] = df["subject"].str.zfill(2)
    df["session"] = df["session"].str.zfill(2)
    return df.set_index(["subject", "session"])


def compare(local: pd.DataFrame, ref: pd.DataFrame) -> list[str]:
    """Return a list of mismatch messages (empty = OK)."""
    problems: list[str] = []
    if local.empty:
        return ["local manifest has no recordings"]
    for key, row in local.iterrows():
        tag = f"sub-{key[0]} ses-{key[1]}"
        if key not in ref.index:
            problems.append(f"{tag}: not in the reference manifest")
            continue
        r = ref.loc[key]
        for col in COMPARE_COLS:
            a, b = str(row.get(col, "")).strip(), str(r.get(col, "")).strip()
            if a != b:
                problems.append(f"{tag}: {col} local={a!r} reference={b!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--local", default=None, help="default: <cache_dir>/manifest.csv")
    p.add_argument("--reference", default=str(REFERENCE))
    a = p.parse_args(argv)
    local_path = Path(a.local) if a.local else Path(load_config()["paths"]["cache_dir"]) / "manifest.csv"
    if not local_path.exists():
        print(f"no local manifest at {local_path}; run python -m scripts.build_cache first")
        return 1
    local, ref = _read(local_path), _read(Path(a.reference))
    problems = compare(local, ref)
    if problems:
        print(f"{len(problems)} mismatch(es) in {len(local)} local recording(s):")
        print("\n".join(problems))
        return 1
    print(f"OK: {len(local)} local recording(s) match the reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
