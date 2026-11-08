"""Check that this machine's cache matches the reference cache built on R02.

Compares ``<cache_dir>/manifest.csv`` (written by ``scripts.build_cache``) with
``docs/manifest_reference.csv`` for the recordings present locally. For each
recording it checks the status, the trials kept per class and per block,
``n_rejected``, ``n_out_of_range``, the rejected trial ids and the config hash.

Without ``--subjects`` only recordings present locally are compared, so a
recording that was never built is not noticed. With ``--subjects`` every
reference recording of those subjects must also be present locally
(``--subjects all`` = all 42), and local recordings of other subjects are ignored.

Usage (from the repo root)::

    python -m scripts.check_manifest
    python -m scripts.check_manifest --subjects 09 10 11 12 13 14 15 16
    python -m scripts.check_manifest --subjects all
    python -m scripts.check_manifest --local cache/manifest.csv --reference docs/manifest_reference.csv

Prints ``OK`` or one line per mismatch. Exit code 0 = OK, 1 = mismatch, missing
recording or no local recordings.
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


def compare(local: pd.DataFrame, ref: pd.DataFrame,
            subjects: list[str] | None = None) -> list[str]:
    """Return a list of mismatch messages (empty = OK).

    ``subjects``: if given, only these subjects are checked, and every reference
    recording of theirs must be present in ``local``.
    """
    problems: list[str] = []
    if subjects is not None:
        subjects = [s.zfill(2) for s in subjects]
        unknown = sorted(set(subjects) - set(ref.index.get_level_values("subject")))
        problems += [f"sub-{s}: not in the reference manifest" for s in unknown]
        local = local[local.index.get_level_values("subject").isin(subjects)]
        expected = ref[ref.index.get_level_values("subject").isin(subjects)]
        for key in expected.index.difference(local.index):
            problems.append(f"sub-{key[0]} ses-{key[1]}: missing locally (not built?)")
    if local.empty:
        return problems + ["local manifest has no recordings"
                           + (" for the requested subjects" if subjects is not None else "")]
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
    p.add_argument("--subjects", nargs="+", default=None,
                   help="subjects that must all be present (e.g. 09 10 ...), or 'all'")
    a = p.parse_args(argv)
    local_path = Path(a.local) if a.local else Path(load_config()["paths"]["cache_dir"]) / "manifest.csv"
    if not local_path.exists():
        print(f"no local manifest at {local_path}; run python -m scripts.build_cache first")
        return 1
    local, ref = _read(local_path), _read(Path(a.reference))
    subjects = a.subjects
    if subjects is not None and [s.lower() for s in subjects] == ["all"]:
        subjects = sorted(set(ref.index.get_level_values("subject")))
    problems = compare(local, ref, subjects)
    if subjects is not None:
        local = local[local.index.get_level_values("subject").isin([s.zfill(2) for s in subjects])]
    if problems:
        print(f"{len(problems)} problem(s) in {len(local)} checked local recording(s):")
        print("\n".join(problems))
        return 1
    scope = f" ({len(subjects)} subject(s), none missing)" if subjects is not None else ""
    print(f"OK: {len(local)} local recording(s) match the reference{scope}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
