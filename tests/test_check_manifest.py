"""scripts.check_manifest: compare a local manifest with the reference."""
from pathlib import Path

import pandas as pd

from scripts import check_manifest as cm

REF = Path(__file__).resolve().parent.parent / "docs" / "manifest_reference.csv"


def _write(df: pd.DataFrame, path: Path) -> Path:
    df.to_csv(path, index=False)
    return path


def test_reference_matches_itself_and_subsets(tmp_path):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False)
    assert len(ref) == 42
    assert cm.main(["--local", str(REF), "--reference", str(REF)]) == 0
    subset = ref[ref["subject"].isin(["09", "10", "11"])]
    assert cm.main(["--local", str(_write(subset, tmp_path / "m.csv")),
                    "--reference", str(REF)]) == 0


def test_mismatch_detected(tmp_path, capsys):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False)
    bad = ref.head(2).copy()
    bad.loc[bad.index[0], "n_rejected"] = "99"
    bad.loc[bad.index[1], "rejected_trial_ids"] = "1 2 3"
    assert cm.main(["--local", str(_write(bad, tmp_path / "m.csv")),
                    "--reference", str(REF)]) == 1
    out = capsys.readouterr().out
    assert "n_rejected" in out and "rejected_trial_ids" in out


def test_empty_local_fails(tmp_path):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False)
    assert cm.main(["--local", str(_write(ref.head(0), tmp_path / "m.csv")),
                    "--reference", str(REF)]) == 1
