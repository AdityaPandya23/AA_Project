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


def test_subjects_reports_missing_recordings(tmp_path, capsys):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False)
    # sub-11 ses-02 never built; sub-12 not built at all; sub-01 present but not requested
    local = ref[ref["subject"].isin(["01", "11"])]
    local = local[~((local["subject"] == "11") & (local["session"] == "02"))]
    path = _write(local, tmp_path / "m.csv")
    assert cm.main(["--local", str(path), "--reference", str(REF)]) == 0  # old behaviour
    assert cm.main(["--local", str(path), "--reference", str(REF),
                    "--subjects", "11", "12"]) == 1
    out = capsys.readouterr().out
    assert "sub-11 ses-02: missing locally" in out
    assert "sub-12 ses-01: missing locally" in out and "sub-12 ses-02: missing locally" in out
    assert "sub-01" not in out.split("problem(s)")[1]


def test_subjects_ok_and_all(tmp_path):
    ref = pd.read_csv(REF, dtype=str, keep_default_na=False)
    subset = _write(ref[ref["subject"].isin(["09", "10"])], tmp_path / "m.csv")
    # 09 and 10 have ses-01 only, so nothing is missing
    assert cm.main(["--local", str(subset), "--reference", str(REF),
                    "--subjects", "9", "10"]) == 0
    assert cm.main(["--local", str(REF), "--reference", str(REF), "--subjects", "all"]) == 0
    assert cm.main(["--local", str(subset), "--reference", str(REF), "--subjects", "all"]) == 1
    assert cm.main(["--local", str(subset), "--reference", str(REF), "--subjects", "99"]) == 1
