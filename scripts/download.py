"""Download the Gao et al. Visual Imagery dataset from figshare.

The file list comes from the figshare API (article 30227503, DOI
10.6084/m9.figshare.30227503); subject IDs are never hard-coded. The script:

* downloads the per-subject ``sub-XX.zip`` files into ``<data_dir>/_zips/`` and
  the top-level BIDS files (``*.json``, ``*.tsv``, ``README``) into ``<data_dir>/``;
* resumes interrupted downloads (``.part`` file + HTTP ``Range``), retries on
  network errors and verifies every file's MD5 against the API;
* extracts each verified zip into ``<data_dir>/`` (the BIDS root), refusing any
  member that would land outside it, and records completion in
  ``<data_dir>/_zips/sub-XX.extracted``.

``<data_dir>`` is ``$AA_DATA_DIR`` or, if unset, ``config.yaml: paths.data_dir``
(``data/raw``). Re-running is cheap: verified files are skipped.

Usage (from the repo root)::

    python -m scripts.download --subjects 01          # meta files + sub-01 only
    python -m scripts.download                         # everything (resumes)
    python -m scripts.download --list                  # just print the file list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ARTICLE_ID = 30227503
API = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
SUBJECT_ZIP = re.compile(r"^sub-(\d+)\.zip$")
META = re.compile(r"^(.+\.(json|tsv)|README)$")
CHUNK = 1 << 20
UA = "AA_Project-downloader/0.1 (research use)"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _get_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def list_files() -> list[dict]:
    """All files of the article (``id, name, size, download_url, computed_md5``)."""
    files, page = [], 1
    while True:
        batch = _get_json(f"{API}/files?page={page}&page_size=100")
        if not batch:
            break
        files.extend(batch)
        if len(batch) < 100:
            break
        page += 1
    return files


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_md5(f: dict) -> str | None:
    return f.get("computed_md5") or f.get("supplied_md5") or None


def download(f: dict, dest: Path, retries: int = 20) -> Path:
    """Download one figshare file to ``dest`` with resume + MD5 check. Idempotent."""
    want_md5, size = _expected_md5(f), int(f["size"])
    ok_marker = dest.with_name(dest.name + ".md5ok")
    if dest.exists() and dest.stat().st_size == size and ok_marker.exists():
        return dest
    if dest.exists() and dest.stat().st_size == size:
        if want_md5 is None or md5sum(dest) == want_md5:
            ok_marker.write_text(want_md5 or "")
            return dest
        log(f"{dest.name}: MD5 mismatch, re-downloading")
        dest.unlink()
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        if have == size:
            break
        headers = {"User-Agent": UA}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            req = urllib.request.Request(f["download_url"], headers=headers)
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:  # server ignored Range -> restart
                    have = 0
                mode = "ab" if have else "wb"
                t0, got, last = time.time(), have, time.time()
                with open(part, mode) as out:
                    for chunk in iter(lambda: r.read(CHUNK), b""):
                        out.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 30:
                            rate = (got - have) / max(time.time() - t0, 1e-6) / 1e6
                            log(f"{dest.name}: {got / 1e6:.0f}/{size / 1e6:.0f} MB ({rate:.1f} MB/s)")
                            last = time.time()
            if part.stat().st_size == size:
                break
            log(f"{dest.name}: short read ({part.stat().st_size}/{size}), resuming")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            wait = min(60, 2 ** attempt)
            log(f"{dest.name}: attempt {attempt} failed ({e!r}); retry in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"{dest.name}: gave up after {retries} attempts")
    if want_md5 is not None:
        got_md5 = md5sum(part)
        if got_md5 != want_md5:
            part.unlink()
            raise RuntimeError(f"{dest.name}: MD5 {got_md5} != {want_md5} (deleted, re-run to retry)")
    part.replace(dest)
    ok_marker.write_text(want_md5 or "")
    log(f"{dest.name}: done ({size / 1e6:.0f} MB)")
    return dest


def safe_extract(zpath: Path, root: Path) -> None:
    """Extract ``zpath`` into ``root``; refuse absolute paths and ``..`` traversal."""
    marker = zpath.with_suffix(".extracted")
    if marker.exists():
        return
    root_r = root.resolve()
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = (root_r / m.filename).resolve()
            if not target.is_relative_to(root_r):
                raise RuntimeError(f"{zpath.name}: unsafe member {m.filename!r}")
        z.extractall(root_r)
    marker.write_text("ok")
    log(f"{zpath.name}: extracted into {root_r}")


def default_data_dir() -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import load_config  # noqa: E402
    return Path(load_config()["paths"]["data_dir"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", type=Path, default=None, help="default: $AA_DATA_DIR or config")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="subject labels to fetch (e.g. 01 02); default: all listed by the API")
    p.add_argument("--no-extract", action="store_true")
    p.add_argument("--list", action="store_true", help="print the API file list and exit")
    a = p.parse_args(argv)

    files = list_files()
    subj = {SUBJECT_ZIP.match(f["name"]).group(1).zfill(2): f
            for f in files if SUBJECT_ZIP.match(f["name"])}
    meta = [f for f in files if META.match(f["name"])]
    if a.list:
        for f in files:
            print(f["id"], f["name"], f["size"], _expected_md5(f))
        print(f"{len(subj)} subject zips: {sorted(subj)}")
        return 0

    root = (a.data_dir or default_data_dir()).resolve()
    zdir = root / "_zips"
    log(f"data dir {root}; {len(subj)} subject zips, {len(meta)} meta files")
    for f in meta:
        download(f, root / f["name"])

    wanted = sorted(subj) if a.subjects is None else [s.zfill(2) for s in a.subjects]
    failed = []
    for s in wanted:
        if s not in subj:
            log(f"sub-{s}: not in the article file list")
            failed.append(s)
            continue
        try:
            z = download(subj[s], zdir / subj[s]["name"])
            if not a.no_extract:
                safe_extract(z, root)
        except Exception as e:  # keep going with the other subjects
            log(f"sub-{s}: FAILED {e!r}")
            failed.append(s)
    log(f"finished; failed: {failed or 'none'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
