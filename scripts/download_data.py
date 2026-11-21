#!/usr/bin/env python3
"""Fetch the CFPB Consumer Complaint Database bulk export.

The archive is ~353 MB and is deliberately *not* committed: see the data
provenance section of the README. Re-running is safe — a complete file is left
alone, and a partial file is resumed.

Usage:
    python scripts/download_data.py            # fetch if missing/incomplete
    python scripts/download_data.py --force    # re-fetch from scratch
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complaint_triage import config as cfg  # noqa: E402

CHUNK = 1 << 20  # 1 MiB


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length else None


def is_complete(path: Path, expected: int | None) -> bool:
    if not path.exists():
        return False
    if expected is not None and path.stat().st_size != expected:
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except zipfile.BadZipFile:
        return False


def download(url: str, dest: Path, expected: int | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    start = dest.stat().st_size if dest.exists() else 0
    if expected is not None and start >= expected:
        start = 0  # oversized leftover; start clean

    headers = {"Range": f"bytes={start}-"} if start else {}
    request = urllib.request.Request(url, headers=headers)
    mode = "ab" if start else "wb"

    with urllib.request.urlopen(request, timeout=120) as response, dest.open(mode) as fh:
        done = start
        while True:
            block = response.read(CHUNK)
            if not block:
                break
            fh.write(block)
            done += len(block)
            if expected:
                pct = 100 * done / expected
                print(f"\r  {done / 1e6:7.1f} / {expected / 1e6:.1f} MB ({pct:5.1f}%)",
                      end="", flush=True)
            else:
                print(f"\r  {done / 1e6:7.1f} MB", end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="ignore any existing file")
    args = parser.parse_args()

    dest = cfg.RAW_ARCHIVE
    print(f"source: {cfg.DATA_URL}")
    print(f"target: {dest}")

    expected = remote_size(cfg.DATA_URL)
    if expected:
        print(f"remote size: {expected / 1e6:.1f} MB")

    if args.force and dest.exists():
        dest.unlink()

    if is_complete(dest, expected):
        print("Already present and valid — nothing to do.")
        return 0

    download(cfg.DATA_URL, dest, expected)

    if not is_complete(dest, expected):
        print("ERROR: archive is incomplete or corrupt after download.", file=sys.stderr)
        print("Re-run with --force to start over.", file=sys.stderr)
        return 1

    with zipfile.ZipFile(dest) as zf:
        inner = [i for i in zf.infolist() if i.filename.lower().endswith(".csv")]
        for info in inner:
            print(f"  contains {info.filename}: {info.file_size / 1e9:.2f} GB uncompressed")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
