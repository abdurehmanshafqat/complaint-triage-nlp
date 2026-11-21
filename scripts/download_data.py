#!/usr/bin/env python3
"""Fetch the complaint corpus: three parquet shards, ~1.8 GB in total.

**Why not the CFPB bulk export?** Because it no longer carries the text. Both
`files.consumerfinance.gov/ccdb/complaints.csv.zip` and the search API return
15 columns with no `Consumer complaint narrative` field. This project trains on
a CC0 snapshot taken while the narratives were still published; see the
provenance note in `complaint_triage.config`.

**Why no mid-file resume?** An earlier version of this script resumed a partial
download by appending with a Range request. The source regenerated the file
between the two transfers, so the result had exactly the expected length and a
spliced, unreadable compression stream -- the byte count agreed and the data
was garbage. Shards here are downloaded to a `.part` file, validated by
actually opening them, and only then moved into place. An interrupted shard is
simply re-fetched; at ~600 MB each that is cheaper than being subtly wrong.

Usage:
    python scripts/download_data.py            # fetch any missing shard
    python scripts/download_data.py --force    # re-fetch everything
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complaint_triage import config as cfg  # noqa: E402

CHUNK = 1 << 20
USER_AGENT = "complaint-triage/0.1"


def is_valid_parquet(path: Path) -> tuple[bool, int]:
    """Open the file and read its footer. Returns (ok, row_count).

    A length check is not enough -- that is exactly what let a corrupt download
    through before. Reading the metadata forces the footer and schema to parse.
    """
    if not path.exists() or path.stat().st_size < 1024:
        return False, 0
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        missing = [c for c in cfg.USE_COLUMNS if c not in set(parquet.schema_arrow.names)]
        if missing:
            print(f"  {path.name}: missing columns {missing}", file=sys.stderr)
            return False, rows
        return rows > 0, rows
    except Exception as exc:  # noqa: BLE001 - any parse failure means "re-fetch"
        print(f"  {path.name}: unreadable ({type(exc).__name__}: {exc})", file=sys.stderr)
        return False, 0


def fetch(url: str, dest: Path) -> None:
    """Download to `dest` via a .part file, so a crash cannot leave a half file."""
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=180) as response:
        total_header = response.headers.get("Content-Length")
        total = int(total_header) if total_header else None
        done = 0
        last = -10.0
        with part.open("wb") as fh:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                fh.write(block)
                done += len(block)
                if total:
                    pct = 100 * done / total
                    if pct - last >= 5.0:
                        print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)")
                        last = pct
                elif done - last >= 100e6:
                    print(f"    {done / 1e6:7.1f} MB")
                    last = done

    shutil.move(str(part), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-fetch every shard")
    args = parser.parse_args()

    cfg.SHARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"dataset: {cfg.DATASET_REPO} [{cfg.DATASET_CONFIG}], {cfg.DATASET_LICENSE}")
    print(f"target : {cfg.SHARD_DIR}")

    total_rows = 0
    failures = 0

    for index, url in enumerate(cfg.DATA_SHARD_URLS):
        dest = cfg.SHARD_DIR / f"{index}.parquet"
        print(f"\n[{index + 1}/{len(cfg.DATA_SHARD_URLS)}] {dest.name}")

        if args.force:
            dest.unlink(missing_ok=True)

        ok, rows = is_valid_parquet(dest)
        if ok:
            print(f"  present and valid ({rows:,} rows) - skipping")
            total_rows += rows
            continue

        try:
            fetch(url, dest)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  ERROR fetching shard {index}: {exc}", file=sys.stderr)
            failures += 1
            continue

        ok, rows = is_valid_parquet(dest)
        if not ok:
            print(f"  ERROR: shard {index} failed validation after download", file=sys.stderr)
            failures += 1
            continue
        print(f"  ok ({rows:,} rows, {dest.stat().st_size / 1e6:.1f} MB)")
        total_rows += rows

    print(f"\n{total_rows:,} rows available across {len(cfg.DATA_SHARD_URLS)} shards")
    if failures:
        print(f"{failures} shard(s) failed; re-run to retry.", file=sys.stderr)
        return 1
    if total_rows and abs(total_rows - cfg.DATASET_ROWS) > 1000:
        print(
            f"NOTE: expected about {cfg.DATASET_ROWS:,} rows; the snapshot may "
            f"have been updated upstream.",
            file=sys.stderr,
        )
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
