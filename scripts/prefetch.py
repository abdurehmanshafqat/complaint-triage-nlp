#!/usr/bin/env python3
"""Fetch everything the project needs, so the rest can run with no network.

The pipeline is designed to need the network exactly once. After this script
completes, training, distillation, quantisation and the test suite all run
offline — the teacher transformer included, because its logits are cached to
disk rather than recomputed.

Usage:
    python scripts/prefetch.py              # data + teacher weights
    python scripts/prefetch.py --data-only  # skip the transformer
    python scripts/prefetch.py --check      # report what is already cached

Afterwards, to prove nothing reaches for the network:
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from complaint_triage import config as cfg  # noqa: E402
from complaint_triage.distill import DEFAULT_TEACHER  # noqa: E402


def human(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def hf_cache_dir() -> Path:
    override = os.environ.get("HF_HOME")
    base = Path(override) if override else Path.home() / ".cache" / "huggingface"
    return base


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------
def fetch_data() -> bool:
    """Delegate to the resumable downloader."""
    print("\n[1/2] CFPB complaint data")
    import download_data  # noqa: PLC0415

    return download_data.main() == 0


def fetch_teacher(name: str = DEFAULT_TEACHER) -> bool:
    """Pull the teacher's weights and tokeniser into the local HF cache."""
    print(f"\n[2/2] teacher weights: {name}")
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        print("  transformers is not installed yet — install deps first:", file=sys.stderr)
        print("  uv pip install -e '.[teacher]'", file=sys.stderr)
        return False

    try:
        AutoTokenizer.from_pretrained(name)
        AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=len(cfg.CANONICAL_LABELS)
        )
    except Exception as exc:  # noqa: BLE001 - surface any hub/network failure plainly
        print(f"  failed to fetch {name}: {exc}", file=sys.stderr)
        return False

    print(f"  cached under {hf_cache_dir()}")
    return True


def report() -> None:
    """Show what is cached locally and whether an offline run would succeed."""
    print("\nlocal cache status")
    print("-" * 58)

    archive_bytes = cfg.RAW_ARCHIVE.stat().st_size if cfg.RAW_ARCHIVE.exists() else 0
    archive_ok = archive_bytes > 300_000_000
    print(f"  {'OK ' if archive_ok else '-- '} complaint archive   {human(archive_bytes)}")

    hub = hf_cache_dir() / "hub"
    teacher_bytes = dir_size(hub)
    teacher_ok = teacher_bytes > 50_000_000
    print(f"  {'OK ' if teacher_ok else '-- '} teacher weights     {human(teacher_bytes)}")

    processed = dir_size(cfg.PROCESSED_DIR)
    print(f"  {'OK ' if processed else '-- '} processed dataset    {human(processed)}")

    logits = dir_size(cfg.MODEL_DIR / "teacher_logits")
    print(f"  {'OK ' if logits else '-- '} cached teacher logits {human(logits)}")

    maven = dir_size(Path.home() / ".m2")
    print(f"  {'OK ' if maven else '-- '} maven deps (java)    {human(maven)}")
    if not shutil.which("mvn"):
        print("       note: mvn is not on PATH")

    print("-" * 58)
    if archive_ok and teacher_ok:
        print("  Ready for an offline run.")
        print("  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1")
    else:
        print("  Not yet ready offline — run this script with network access.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-only", action="store_true", help="skip the transformer")
    parser.add_argument("--check", action="store_true", help="only report cache status")
    args = parser.parse_args()

    if args.check:
        report()
        return 0

    ok = fetch_data()
    if not args.data_only:
        ok = fetch_teacher() and ok

    report()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
