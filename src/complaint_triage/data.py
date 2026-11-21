"""Loading, cleaning, deduplicating and splitting the CFPB complaint corpus.

The module is deliberately explicit about two things that quietly inflate
reported accuracy on this dataset:

1. **Duplicate narratives.** A large share of complaints are boilerplate filed
   in bulk (credit-repair mills especially). If the same text lands in both the
   training and test split, the test score measures memorisation.
2. **Random splits.** A model deployed on a routing queue sees *future*
   complaints. Scoring it on a random split overstates what it will do in
   production, because product mix and company behaviour drift over time.

`build_dataset` therefore reports both, and the README publishes both numbers.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)

SplitStrategy = Literal["random", "temporal"]

_REDACTION_RE = re.compile(cfg.REDACTION_PATTERN)
_WHITESPACE_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
@dataclass
class CleaningReport:
    """Row counts at each stage, so the funnel is auditable rather than implied."""

    rows_read: int = 0
    dropped_no_narrative: int = 0
    dropped_too_short: int = 0
    dropped_unmapped_product: int = 0
    dropped_duplicate: int = 0
    rows_kept: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    # Rows that passed cleaning per class, before any per-class cap. The
    # gap against `label_counts` is how much of each class was sampled away.
    rows_available: dict[str, int] = field(default_factory=dict)
    dropped_rare_class: int = 0
    excluded_labels: dict[str, int] = field(default_factory=dict)

    def as_markdown(self) -> str:
        rows = [
            ("read from source", self.rows_read),
            ("dropped: no narrative", -self.dropped_no_narrative),
            (f"dropped: under {cfg.MIN_NARRATIVE_CHARS} chars", -self.dropped_too_short),
            ("dropped: unmapped product", -self.dropped_unmapped_product),
            ("dropped: duplicate narrative", -self.dropped_duplicate),
            (
                f"dropped: class under {cfg.MIN_CLASS_SUPPORT:,} examples",
                -self.dropped_rare_class,
            ),
            ("**kept**", self.rows_kept),
        ]
        out = ["| stage | rows |", "| --- | ---: |"]
        out += [f"| {name} | {count:,} |" for name, count in rows]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def shard_paths(source: Path | None = None) -> list[Path]:
    """Resolve a parquet file, or every shard in a directory, to a sorted list."""
    source = source or cfg.SHARD_DIR
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(
            f"{source} not found. Run `python scripts/download_data.py` first."
        )
    shards = sorted(source.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no .parquet shards in {source}. Run `python scripts/download_data.py`."
        )
    return shards


def iter_raw_chunks(
    source: Path | None = None,
    chunksize: int = 200_000,
    nrows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the corpus in bounded batches.

    Parquet is columnar, so only the eight columns this project uses are read
    off disk -- the snapshot carries eighteen. Combined with row-group batching
    that keeps peak memory flat regardless of how large the corpus grows, which
    matters at 1.7 million narratives.

    Args:
        source: a single .parquet file, or a directory of shards.
        chunksize: rows per yielded batch.
        nrows: stop after this many rows, for fast iteration in notebooks.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415

    yielded = 0
    for shard in shard_paths(source):
        parquet = pq.ParquetFile(shard)
        available = set(parquet.schema_arrow.names)
        missing = [c for c in cfg.USE_COLUMNS if c not in available]
        if missing:
            raise ValueError(
                f"{shard.name} is missing expected columns {missing}; "
                f"it has {sorted(available)}"
            )

        for batch in parquet.iter_batches(
            batch_size=chunksize, columns=list(cfg.USE_COLUMNS)
        ):
            frame = batch.to_pandas()
            if nrows is not None and yielded + len(frame) > nrows:
                frame = frame.iloc[: nrows - yielded]
            yielded += len(frame)
            if not frame.empty:
                yield frame
            if nrows is not None and yielded >= nrows:
                return


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_text(value: str) -> str:
    """Collapse CFPB redaction runs and normalise whitespace.

    The CFPB masks identifying details with runs of `X`. Those runs are
    frequent and carry no class signal, but they do consume tokens, so they are
    collapsed to a single sentinel rather than deleted: *that* a field was
    redacted is weak evidence about the product type, so the sentinel is kept.
    """
    text = _REDACTION_RE.sub(" [redacted] ", value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalise_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Map historical `Product` strings onto the stable routing taxonomy."""
    out = frame.copy()
    out["label"] = out[cfg.COL_PRODUCT].map(cfg.PRODUCT_TO_CANONICAL)
    return out


def clean_frame(frame: pd.DataFrame, report: CleaningReport) -> pd.DataFrame:
    """Apply the full row-level cleaning funnel to one chunk, updating `report`."""
    report.rows_read += len(frame)

    has_narrative = frame[cfg.COL_NARRATIVE].notna()
    report.dropped_no_narrative += int((~has_narrative).sum())
    frame = frame.loc[has_narrative].copy()

    frame["text"] = frame[cfg.COL_NARRATIVE].map(clean_text)

    long_enough = frame["text"].str.len() >= cfg.MIN_NARRATIVE_CHARS
    report.dropped_too_short += int((~long_enough).sum())
    frame = frame.loc[long_enough].copy()

    frame = normalise_labels(frame)
    mapped = frame["label"].notna()
    report.dropped_unmapped_product += int((~mapped).sum())
    if (~mapped).any():
        unknown = sorted(frame.loc[~mapped, cfg.COL_PRODUCT].dropna().unique())
        if unknown:
            log.warning("Unmapped Product values encountered: %s", unknown)
    frame = frame.loc[mapped].copy()

    frame["date_received"] = pd.to_datetime(frame[cfg.COL_DATE], errors="coerce", utc=True)
    frame[cfg.COL_ID] = frame[cfg.COL_ID].astype(str)
    keep = [
        cfg.COL_ID,
        "date_received",
        "label",
        "text",
        cfg.COL_SUBPRODUCT,
        cfg.COL_ISSUE,
        cfg.COL_COMPANY,
        cfg.COL_STATE,
    ]
    return frame[keep].rename(
        columns={
            cfg.COL_ID: "complaint_id",
            cfg.COL_SUBPRODUCT: "sub_product",
            cfg.COL_ISSUE: "issue",
            cfg.COL_COMPANY: "company",
            cfg.COL_STATE: "state",
        }
    )


def drop_duplicate_narratives(
    frame: pd.DataFrame, report: CleaningReport
) -> pd.DataFrame:
    """Remove exact duplicate narratives, keeping the earliest filing.

    Bulk-filed boilerplate is the single largest source of optimistic bias on
    this dataset. Keeping the earliest copy preserves the temporal split's
    meaning (a later duplicate must not become "future" evidence).
    """
    before = len(frame)
    frame = frame.sort_values("date_received").drop_duplicates(
        subset="text", keep="first"
    )
    report.dropped_duplicate += before - len(frame)
    return frame


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def make_splits(
    frame: pd.DataFrame,
    strategy: SplitStrategy = "temporal",
    test_fraction: float = cfg.TEST_FRACTION,
    val_fraction: float = cfg.VAL_FRACTION,
    seed: int = cfg.RANDOM_SEED,
) -> pd.DataFrame:
    """Add a `split` column of train/val/test.

    `temporal` is the honest setting and the project default: validate and test
    on complaints filed after everything the model trained on. `random` is
    provided only so the README can quantify how much a random split flatters
    the same model.
    """
    out = frame.copy()
    if strategy == "random":
        rng = out.sample(frac=1.0, random_state=seed).index
        out = out.loc[rng]
        n = len(out)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        split = ["train"] * n
        split[:n_test] = ["test"] * n_test
        split[n_test : n_test + n_val] = ["val"] * n_val
        out["split"] = split
        return out.sort_index()

    if strategy == "temporal":
        out = out.sort_values("date_received")
        n = len(out)
        n_test = int(n * test_fraction)
        n_val = int(n * val_fraction)
        n_train = n - n_val - n_test
        out["split"] = (
            ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        )
        return out

    raise ValueError(f"Unknown split strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def cap_per_class(
    frame: pd.DataFrame, cap: int, seed: int = cfg.RANDOM_SEED
) -> pd.DataFrame:
    """Take at most `cap` rows per label, leaving smaller classes untouched.

    Written as an explicit loop rather than `groupby().sample(n=...)` or
    `groupby().apply(...)` on purpose. `groupby().sample(n=cap)` raises as soon
    as any group holds fewer than `cap` rows -- which is every minority route
    here -- and `apply(..., include_groups=True)` was removed in pandas 3. The
    loop is longer but behaves identically across versions and states its
    intent plainly.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive, got {cap}")

    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("label", sort=False):
        take = min(len(group), cap)
        parts.append(
            group if take == len(group) else group.sample(n=take, random_state=seed)
        )
    return pd.concat(parts, ignore_index=True)


def build_dataset(
    nrows: int | None = None,
    sample_per_class: int | None = None,
    strategy: SplitStrategy = "temporal",
    deduplicate: bool = True,
    min_class_support: int = cfg.MIN_CLASS_SUPPORT,
    out_path: Path | None = None,
    source: Path | None = None,
    chunksize: int = 200_000,
    seed: int = cfg.RANDOM_SEED,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Read, clean, deduplicate, subsample and split the corpus in one pass.

    The uncompressed CSV is ~5.9 GB and holds roughly five million published
    narratives. Reading it, cleaning it, and only then capping per class would
    mean holding several gigabytes of Python strings at once, so the cap is
    applied *while streaming* instead:

    * **Deduplication** uses a set of narrative hashes, so a repeated text is
      rejected the moment it reappears rather than after everything is loaded.
      Which copy survives is not important; that no narrative appears on both
      sides of a split is.
    * **Subsampling** uses per-class reservoir sampling (Algorithm R). Each
      class keeps a uniform random sample of every row it ever saw, with memory
      bounded by `sample_per_class`, in a single pass and without knowing the
      class frequencies up front. Taking the first N per class instead would be
      cheaper and wrong: the export is roughly chronological, so it would hand
      back only the oldest complaints and leave the temporal split with no
      recent data to test on.

    Args:
        nrows: cap on raw rows read, for fast iteration in notebooks.
        sample_per_class: reservoir size per label. `None` keeps everything,
            which needs a lot of memory on the full corpus.
        strategy: "temporal" (default) or "random".
        deduplicate: reject repeated narratives.
        min_class_support: drop labels with fewer examples than this. Set to 0
            to keep every label, including ones too rare to learn.
        out_path: parquet destination; defaults to data/processed.
        source: parquet file or shard directory; defaults to data/raw/parquet.
        chunksize: rows per read chunk.
        seed: controls the reservoir sampling.
    """
    import random

    rng = random.Random(seed)
    report = CleaningReport()

    seen_hashes: set[int] = set()
    reservoirs: dict[str, list[tuple]] = {}
    counts_seen: dict[str, int] = {}
    columns: list[str] | None = None

    for chunk in iter_raw_chunks(source=source, chunksize=chunksize, nrows=nrows):
        cleaned = clean_frame(chunk, report)
        if cleaned.empty:
            continue
        if columns is None:
            columns = list(cleaned.columns)

        if deduplicate:
            hashes = cleaned["text"].map(hash)
            fresh = ~hashes.isin(seen_hashes) & ~hashes.duplicated(keep="first")
            report.dropped_duplicate += int((~fresh).sum())
            cleaned = cleaned.loc[fresh]
            seen_hashes.update(hashes.loc[fresh].tolist())
            if cleaned.empty:
                continue

        for row in cleaned.itertuples(index=False, name=None):
            label = row[columns.index("label")]
            position = counts_seen.get(label, 0)
            counts_seen[label] = position + 1

            if sample_per_class is None:
                reservoirs.setdefault(label, []).append(row)
                continue

            bucket = reservoirs.setdefault(label, [])
            if len(bucket) < sample_per_class:
                bucket.append(row)
            else:
                # Algorithm R: the (position+1)-th item replaces a uniformly
                # chosen incumbent with probability k/(position+1).
                candidate = rng.randrange(position + 1)
                if candidate < sample_per_class:
                    bucket[candidate] = row

    if not reservoirs:
        raise ValueError("no rows survived cleaning; check the source archive")

    if min_class_support > 0:
        rare = {
            label: count
            for label, count in counts_seen.items()
            if count < min_class_support
        }
        if rare:
            report.excluded_labels = rare
            report.dropped_rare_class = sum(
                len(reservoirs.get(label, [])) for label in rare
            )
            for label in rare:
                reservoirs.pop(label, None)
            log.warning(
                "Excluded %d label(s) under the %s-example floor: %s",
                len(rare),
                f"{min_class_support:,}",
                rare,
            )

    if not reservoirs:
        raise ValueError("every class fell below min_class_support")

    frame = pd.DataFrame(
        [row for bucket in reservoirs.values() for row in bucket], columns=columns
    )
    frame = make_splits(frame, strategy=strategy, seed=seed)
    report.rows_kept = len(frame)
    report.label_counts = frame["label"].value_counts().to_dict()
    report.rows_available = dict(counts_seen)

    out_path = out_path or (cfg.PROCESSED_DIR / f"complaints_{strategy}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    log.info("Wrote %s rows to %s", f"{len(frame):,}", out_path)
    return frame, report


def load_processed(
    strategy: SplitStrategy = "temporal", path: Path | None = None
) -> pd.DataFrame:
    """Read back a previously built dataset."""
    path = path or (cfg.PROCESSED_DIR / f"complaints_{strategy}.parquet")
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it with "
            f"`complaint_triage.data.build_dataset(strategy={strategy!r})`."
        )
    return pd.read_parquet(path)
