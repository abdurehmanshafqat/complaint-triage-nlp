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
import zipfile
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

    def as_markdown(self) -> str:
        rows = [
            ("read from source", self.rows_read),
            ("dropped: no narrative", -self.dropped_no_narrative),
            (f"dropped: under {cfg.MIN_NARRATIVE_CHARS} chars", -self.dropped_too_short),
            ("dropped: unmapped product", -self.dropped_unmapped_product),
            ("dropped: duplicate narrative", -self.dropped_duplicate),
            ("**kept**", self.rows_kept),
        ]
        out = ["| stage | rows |", "| --- | ---: |"]
        out += [f"| {name} | {count:,} |" for name, count in rows]
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def iter_raw_chunks(
    archive: Path | None = None,
    chunksize: int = 200_000,
    nrows: int | None = None,
) -> Iterator[pd.DataFrame]:
    """Stream the bulk CSV straight out of the zip.

    The uncompressed CSV is several GB; streaming keeps peak memory bounded and
    avoids leaving an extracted copy on disk.
    """
    archive = archive or cfg.RAW_ARCHIVE
    if not archive.exists():
        raise FileNotFoundError(
            f"{archive} not found. Run `python scripts/download_data.py` first."
        )

    with zipfile.ZipFile(archive) as zf:
        inner = _resolve_inner_csv(zf)
        with zf.open(inner) as handle:
            reader = pd.read_csv(
                handle,
                usecols=list(cfg.USE_COLUMNS),
                dtype=str,
                chunksize=chunksize,
                nrows=nrows,
                on_bad_lines="warn",
                low_memory=False,
            )
            for chunk in reader:
                yield chunk


def _resolve_inner_csv(zf: zipfile.ZipFile) -> str:
    names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not names:
        raise ValueError(f"No CSV inside archive; contents: {zf.namelist()}")
    if cfg.RAW_CSV_NAME in names:
        return cfg.RAW_CSV_NAME
    return names[0]


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
    out_path: Path | None = None,
    archive: Path | None = None,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Read, clean, optionally balance, split and persist the corpus.

    Args:
        nrows: cap on raw rows read, for fast iteration in notebooks.
        sample_per_class: cap per label after cleaning. The raw corpus is
            heavily skewed toward credit reporting; capping keeps training
            tractable on a laptop GPU without discarding the minority classes.
        strategy: "temporal" (default) or "random".
        deduplicate: drop exact duplicate narratives.
        out_path: parquet destination; defaults to data/processed.
        archive: source zip; defaults to the configured raw archive.
    """
    report = CleaningReport()
    cleaned: list[pd.DataFrame] = []
    for chunk in iter_raw_chunks(archive=archive, nrows=nrows):
        cleaned.append(clean_frame(chunk, report))

    frame = pd.concat(cleaned, ignore_index=True)
    if deduplicate:
        frame = drop_duplicate_narratives(frame, report)

    if sample_per_class is not None:
        frame = cap_per_class(frame, sample_per_class)

    frame = make_splits(frame, strategy=strategy)
    report.rows_kept = len(frame)
    report.label_counts = frame["label"].value_counts().to_dict()

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
