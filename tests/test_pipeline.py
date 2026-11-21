"""End-to-end pipeline test over a synthetic parquet shard.

Runs the real `build_dataset` against a hand-built file, so the batched reader,
the cleaning funnel, deduplication, reservoir sampling and the split logic are
all exercised together without needing the 1.8 GB download.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from complaint_triage import config as cfg
from complaint_triage import data

ROWS = [
    # id, date, product, sub-product, issue, narrative, company, state
    ("1", "2018-01-05", "Mortgage", "FHA", "Trouble paying", "I cannot reach my servicer about escrow at all", "ACME", "CA"),
    ("2", "2018-06-11", "Credit reporting", "", "Incorrect info", "There is an account on my report that is not mine whatsoever", "BUREAU", "TX"),
    # Same narrative as 2, filed later: the duplicate that must be dropped.
    ("3", "2020-02-02", "Credit reporting, credit repair services, or other personal consumer reports", "", "Incorrect info", "There is an account on my report that is not mine whatsoever", "BUREAU", "FL"),
    ("4", "2021-03-03", "Student loan", "Federal", "Repayment", "My income driven repayment plan was recalculated without notice", "LOANCO", "NY"),
    ("5", "2022-04-04", "Checking or savings account", "Checking", "Fees", "They charged me overdraft fees three times in one single day", "BANK", "WA"),
    # No narrative: dropped.
    ("6", "2022-05-05", "Mortgage", "VA", "Closing", "", "ACME", "OR"),
    # Narrative too short: dropped.
    ("7", "2022-06-06", "Mortgage", "VA", "Closing", "too short", "ACME", "OR"),
    # Unmappable product: dropped.
    ("8", "2022-07-07", "Interplanetary Banking", "", "Other", "Something went wrong with my orbital transfer account", "SPACE", "NV"),
    ("9", "2023-08-08", "Debt collection", "Medical", "Contact", "A collector keeps calling my workplace after I told them to stop", "COLLECT", "AZ"),
]


@pytest.fixture
def shard(tmp_path: Path) -> Path:
    """Build a parquet file with the same column names as the real snapshot."""
    frame = pd.DataFrame(
        ROWS,
        columns=[
            cfg.COL_ID,
            cfg.COL_DATE,
            cfg.COL_PRODUCT,
            cfg.COL_SUBPRODUCT,
            cfg.COL_ISSUE,
            cfg.COL_NARRATIVE,
            cfg.COL_COMPANY,
            cfg.COL_STATE,
        ],
    )
    # The real file stores an absent narrative as null, not as an empty string.
    frame[cfg.COL_NARRATIVE] = frame[cfg.COL_NARRATIVE].replace("", None)
    path = tmp_path / "0.parquet"
    frame.to_parquet(path, index=False)
    return path


def test_reader_streams_in_batches(shard: Path) -> None:
    chunks = list(data.iter_raw_chunks(source=shard, chunksize=4))
    assert sum(len(c) for c in chunks) == len(ROWS)
    assert len(chunks) > 1, "batch_size was ignored; large files would load whole"


def test_cleaning_funnel_accounts_for_every_dropped_row(shard: Path) -> None:
    report = data.CleaningReport()
    frames = [
        data.clean_frame(chunk, report)
        for chunk in data.iter_raw_chunks(source=shard, chunksize=100)
    ]
    frame = pd.concat(frames, ignore_index=True)

    assert report.rows_read == len(ROWS)
    assert report.dropped_no_narrative == 1
    assert report.dropped_too_short == 1
    assert report.dropped_unmapped_product == 1
    assert len(frame) == len(ROWS) - 3

    accounted = (
        len(frame)
        + report.dropped_no_narrative
        + report.dropped_too_short
        + report.dropped_unmapped_product
    )
    assert accounted == report.rows_read, "funnel does not balance"


def test_build_dataset_dedupes_splits_and_persists(shard: Path, tmp_path: Path) -> None:
    out = tmp_path / "out.parquet"
    frame, report = data.build_dataset(
        strategy="temporal", source=shard, out_path=out
    )

    assert out.exists(), "dataset was not persisted"
    assert report.dropped_duplicate == 1, "the repeated boilerplate was not removed"
    assert frame["text"].is_unique

    # The renamed credit-reporting products must have collapsed to one label.
    assert frame.loc[frame["complaint_id"] == "2", "label"].item() == "credit_reporting"
    assert set(frame["label"]) <= set(cfg.CANONICAL_LABELS)
    assert set(frame["split"]) <= {"train", "val", "test"}

    reloaded = pd.read_parquet(out)
    assert len(reloaded) == len(frame)


def test_per_class_cap_leaves_small_classes_intact() -> None:
    frame = pd.DataFrame(
        {"label": ["mortgage"] * 5 + ["card"] * 2, "text": [f"t{i}" for i in range(7)]}
    )
    capped = data.cap_per_class(frame, cap=3)
    counts = capped["label"].value_counts().to_dict()
    assert counts["mortgage"] == 3
    assert counts["card"] == 2


def test_per_class_cap_rejects_nonsense_cap() -> None:
    frame = pd.DataFrame({"label": ["a"], "text": ["x"]})
    with pytest.raises(ValueError, match="cap must be positive"):
        data.cap_per_class(frame, cap=0)
