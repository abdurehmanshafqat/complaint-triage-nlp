"""Cleaning and splitting behaviour — the parts that cause silent leakage."""
from __future__ import annotations

import pandas as pd
import pytest

from complaint_triage import config as cfg
from complaint_triage import data


def test_clean_text_collapses_redaction_runs() -> None:
    """CFPB masks identifying details with runs of X; each run becomes one token."""
    assert data.clean_text('paid XXXXXXXX on XX/XX/2020') == 'paid [redacted] on [redacted] / [redacted] /2020'


def test_clean_text_leaves_single_x_alone() -> None:
    """A lone 'X' is ordinary text (e.g. 'Series X'), not a redaction marker."""
    assert "redacted" not in data.clean_text("the Series X card")


def test_clean_text_normalises_whitespace() -> None:
    assert data.clean_text("too   many\n\nspaces\there") == "too many spaces here"


def test_normalise_labels_maps_known_products() -> None:
    frame = pd.DataFrame({cfg.COL_PRODUCT: ["Mortgage", "Student loan", "Nonsense"]})
    out = data.normalise_labels(frame)
    assert out["label"].tolist()[:2] == ["mortgage", "student_loan"]
    assert pd.isna(out["label"].iloc[2]), "unknown products must be NaN, not guessed"


def _dup_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": ["1", "2", "3"],
            "text": ["same boilerplate", "same boilerplate", "unique"],
            "label": ["credit_reporting"] * 3,
            "date_received": pd.to_datetime(
                ["2020-01-02", "2019-05-01", "2021-03-03"], utc=True
            ),
        }
    )


def test_duplicates_dropped_keeping_earliest() -> None:
    report = data.CleaningReport()
    out = data.drop_duplicate_narratives(_dup_frame(), report)
    assert len(out) == 2
    assert report.dropped_duplicate == 1
    kept = out.loc[out["text"] == "same boilerplate", "complaint_id"].item()
    assert kept == "2", "the earliest filing must survive, to keep temporal splits honest"


def _split_frame(n: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": [str(i) for i in range(n)],
            "text": [f"complaint number {i}" for i in range(n)],
            "label": ["mortgage" if i % 2 else "card" for i in range(n)],
            "date_received": pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC"),
        }
    )


def test_temporal_split_puts_the_future_in_test() -> None:
    out = data.make_splits(_split_frame(), strategy="temporal")
    latest_train = out.loc[out["split"] == "train", "date_received"].max()
    earliest_test = out.loc[out["split"] == "test", "date_received"].min()
    assert latest_train < earliest_test, "temporal split leaked future rows into train"


def test_splits_partition_the_frame_exactly_once() -> None:
    for strategy in ("temporal", "random"):
        out = data.make_splits(_split_frame(), strategy=strategy)
        assert len(out) == 100
        assert set(out["split"]) <= {"train", "val", "test"}
        assert out["complaint_id"].nunique() == 100


def test_unknown_split_strategy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown split strategy"):
        data.make_splits(_split_frame(), strategy="stratified")  # type: ignore[arg-type]


def test_cleaning_report_renders_a_funnel() -> None:
    report = data.CleaningReport(rows_read=10, dropped_no_narrative=4, rows_kept=6)
    table = report.as_markdown()
    assert "read from source" in table and "10" in table
