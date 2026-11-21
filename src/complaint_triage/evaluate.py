"""Shared evaluation: metrics, confusion matrices and a results ledger.

Every model in the project reports the same numbers through this module, so the
README's comparison table is apples-to-apples rather than assembled by hand.

Macro-F1 is the headline metric, not accuracy. The corpus is dominated by
credit-reporting complaints; a model that predicts that class alone scores well
on accuracy while being useless as a router.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from . import config as cfg


@dataclass
class ModelResult:
    """One row of the project's comparison table."""

    name: str
    family: str
    accuracy: float
    macro_f1: float
    weighted_f1: float
    n_params: int | None = None
    size_bytes: int | None = None
    latency_ms_per_item: float | None = None
    split_strategy: str = "temporal"
    notes: str = ""
    per_class_f1: dict[str, float] = field(default_factory=dict)

    @property
    def size_mb(self) -> float | None:
        return None if self.size_bytes is None else self.size_bytes / 1e6


def labels_in(frame, column: str = "label") -> list[str]:
    """Sorted labels actually present in a dataset.

    Metrics are computed against this rather than `CANONICAL_LABELS`, because
    the support floor can exclude a label. Scoring against a label with no test
    examples silently adds a zero to every macro average.
    """
    return sorted(frame[column].dropna().unique().tolist())


def score(
    name: str,
    family: str,
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Sequence[str] = cfg.CANONICAL_LABELS,
    **extra: object,
) -> ModelResult:
    """Compute the standard metric set for one model's predictions."""
    per_class = f1_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)
    return ModelResult(
        name=name,
        family=family,
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        per_class_f1={lab: float(v) for lab, v in zip(labels, per_class, strict=False)},
        **extra,  # type: ignore[arg-type]
    )


def report_text(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Sequence[str] = cfg.CANONICAL_LABELS,
) -> str:
    return classification_report(
        y_true, y_pred, labels=list(labels), zero_division=0, digits=3
    )


def confusion_frame(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    labels: Sequence[str] = cfg.CANONICAL_LABELS,
    normalise: bool = True,
) -> pd.DataFrame:
    """Confusion matrix as a labelled frame, row-normalised by default."""
    matrix = confusion_matrix(y_true, y_pred, labels=list(labels))
    if normalise:
        totals = matrix.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            matrix = np.divide(matrix, totals, where=totals != 0)
    return pd.DataFrame(matrix, index=list(labels), columns=list(labels))


def benchmark_latency(
    predict_fn,
    samples: Sequence[str],
    repeats: int = 3,
    warmup: int = 1,
) -> float:
    """Median wall-clock milliseconds per item for a single-item call path.

    Measured one item at a time on purpose: a routing queue scores complaints as
    they arrive, so per-item latency is the number that matters, not the
    throughput you would get from a large batch.
    """
    for _ in range(warmup):
        predict_fn(samples[0])

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for text in samples:
            predict_fn(text)
        elapsed = time.perf_counter() - start
        timings.append(elapsed / len(samples) * 1000)
    return float(np.median(timings))


class ResultsLedger:
    """Collects `ModelResult`s and renders the README's comparison table."""

    def __init__(self) -> None:
        self._rows: list[ModelResult] = []

    def add(self, result: ModelResult) -> ModelResult:
        self._rows = [r for r in self._rows if r.name != result.name]
        self._rows.append(result)
        return result

    def to_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "model": r.name,
                    "family": r.family,
                    "macro_f1": round(r.macro_f1, 4),
                    "accuracy": round(r.accuracy, 4),
                    "params": r.n_params,
                    "size_mb": None if r.size_mb is None else round(r.size_mb, 2),
                    "ms_per_item": None
                    if r.latency_ms_per_item is None
                    else round(r.latency_ms_per_item, 2),
                    "split": r.split_strategy,
                    "notes": r.notes,
                }
                for r in self._rows
            ]
        )
        if frame.empty:
            return frame
        return frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)

    def to_markdown(self) -> str:
        """Render the table without pulling in `tabulate`.

        `DataFrame.to_markdown` needs an optional dependency purely to format a
        handful of rows for the README; this does the same job in the stdlib.
        """
        frame = self.to_frame()
        if frame.empty:
            return "_no results yet_"

        headers = list(frame.columns)
        rows = [
            ["" if value is None or value != value else str(value) for value in record]
            for record in frame.to_numpy().tolist()
        ]
        widths = [
            max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
            for i in range(len(headers))
        ]
        numeric = {"macro_f1", "accuracy", "params", "size_mb", "ms_per_item"}

        def render(cells: list[str]) -> str:
            padded = [
                cells[i].rjust(widths[i])
                if headers[i] in numeric
                else cells[i].ljust(widths[i])
                for i in range(len(cells))
            ]
            return "| " + " | ".join(padded) + " |"

        rule = "| " + " | ".join(
            ("-" * (widths[i] - 1) + ":") if headers[i] in numeric else "-" * widths[i]
            for i in range(len(headers))
        ) + " |"

        return "\n".join([render(headers), rule, *(render(row) for row in rows)])

    def save(self, path: Path | None = None) -> Path:
        path = path or (cfg.REPORT_DIR / "results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in self._rows], indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "ResultsLedger":
        path = path or (cfg.REPORT_DIR / "results.json")
        ledger = cls()
        if path.exists():
            for raw in json.loads(path.read_text()):
                ledger.add(ModelResult(**raw))
        return ledger
