"""The label taxonomy is load-bearing: if it drifts, every metric shifts."""
from __future__ import annotations

from complaint_triage import config as cfg


def test_every_mapping_target_is_a_declared_label() -> None:
    unknown = set(cfg.PRODUCT_TO_CANONICAL.values()) - set(cfg.CANONICAL_LABELS)
    assert not unknown, f"mapping produces labels missing from CANONICAL_LABELS: {unknown}"


def test_canonical_labels_are_unique_and_sorted_stably() -> None:
    labels = cfg.CANONICAL_LABELS
    assert len(labels) == len(set(labels)), "duplicate entries in CANONICAL_LABELS"


def test_every_declared_label_is_reachable() -> None:
    """A label nothing maps to would silently never be predicted."""
    reachable = set(cfg.PRODUCT_TO_CANONICAL.values())
    orphans = set(cfg.CANONICAL_LABELS) - reachable
    assert not orphans, f"labels no Product maps onto: {orphans}"


def test_historical_credit_reporting_names_collapse_together() -> None:
    """The rename that would otherwise split one class into three."""
    variants = [
        "Credit reporting",
        "Credit reporting, credit repair services, or other personal consumer reports",
        "Credit reporting or other personal consumer reports",
    ]
    mapped = {cfg.PRODUCT_TO_CANONICAL[v] for v in variants}
    assert mapped == {"credit_reporting"}, mapped
