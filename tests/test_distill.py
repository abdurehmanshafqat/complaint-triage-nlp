"""Distillation plumbing: the alignment and shape checks that prevent silent bugs."""
from __future__ import annotations

import numpy as np
import pytest

from complaint_triage import distill


def test_config_rejects_impossible_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        distill.DistillConfig(alpha=1.5)


def test_config_rejects_nonpositive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        distill.DistillConfig(temperature=0.0)


def test_pack_targets_concatenates_hard_and_soft() -> None:
    y = np.array([0, 2])
    logits = np.array([[1.0, 0.0, -1.0], [0.1, 0.2, 3.0]])
    packed = distill.pack_targets(y, logits, n_classes=3)
    assert packed.shape == (2, 6)
    np.testing.assert_allclose(packed[0, :3], [1, 0, 0])
    np.testing.assert_allclose(packed[1, :3], [0, 0, 1])
    np.testing.assert_allclose(packed[:, 3:], logits, rtol=1e-6)


def test_pack_targets_catches_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        distill.pack_targets(np.array([0, 1]), np.zeros((3, 3)), n_classes=3)


def test_pack_targets_catches_class_count_mismatch() -> None:
    with pytest.raises(ValueError, match="expected"):
        distill.pack_targets(np.array([0]), np.zeros((1, 5)), n_classes=3)


def test_missing_teacher_logits_raise_a_clear_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(distill.cfg, "MODEL_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="train_teacher"):
        distill.load_teacher_logits("train", np.array(["1"]))
