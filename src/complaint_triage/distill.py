"""Knowledge distillation from a transformer teacher into a small student.

Design decision: **offline distillation.** The teacher is fine-tuned once with
PyTorch + Hugging Face, its logits over the whole corpus are cached to disk, and
the Keras student then trains against those cached soft targets.

Why not train both together in one process? Three reasons, all practical:

1. The teacher's forward pass is the expensive part. Caching means it runs once
   instead of once per student epoch — and this project trains several students.
2. It keeps PyTorch and TensorFlow out of the same process and off the same GPU
   at the same time, which on a 6 GB laptop card is the difference between
   working and an out-of-memory crash.
3. The student training loop stays plain `model.fit`, so it is easy to read.

The loss follows Hinton et al. (2015): a temperature-softened KL term against
the teacher plus the ordinary hard-label cross-entropy, mixed by `alpha`. Note
the `T^2` factor — without it, the softened gradients shrink as temperature
rises and the distillation term quietly stops contributing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

DEFAULT_TEACHER = "distilbert-base-uncased"


@dataclass
class DistillConfig:
    """Hyper-parameters for the distillation run."""

    temperature: float = 4.0
    alpha: float = 0.9  # weight on the soft (teacher) term
    teacher_name: str = DEFAULT_TEACHER
    max_length: int = 256

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be positive, got {self.temperature}")


# ---------------------------------------------------------------------------
# Cached teacher logits
# ---------------------------------------------------------------------------
def logits_path(split: str, config: DistillConfig | None = None) -> Path:
    config = config or DistillConfig()
    safe = config.teacher_name.replace("/", "__")
    return cfg.MODEL_DIR / "teacher_logits" / f"{safe}__{split}.npz"


def save_teacher_logits(
    split: str,
    complaint_ids: np.ndarray,
    logits: np.ndarray,
    config: DistillConfig | None = None,
) -> Path:
    """Persist teacher logits keyed by complaint id.

    Storing the ids alongside the logits is what makes this safe: the student
    can assert alignment rather than trusting that two files happen to share a
    row order.
    """
    path = logits_path(split, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, complaint_id=complaint_ids, logits=logits)
    log.info("Wrote %s teacher logits to %s", f"{len(logits):,}", path)
    return path


def load_teacher_logits(
    split: str, complaint_ids: np.ndarray, config: DistillConfig | None = None
) -> np.ndarray:
    """Load cached logits and reorder them to match `complaint_ids` exactly."""
    path = logits_path(split, config)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/train_teacher.py` first."
        )
    cached = np.load(path, allow_pickle=False)
    cached_ids = cached["complaint_id"].astype(str)
    logits = cached["logits"]

    position = {cid: index for index, cid in enumerate(cached_ids)}
    wanted = np.asarray(complaint_ids).astype(str)
    missing = [cid for cid in wanted if cid not in position]
    if missing:
        raise KeyError(
            f"{len(missing)} complaint ids have no cached teacher logits "
            f"(first few: {missing[:5]}). Re-run the teacher over this split."
        )
    order = np.fromiter((position[cid] for cid in wanted), dtype=np.int64, count=len(wanted))
    return logits[order]


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def make_distillation_loss(config: DistillConfig):
    """Build the combined soft/hard loss.

    The target tensor is the hard label's one-hot concatenated with the
    teacher's logits, i.e. shape `(batch, 2 * n_classes)`. Packing both into one
    tensor means the student trains with an unmodified `model.fit`, with no
    custom `train_step` and no backend-specific code.
    """
    import keras  # noqa: PLC0415
    from keras import ops  # noqa: PLC0415

    temperature = config.temperature
    alpha = config.alpha

    def loss_fn(y_true, y_pred):
        n_classes = ops.shape(y_pred)[-1]
        hard_target = y_true[:, :n_classes]
        teacher_logits = y_true[:, n_classes:]

        hard_loss = keras.losses.categorical_crossentropy(
            hard_target, y_pred, from_logits=True
        )

        teacher_soft = ops.softmax(teacher_logits / temperature, axis=-1)
        student_log_soft = ops.log_softmax(y_pred / temperature, axis=-1)
        soft_loss = -ops.sum(teacher_soft * student_log_soft, axis=-1)

        # T^2 keeps the soft term's gradient magnitude comparable to the hard
        # term's as temperature rises.
        return alpha * (temperature**2) * soft_loss + (1.0 - alpha) * hard_loss

    loss_fn.__name__ = f"distill_T{temperature:g}_a{alpha:g}"
    return loss_fn


def pack_targets(
    y_index: np.ndarray, teacher_logits: np.ndarray, n_classes: int
) -> np.ndarray:
    """Concatenate one-hot hard labels with teacher logits for `make_distillation_loss`."""
    if len(y_index) != len(teacher_logits):
        raise ValueError(
            f"label/logit length mismatch: {len(y_index)} vs {len(teacher_logits)}"
        )
    if teacher_logits.shape[1] != n_classes:
        raise ValueError(
            f"teacher produced {teacher_logits.shape[1]} classes, expected {n_classes}"
        )
    one_hot = np.zeros((len(y_index), n_classes), dtype="float32")
    one_hot[np.arange(len(y_index)), y_index] = 1.0
    return np.concatenate([one_hot, teacher_logits.astype("float32")], axis=1)


def compile_student(student, config: DistillConfig, learning_rate: float = 1e-3):
    """Compile the student with the distillation loss.

    Accuracy is intentionally *not* compiled in as a metric: Keras would try to
    compare the double-width target tensor against the student's logits and
    report nonsense. Student accuracy is measured explicitly after training.
    """
    import keras  # noqa: PLC0415

    student.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=make_distillation_loss(config),
    )
    return student
