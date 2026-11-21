"""Keras neural models: feed-forward, convolutional and recurrent.

All three core architectures the role asks about are built here, over the same
corpus, so the comparison is meaningful:

* **Feed-forward** over a TF-IDF bag — the neural analogue of the linear
  baseline. Establishes how much of the task is pure vocabulary.
* **1-D convolutional** over learned embeddings — picks up local phrase
  patterns ("loan modification", "charged twice") regardless of position.
* **Bidirectional recurrent** over the same embeddings — adds order and
  long-range context, at a real cost in latency.

Tokenisation is kept *outside* the exported model on purpose. Keras's
`TextVectorization` relies on string ops that do not survive TFLite conversion
cleanly, and the Java service needs to tokenise anyway. So the contract between
Python and production is: **integer sequences in, logits out**, with the
vocabulary shipped alongside as JSON.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import config as cfg

log = logging.getLogger(__name__)

MAX_TOKENS = 30_000
SEQUENCE_LENGTH = 256


def _keras():
    """Import Keras lazily so the classical path needs no TensorFlow install."""
    import keras  # noqa: PLC0415

    return keras


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------
@dataclass
class TextEncoder:
    """Adapted `TextVectorization`, plus vocabulary export for the Java service."""

    layer: object
    max_tokens: int = MAX_TOKENS
    sequence_length: int = SEQUENCE_LENGTH

    @classmethod
    def fit(
        cls,
        texts: Sequence[str],
        max_tokens: int = MAX_TOKENS,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> "TextEncoder":
        keras = _keras()
        layer = keras.layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=sequence_length,
            standardize="lower_and_strip_punctuation",
            split="whitespace",
        )
        layer.adapt(list(texts))
        return cls(layer=layer, max_tokens=max_tokens, sequence_length=sequence_length)

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self.layer(list(texts)), dtype="int32")

    @property
    def vocabulary(self) -> list[str]:
        return [str(t) for t in self.layer.get_vocabulary()]

    def save_vocabulary(self, directory: Path) -> dict[str, Path]:
        """Export the vocabulary and the tokenisation contract.

        Three files, deliberately:

        * ``vocab.txt`` — one token per line, index == line number. The same
          convention BERT ships, and readable from any language with no JSON
          dependency at all. This is what the Java service loads.
        * ``tokenizer.properties`` — the contract (sequence length, OOV and pad
          indices, standardisation and split rules) in a format the JVM parses
          natively.
        * ``tokenizer.json`` — the same information for humans and for Python.

        Writing the *rules* and not just the vocabulary is the point. Train/serve
        skew in text models almost always comes from the serving tokeniser
        quietly disagreeing with the training one about case, punctuation or
        truncation. Recording the rules lets the Java test suite assert
        agreement instead of hoping for it.
        """
        directory.mkdir(parents=True, exist_ok=True)
        vocabulary = self.vocabulary

        vocab_path = directory / "vocab.txt"
        # Newlines inside a token would corrupt a line-indexed file; the Keras
        # whitespace splitter cannot produce one, so assert rather than escape.
        bad = [t for t in vocabulary if "\n" in t or "\r" in t]
        if bad:
            raise ValueError(f"vocabulary tokens contain newlines: {bad[:3]}")
        vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")

        contract = {
            "sequence_length": self.sequence_length,
            "max_tokens": self.max_tokens,
            "vocabulary_size": len(vocabulary),
            "pad_index": 0,
            "oov_index": 1,
            "standardize": "lower_and_strip_punctuation",
            "split": "whitespace",
            "truncation": "post",
            "padding": "post",
        }

        props_path = directory / "tokenizer.properties"
        props_path.write_text(
            "\n".join(f"{key}={value}" for key, value in contract.items()) + "\n",
            encoding="utf-8",
        )

        json_path = directory / "tokenizer.json"
        json_path.write_text(json.dumps({**contract, "vocabulary": vocabulary}, indent=2))

        return {"vocab": vocab_path, "properties": props_path, "json": json_path}


def build_tfidf_encoder(texts: Sequence[str], max_tokens: int = MAX_TOKENS):
    """A separate `TextVectorization` in TF-IDF mode, for the feed-forward net."""
    keras = _keras()
    layer = keras.layers.TextVectorization(
        max_tokens=max_tokens, output_mode="tf_idf", standardize="lower_and_strip_punctuation"
    )
    layer.adapt(list(texts))
    return layer


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
def build_mlp(input_dim: int, n_classes: int, hidden: tuple[int, ...] = (512, 256)):
    """Feed-forward network over a TF-IDF bag-of-words vector."""
    keras = _keras()
    layers = [keras.layers.Input(shape=(input_dim,), name="tfidf")]
    for index, units in enumerate(hidden):
        layers.append(keras.layers.Dense(units, activation="relu", name=f"dense_{index}"))
        layers.append(keras.layers.Dropout(0.3, name=f"dropout_{index}"))
    layers.append(keras.layers.Dense(n_classes, name="logits"))
    model = keras.Sequential(layers, name="tfidf_mlp")
    return model


def build_text_cnn(
    vocab_size: int,
    n_classes: int,
    sequence_length: int = SEQUENCE_LENGTH,
    embed_dim: int = 128,
    kernel_sizes: tuple[int, ...] = (3, 4, 5),
    filters: int = 128,
):
    """Multi-width 1-D CNN over learned embeddings.

    Parallel kernel widths act as a learned n-gram detector bank; global max
    pooling makes the result position-invariant, which suits complaints where
    the decisive phrase can appear anywhere in a long narrative.
    """
    keras = _keras()
    inputs = keras.layers.Input(shape=(sequence_length,), dtype="int32", name="tokens")
    embedded = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, mask_zero=False, name="embedding"
    )(inputs)
    embedded = keras.layers.SpatialDropout1D(0.2)(embedded)

    branches = []
    for width in kernel_sizes:
        conv = keras.layers.Conv1D(
            filters, width, activation="relu", padding="same", name=f"conv_{width}"
        )(embedded)
        branches.append(keras.layers.GlobalMaxPooling1D(name=f"pool_{width}")(conv))

    merged = keras.layers.Concatenate(name="concat")(branches)
    hidden = keras.layers.Dropout(0.4)(merged)
    hidden = keras.layers.Dense(128, activation="relu", name="dense")(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    outputs = keras.layers.Dense(n_classes, name="logits")(hidden)
    return keras.Model(inputs, outputs, name="text_cnn")


def build_bilstm(
    vocab_size: int,
    n_classes: int,
    sequence_length: int = SEQUENCE_LENGTH,
    embed_dim: int = 128,
    units: int = 96,
):
    """Bidirectional LSTM over learned embeddings."""
    keras = _keras()
    inputs = keras.layers.Input(shape=(sequence_length,), dtype="int32", name="tokens")
    embedded = keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, mask_zero=True, name="embedding"
    )(inputs)
    embedded = keras.layers.SpatialDropout1D(0.2)(embedded)
    recurrent = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=False), name="bilstm"
    )(embedded)
    hidden = keras.layers.Dropout(0.4)(recurrent)
    hidden = keras.layers.Dense(128, activation="relu", name="dense")(hidden)
    outputs = keras.layers.Dense(n_classes, name="logits")(hidden)
    return keras.Model(inputs, outputs, name="bilstm")


def build_student(
    vocab_size: int,
    n_classes: int,
    sequence_length: int = SEQUENCE_LENGTH,
    embed_dim: int = 64,
    filters: int = 96,
):
    """Deliberately small CNN used as the distillation student.

    Roughly an order of magnitude fewer parameters than the CNN above: narrower
    embeddings, one kernel width, no dense bottleneck. On its own it
    underperforms; trained against the teacher's soft targets it recovers most
    of the gap, which is the entire point of the distillation notebook.
    """
    keras = _keras()
    inputs = keras.layers.Input(shape=(sequence_length,), dtype="int32", name="tokens")
    embedded = keras.layers.Embedding(vocab_size, embed_dim, name="embedding")(inputs)
    conv = keras.layers.Conv1D(filters, 4, activation="relu", padding="same", name="conv")(
        embedded
    )
    pooled = keras.layers.GlobalMaxPooling1D(name="pool")(conv)
    hidden = keras.layers.Dropout(0.3)(pooled)
    outputs = keras.layers.Dense(n_classes, name="logits")(hidden)
    return keras.Model(inputs, outputs, name="student_cnn")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def class_weights(labels: Sequence[str], classes: Sequence[str] = cfg.CANONICAL_LABELS) -> dict[int, float]:
    """Inverse-frequency weights, keyed by class index."""
    counts = {label: 0 for label in classes}
    for label in labels:
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    n_classes = sum(1 for value in counts.values() if value)
    return {
        index: (total / (n_classes * count)) if count else 0.0
        for index, (_, count) in enumerate(counts.items())
    }


def compile_model(model, learning_rate: float = 1e-3):
    """Compile for integer targets with logit outputs."""
    keras = _keras()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    return model


def default_callbacks(patience: int = 3, checkpoint: Path | None = None):
    keras = _keras()
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1),
    ]
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                str(checkpoint), monitor="val_loss", save_best_only=True
            )
        )
    return callbacks
