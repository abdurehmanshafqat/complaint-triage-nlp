"""Export and quantise the trained student for CPU serving.

What "quantisation" can mean here depends on the model's input. This student
takes **token indices**, not continuous features, so the usual full-integer
recipe does not apply unchanged:

* The embedding lookup consumes `int32` indices. Forcing the *input* tensor to
  `int8` would be meaningless — index 7 is not a quantity to be rescaled — so
  the input dtype stays `int32` in every variant below.
* What does get quantised is the weights and, for the full-integer path, the
  intermediate activations. The embedding matrix dominates this model's size,
  and it quantises well, which is why the saving is large.

Three variants are produced so the trade-off is measured rather than asserted:

| variant   | weights | activations | typical use                     |
| --------- | ------- | ----------- | ------------------------------- |
| `float32` | fp32    | fp32        | reference accuracy              |
| `float16` | fp16    | fp32        | ~2x smaller, lossless in practice |
| `int8`    | int8    | fp32        | ~4x smaller, small accuracy cost |

A second export to ONNX feeds the Java service, so the JVM side does not need
a TensorFlow dependency.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Literal

import numpy as np

log = logging.getLogger(__name__)

Variant = Literal["float32", "float16", "int8"]


@dataclass
class ExportResult:
    """Size and accuracy of one exported artefact."""

    variant: str
    path: Path
    size_bytes: int
    macro_f1: float | None = None
    accuracy: float | None = None
    latency_ms_per_item: float | None = None

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1e6


# ---------------------------------------------------------------------------
# TFLite
# ---------------------------------------------------------------------------
def _representative_dataset(
    token_batches: np.ndarray, limit: int = 300
) -> Callable[[], Iterator[list[np.ndarray]]]:
    """Yield single examples for the converter to calibrate activation ranges."""

    def generator() -> Iterator[list[np.ndarray]]:
        for row in token_batches[:limit]:
            yield [np.expand_dims(row.astype("int32"), axis=0)]

    return generator


def export_tflite(
    model,
    path: Path,
    variant: Variant = "int8",
    representative_tokens: np.ndarray | None = None,
) -> ExportResult:
    """Convert a Keras model to TFLite under one quantisation variant."""
    import tensorflow as tf  # noqa: PLC0415

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if variant == "float32":
        pass
    elif variant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif variant == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_tokens is not None:
            # Calibrating on real token sequences lets the converter quantise
            # activations too, not just weights.
            converter.representative_dataset = _representative_dataset(
                representative_tokens
            )
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    blob = converter.convert()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    log.info("Wrote %s (%.2f MB)", path.name, len(blob) / 1e6)
    return ExportResult(variant=variant, path=path, size_bytes=len(blob))


class TFLiteRunner:
    """Minimal single-example inference wrapper around a `.tflite` file."""

    def __init__(self, path: Path) -> None:
        try:
            import tflite_runtime.interpreter as tflite  # noqa: PLC0415

            self._interpreter = tflite.Interpreter(model_path=str(path))
        except ImportError:
            import tensorflow as tf  # noqa: PLC0415

            self._interpreter = tf.lite.Interpreter(model_path=str(path))

        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]
        self._output = self._interpreter.get_output_details()[0]

    def predict_logits(self, tokens: np.ndarray) -> np.ndarray:
        """Score one sequence. Shape `(sequence_length,)` in, `(n_classes,)` out."""
        batch = np.expand_dims(tokens.astype(self._input["dtype"]), axis=0)
        self._interpreter.set_tensor(self._input["index"], batch)
        self._interpreter.invoke()
        return self._interpreter.get_tensor(self._output["index"])[0]

    def predict_batch(self, tokens: np.ndarray, progress: bool = False) -> np.ndarray:
        """Score many sequences one at a time.

        TFLite interpreters are fixed-batch by default; looping is the honest
        way to measure the serving path rather than a vectorised shortcut.
        """
        rows = [self.predict_logits(row) for row in tokens]
        return np.vstack(rows)


# ---------------------------------------------------------------------------
# ONNX (for the Java service)
# ---------------------------------------------------------------------------
def export_onnx(
    model, path: Path, sequence_length: int, opset: int = 17
) -> ExportResult:
    """Export to ONNX so the JVM can serve the model without TensorFlow."""
    import tensorflow as tf  # noqa: PLC0415
    import tf2onnx  # noqa: PLC0415

    signature = (
        tf.TensorSpec((None, sequence_length), tf.int32, name="tokens"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tf2onnx.convert.from_keras(
        model, input_signature=signature, opset=opset, output_path=str(path)
    )
    size = path.stat().st_size
    log.info("Wrote %s (%.2f MB)", path.name, size / 1e6)
    return ExportResult(variant="onnx-float32", path=path, size_bytes=size)


def quantize_onnx(source: Path, dest: Path) -> ExportResult:
    """Dynamic INT8 quantisation of an ONNX graph."""
    from onnxruntime.quantization import QuantType, quantize_dynamic  # noqa: PLC0415

    dest.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(source), model_output=str(dest), weight_type=QuantType.QInt8
    )
    size = dest.stat().st_size
    log.info("Wrote %s (%.2f MB)", dest.name, size / 1e6)
    return ExportResult(variant="onnx-int8", path=dest, size_bytes=size)


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------
def time_runner(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    tokens: np.ndarray,
    repeats: int = 3,
    warmup: int = 5,
) -> float:
    """Median milliseconds per single-item call."""
    for row in tokens[:warmup]:
        predict_fn(row)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for row in tokens:
            predict_fn(row)
        timings.append((time.perf_counter() - start) / len(tokens) * 1000)
    return float(np.median(timings))
